"""Compare the promptfoo llm-rubric judge against data/human_labels.jsonl.

Per the architecture split in README.md section 2, the actual grading logic
must stay inside promptfoo (llm-rubric + a pinned provider) -- this module
never reimplements judging in Python. Two modes:

  - `run_id` given: cross-check `gradingResult.pass` already recorded in that
    run's output.json against each human_labels.jsonl case. No new API calls.
  - `run_id` omitted, answer_type=text: re-grade fresh via a throwaway
    promptfoo eval that replays each human_labels.jsonl `output_raw` string
    through promptfoo's built-in `echo` provider
    (https://www.promptfoo.dev/docs/providers/echo/, "returns the prompt
    as-is... useful ... to evaluate existing outputs without re-generating
    them") into the *same* llm-rubric assert build.py would generate. This
    still delegates grading to promptfoo.
  - `run_id` omitted, answer_type=label/json: production grading is a
    deterministic assert, so output_raw is replayed through the pinned Python
    ports of label_match.js / json_field_match.js instead -- no promptfoo
    round-trip, no LLM call, and no rubric file needed (issue #50).

In both modes, judge verdicts are joined back to human labels on the
(case_id, model_label) composite key -- never on case_id alone, since the
same case may carry one label per model (see docs/DESIGN.md section 5.3).

Iron rule #6: results/reports must show an explicit warning whenever the
judge is uncalibrated or below judge.agreement_threshold. This module writes
``results/<task>/calibration.json`` and stamps matching run ``meta.json``
files so ``run`` / ``report`` keep the status across runs (issue #100).
"""

from __future__ import annotations

import json
import statistics
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from evalloop import run as run_mod
from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import (
    Config,
    GoldenCase,
    HumanLabel,
    load_golden_jsonl,
    load_human_labels,
    parse_promptfoo_output,
)


class CalibrateError(RuntimeError):
    pass


def load_task_calibration(paths: TaskPaths, *, judge_provider: str | None = None) -> dict | None:
    """Load task-level calibration.json; None if missing or judge provider mismatches."""
    path = paths.calibration
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if judge_provider is not None and data.get("judge_provider") != judge_provider:
        return None
    return data


def save_task_calibration(paths: TaskPaths, config: Config, result: CalibrationResult) -> dict:
    """Persist calibration so later ``run`` / ``report`` pick up the status (issue #100)."""
    status = "calibrated" if result.status == "calibrated" else result.status
    payload = {
        "judge_provider": config.judge.provider,
        "calibration_status": status,
        "agreement_rate": result.agreement_rate,
        "agreement_threshold": result.threshold,
        "n_compared": result.n_compared,
        "n_skipped": result.n_skipped,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.calibration.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[calibrate] wrote {paths.calibration} status={status}")
    return payload


def uses_llm_rubric_judge(meta: dict) -> bool:
    """True when this run actually graded with llm-rubric (not label/json asserts)."""
    grader = meta.get("grader") or {}
    grader_type = grader.get("type") if isinstance(grader, dict) else None
    if grader_type == "llm-rubric":
        return True
    if grader_type in ("label-match", "json-field-match"):
        return False
    # Legacy metas without a grader block: text tasks used the LLM judge.
    return not grader and meta.get("answer_type") == "text"


def apply_calibration_to_meta(meta: dict, calibration: dict) -> bool:
    """Stamp judge/grader calibration fields from a task-level snapshot. Returns True if changed."""
    if not uses_llm_rubric_judge(meta):
        return False
    status = calibration.get("calibration_status")
    if not status:
        return False
    rate = calibration.get("agreement_rate")
    changed = False
    for key in ("judge", "grader"):
        block = meta.setdefault(key, {})
        if not isinstance(block, dict):
            continue
        if block.get("calibration_status") != status or block.get("agreement_rate") != rate:
            block["calibration_status"] = status
            block["agreement_rate"] = rate
            changed = True
    return changed


def apply_calibration_to_runs(paths: TaskPaths, calibration: dict) -> int:
    """Update meta.json for llm-rubric runs whose judge provider matches the snapshot."""
    provider = calibration.get("judge_provider")
    if not provider or not paths.runs_dir.is_dir():
        return 0
    updated = 0
    for meta_path in sorted(paths.runs_dir.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict) or not uses_llm_rubric_judge(meta):
            continue
        run_provider = (meta.get("grader") or {}).get("provider") or (meta.get("judge") or {}).get(
            "provider"
        )
        if run_provider != provider:
            continue
        if apply_calibration_to_meta(meta, calibration):
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            updated += 1
    if updated:
        print(f"[calibrate] updated judge.calibration_status on {updated} run meta.json file(s)")
    return updated


@dataclass
class CaseAgreement:
    case_id: str
    alias: str
    human_pass: bool
    judge_pass: bool | None
    agrees: bool | None


@dataclass
class CalibrationResult:
    agreement_rate: float | None
    n_compared: int
    n_skipped: int
    threshold: float
    status: str  # "calibrated" | "low_agreement" | "no_data"
    cases: list[CaseAgreement]


def _judge_verdicts_from_run(run_id: str, paths: TaskPaths) -> dict[tuple[str, str], bool]:
    output_path = paths.runs_dir / run_id / "output.json"
    if not output_path.exists():
        raise CalibrateError(f"run {run_id!r} has no output.json at {output_path}")
    parsed = parse_promptfoo_output(output_path)

    grouped: dict[tuple[str, str], list[bool]] = {}
    for r in parsed.results:
        if r.case_id is None or r.alias is None or r.passed is None:
            continue
        grouped.setdefault((r.case_id, r.alias), []).append(bool(r.passed))

    # repeat>1 collapses to a majority vote per (case_id, alias)
    return {key: (sum(flags) / len(flags)) >= 0.5 for key, flags in grouped.items()}


def _judge_verdicts_fresh(
    labels: list[HumanLabel], golden_by_id: dict[str, GoldenCase], cfg: Config, paths: TaskPaths
) -> dict[tuple[str, str], bool]:
    tests = []
    for label in labels:
        case = golden_by_id.get(label.case_id)
        if case is None:
            continue
        tests.append(
            {
                "description": f"{label.case_id}:{label.model_label}",
                "vars": {
                    "case_id": label.case_id,
                    # model_labelはプロンプトでは未使用だが、結果行をラベルへ
                    # (case_id, model_label)の複合キーで1:1にマッチバックする
                    # ために必須。case_id単独では同一caseの複数モデルラベルが
                    # 互いに上書きされ、一致率が壊れる（issue #6）
                    "model_label": label.model_label,
                    "output_raw": label.output_raw,
                    "input": case.input,
                    "expected": case.expected,
                },
            }
        )
    if not tests:
        return {}

    rubric_path = REPO_ROOT / cfg.judge.rubric_file
    if not rubric_path.exists():
        raise CalibrateError(
            f"rubric file not found: {rubric_path}\n"
            "Fresh re-grading requires a judge rubric file. "
            "If this task is not graded by an LLM judge, pass --run-id to cross-check an existing run instead."
        )
    promptfoo_config = {
        "description": "evalloop calibrate (echo replay)",
        "providers": [{"id": "echo", "label": "echo"}],
        "prompts": ["{{output_raw}}"],
        "defaultTest": {
            "assert": [
                {
                    "type": "llm-rubric",
                    # inline, not file:// -- see build.py's comment on why:
                    # file://-loaded llm-rubric values don't get Nunjucks
                    # substitution on {{input}}/{{expected}} in promptfoo 0.121.17
                    "value": rubric_path.read_text(encoding="utf-8"),
                    "provider": cfg.judge.provider,
                    "threshold": cfg.judge.threshold,
                }
            ]
        },
        "tests": tests,
    }

    tmp_name = f"_calibrate_{uuid.uuid4().hex[:8]}.yaml"
    tmp_config_path = paths.promptfoo_dir / tmp_name
    paths.promptfoo_dir.mkdir(parents=True, exist_ok=True)
    tmp_config_path.write_text(yaml.safe_dump(promptfoo_config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    try:
        with tempfile.TemporaryDirectory(prefix="evalloop-calibrate-") as tmp_dir:
            output_path = Path(tmp_dir) / "calibrate_output.json"
            # no timeout: re-grading many labels through a slow local judge can
            # legitimately take a while (see run.py's run_promptfoo_eval docstring)
            proc = run_mod.run_promptfoo_eval(tmp_config_path, output_path, repeat=1, no_cache=True)
            if not output_path.exists():
                raise CalibrateError(
                    f"fresh judge re-grading failed (exit {proc.returncode}); "
                    f"pass --run-id to cross-check an existing run instead.\nstderr:\n{proc.stderr}"
                )
            parsed = parse_promptfoo_output(output_path)
    finally:
        tmp_config_path.unlink(missing_ok=True)

    verdicts: dict[tuple[str, str], bool] = {}
    for r in parsed.results:
        row_vars = (
            r.raw.get("vars")
            or (r.raw.get("testCase") or {}).get("vars")
            or {}
        )
        case_id = row_vars.get("case_id") or r.case_id
        # each echo-replay row corresponds to exactly one human label; the
        # model_label var (e.g. "haiku45") identifies which one -- the echo
        # provider's own alias is useless here, and joining by case_id alone
        # would let multiple models on the same case overwrite each other
        model_label = row_vars.get("model_label")
        if case_id is None or model_label is None or r.passed is None:
            continue
        verdicts[(case_id, model_label)] = bool(r.passed)
    return verdicts


def _judge_verdicts_deterministic(
    labels: list[HumanLabel], golden_by_id: dict[str, GoldenCase], cfg: Config
) -> dict[tuple[str, str], bool]:
    """Fresh mode for label/json tasks: production grading is a deterministic
    assert (label_match.js / json_field_match.js), so replaying output_raw
    through the pinned Python ports of those asserts IS the judge -- no
    promptfoo round-trip and no LLM call. The ports are locked to the JS
    implementations by shared fixtures (tests/fixtures/*_cases.json).
    """
    # local import: optimize pulls in dspy, which plain calibrate runs don't need
    from evalloop.optimize import json_score_and_feedback, label_score_and_feedback

    verdicts: dict[tuple[str, str], bool] = {}
    for label in labels:
        case = golden_by_id.get(label.case_id)
        if case is None:
            continue
        if cfg.task.answer_type == "label":
            score, _feedback = label_score_and_feedback(label.output_raw, case.expected, cfg.task.labels)
        else:  # json
            score, _feedback = json_score_and_feedback(label.output_raw, case.expected)
        verdicts[(label.case_id, label.model_label)] = score >= 1.0
    return verdicts


def calibrate(config: Config, paths: TaskPaths, run_id: str | None = None) -> CalibrationResult:
    cfg = config
    labels = load_human_labels(paths.human_labels)
    if not labels:
        raise CalibrateError(f"{paths.human_labels} is empty; nothing to calibrate against")

    golden_by_id = {c.id: c for c in load_golden_jsonl(paths.golden)}

    if run_id is not None:
        verdicts = _judge_verdicts_from_run(run_id, paths)
    elif cfg.task.answer_type == "text":
        verdicts = _judge_verdicts_fresh(labels, golden_by_id, cfg, paths)
    else:
        # label/json grading is deterministic -- replay through the Python
        # ports of the production asserts instead of a promptfoo round-trip
        verdicts = _judge_verdicts_deterministic(labels, golden_by_id, cfg)

    cases: list[CaseAgreement] = []
    skipped = 0
    for label in labels:
        if label.case_id not in golden_by_id:
            print(f"[calibrate] WARNING: {label.case_id} not found in golden.jsonl; skipping")
            skipped += 1
            continue
        judge_pass = verdicts.get((label.case_id, label.model_label))
        if judge_pass is None:
            print(
                f"[calibrate] WARNING: no judge verdict found for case_id={label.case_id} "
                f"model_label={label.model_label}; skipping"
            )
            skipped += 1
            continue
        human_pass = label.human_verdict == "pass"
        cases.append(
            CaseAgreement(
                case_id=label.case_id,
                alias=label.model_label,
                human_pass=human_pass,
                judge_pass=judge_pass,
                agrees=(human_pass == judge_pass),
            )
        )

    if not cases:
        result = CalibrationResult(
            agreement_rate=None, n_compared=0, n_skipped=skipped, threshold=cfg.judge.agreement_threshold,
            status="no_data", cases=[],
        )
    else:
        rate = statistics.mean(1.0 if c.agrees else 0.0 for c in cases)
        status = "calibrated" if rate >= cfg.judge.agreement_threshold else "low_agreement"
        result = CalibrationResult(
            agreement_rate=rate, n_compared=len(cases), n_skipped=skipped,
            threshold=cfg.judge.agreement_threshold, status=status, cases=cases,
        )

    print(f"[calibrate] compared {result.n_compared} case(s), skipped {result.n_skipped}")
    if result.agreement_rate is not None:
        print(f"[calibrate] agreement rate: {result.agreement_rate:.1%} (threshold: {result.threshold:.1%})")
        if result.status == "low_agreement":
            print(
                f"[calibrate] WARNING: agreement {result.agreement_rate:.1%} is BELOW "
                f"judge.agreement_threshold ({result.threshold:.1%}). Reports will flag this judge as uncalibrated."
            )
        for c in result.cases:
            if not c.agrees:
                print(f"[calibrate]   mismatch: {c.case_id}/{c.alias} human={c.human_pass} judge={c.judge_pass}")
    else:
        print("[calibrate] WARNING: no comparable cases; calibration status left as 'no_data'")

    snapshot = save_task_calibration(paths, cfg, result)

    if run_id is not None:
        meta_path = paths.runs_dir / run_id / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.setdefault("judge", {})
            if meta.get("answer_type") == "text" or meta.get("grader", {}).get("type") == "llm-rubric":
                meta.setdefault(
                    "grader",
                    {"type": "llm-rubric", "provider": meta["judge"].get("provider")},
                )
            apply_calibration_to_meta(meta, snapshot)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[calibrate] updated {meta_path} "
                f"judge.calibration_status={meta['judge']['calibration_status']}"
            )

    apply_calibration_to_runs(paths, snapshot)
    return result
