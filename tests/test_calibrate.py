import json

import pytest
import yaml

from evalloop import calibrate as calibrate_mod
from evalloop import run as run_mod
from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import (
    BlogConfig,
    Config,
    JudgeConfig,
    ModelConfig,
    OptimizeConfig,
    RunConfig,
    TaskConfig,
)


def _write_golden(path, cases):
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def _write_human_labels(path, labels):
    with path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")


def _make_config(rubric_file, answer_type="text", labels=None):
    return Config(
        task=TaskConfig(
            name="t1",
            answer_type=answer_type,
            prompt_file="tasks/sample-inquiry/prompts/task.txt",
            labels=labels or [],
        ),
        models=[
            ModelConfig(provider="ollama:chat:qwen2.5:7b", alias="qwen7b", tier="local"),
            ModelConfig(provider="anthropic:messages:claude-haiku-4-5-20251001", alias="haiku45", tier="small"),
        ],
        run=RunConfig(),
        judge=JudgeConfig(
            provider="anthropic:messages:claude-sonnet-4-6",
            threshold=0.8,
            agreement_threshold=0.85,
            rubric_file=str(rubric_file),
        ),
        optimize=OptimizeConfig(target_alias="qwen7b", reflection_provider="r"),
        blog=BlogConfig(),
        path=REPO_ROOT / "config.yaml",
    )


@pytest.fixture
def calibrate_env(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    paths.task_dir.mkdir(parents=True)

    _write_golden(
        paths.golden,
        [
            {"id": "case-0001", "input": "x", "expected": "契約照会", "split": "test", "meta": {"category": "基本", "source": "self-made"}},
            {"id": "case-0002", "input": "y", "expected": "障害報告", "split": "test", "meta": {"category": "基本", "source": "self-made"}},
            {"id": "case-0003", "input": "z", "expected": "機能要望", "split": "test", "meta": {"category": "基本", "source": "self-made"}},
        ],
    )

    # fresh re-grading reads the rubric via cfg.judge.rubric_file; scaffold one
    # with the {{input}}/{{expected}} placeholders promptfoo would substitute
    paths.rubric_file.parent.mkdir(parents=True)
    paths.rubric_file.write_text("問い合わせ: {{input}}\n期待: {{expected}}\nを採点してください。\n", encoding="utf-8")

    return {"paths": paths, "cfg": _make_config(paths.rubric_file)}


_JUDGE = "anthropic:messages:claude-sonnet-4-6"


def _write_run_output(paths, run_id, rows, judge_provider=_JUDGE):
    run_dir = paths.runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "output.json").write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "answer_type": "text",
                "grader": {
                    "type": "llm-rubric",
                    "provider": judge_provider,
                    "calibration_status": "uncalibrated",
                    "agreement_rate": None,
                },
                "judge": {
                    "provider": judge_provider,
                    "calibration_status": "uncalibrated",
                    "agreement_rate": None,
                },
            }
        ),
        encoding="utf-8",
    )


def _row(case_id, alias, passed):
    return {
        "vars": {"case_id": case_id, "expected": "x", "category": "基本"},
        "provider": {"id": "p", "label": alias},
        "response": {"output": "o"},
        "gradingResult": {"pass": passed, "score": 1 if passed else 0, "reason": "r"},
        "success": passed,
    }


def test_calibrate_fresh_mode_label_task_replays_deterministically(calibrate_env, monkeypatch):
    """label tasks are graded by a deterministic assert, so fresh calibration
    replays output_raw through the Python port of label_match.js -- no
    promptfoo round-trip and no rubric file needed (issue #50)."""
    paths = calibrate_env["paths"]
    labels4 = ["契約照会", "障害報告", "機能要望", "その他"]
    cfg = _make_config(rubric_file="does-not-exist.txt", answer_type="label", labels=labels4)
    _write_human_labels(
        paths.human_labels,
        [
            # exact label match -> judge pass, human pass -> agree
            {"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"},
            # containment fallback -> judge pass, human pass -> agree
            {"case_id": "case-0002", "model_label": "haiku45", "output_raw": "回答: 障害報告 です", "human_verdict": "pass"},
            # wrong label (golden expects 機能要望) -> judge fail, human fail -> agree
            {"case_id": "case-0003", "model_label": "haiku45", "output_raw": "その他", "human_verdict": "fail"},
        ],
    )

    def _no_promptfoo(*args, **kwargs):
        raise AssertionError("deterministic fresh calibration must not shell out to promptfoo")

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", _no_promptfoo)

    result = calibrate_mod.calibrate(cfg, paths, run_id=None)

    assert result.n_compared == 3
    assert result.agreement_rate == pytest.approx(1.0)
    assert result.status == "calibrated"


def test_calibrate_run_id_mode_high_agreement(calibrate_env):
    paths, cfg = calibrate_env["paths"], calibrate_env["cfg"]
    _write_human_labels(
        paths.human_labels,
        [
            {"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"},
            {"case_id": "case-0002", "model_label": "haiku45", "output_raw": "障害報告", "human_verdict": "pass"},
            {"case_id": "case-0003", "model_label": "haiku45", "output_raw": "その他", "human_verdict": "fail"},
        ],
    )
    _write_run_output(
        paths,
        "run-1",
        [
            _row("case-0001", "haiku45", True),
            _row("case-0002", "haiku45", True),
            _row("case-0003", "haiku45", False),
        ],
    )
    # Sibling run with the same judge should also pick up the task-level stamp.
    _write_run_output(paths, "run-sibling", [_row("case-0001", "haiku45", True)])
    _write_run_output(
        paths,
        "run-other-judge",
        [_row("case-0001", "haiku45", True)],
        judge_provider="ollama:chat:other",
    )
    # Non-LLM graders share judge.provider in meta but must keep not_applicable.
    label_dir = paths.runs_dir / "run-label"
    label_dir.mkdir(parents=True)
    (label_dir / "output.json").write_text(
        json.dumps({"results": {"results": [_row("case-0001", "haiku45", True)]}}),
        encoding="utf-8",
    )
    (label_dir / "meta.json").write_text(
        json.dumps(
            {
                "answer_type": "label",
                "grader": {"type": "label-match", "calibration_status": "not_applicable"},
                "judge": {
                    "provider": _JUDGE,
                    "calibration_status": "not_applicable",
                    "agreement_rate": None,
                },
            }
        ),
        encoding="utf-8",
    )

    result = calibrate_mod.calibrate(cfg, paths, run_id="run-1")

    assert result.n_compared == 3
    assert result.agreement_rate == pytest.approx(1.0)
    assert result.status == "calibrated"

    meta = json.loads((paths.runs_dir / "run-1" / "meta.json").read_text(encoding="utf-8"))
    assert meta["judge"]["calibration_status"] == "calibrated"
    assert meta["judge"]["agreement_rate"] == pytest.approx(1.0)
    assert meta["grader"]["calibration_status"] == "calibrated"
    assert meta["grader"]["agreement_rate"] == pytest.approx(1.0)

    snap = json.loads(paths.calibration.read_text(encoding="utf-8"))
    assert snap["judge_provider"] == _JUDGE
    assert snap["calibration_status"] == "calibrated"
    assert snap["agreement_rate"] == pytest.approx(1.0)

    sibling = json.loads((paths.runs_dir / "run-sibling" / "meta.json").read_text(encoding="utf-8"))
    assert sibling["judge"]["calibration_status"] == "calibrated"
    other = json.loads((paths.runs_dir / "run-other-judge" / "meta.json").read_text(encoding="utf-8"))
    assert other["judge"]["calibration_status"] == "uncalibrated"
    label_meta = json.loads((label_dir / "meta.json").read_text(encoding="utf-8"))
    assert label_meta["grader"]["calibration_status"] == "not_applicable"
    assert label_meta["judge"]["calibration_status"] == "not_applicable"


def test_calibrate_run_id_mode_low_agreement_warns(calibrate_env):
    paths, cfg = calibrate_env["paths"], calibrate_env["cfg"]
    _write_human_labels(
        paths.human_labels,
        [
            {"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"},
            {"case_id": "case-0002", "model_label": "haiku45", "output_raw": "障害報告", "human_verdict": "fail"},
            {"case_id": "case-0003", "model_label": "haiku45", "output_raw": "その他", "human_verdict": "fail"},
        ],
    )
    # judge disagrees with human on 2 of 3
    _write_run_output(
        paths,
        "run-2",
        [
            _row("case-0001", "haiku45", True),
            _row("case-0002", "haiku45", True),
            _row("case-0003", "haiku45", True),
        ],
    )

    result = calibrate_mod.calibrate(cfg, paths, run_id="run-2")

    assert result.agreement_rate == pytest.approx(1 / 3)
    assert result.status == "low_agreement"

    meta = json.loads((paths.runs_dir / "run-2" / "meta.json").read_text(encoding="utf-8"))
    assert meta["judge"]["calibration_status"] == "low_agreement"
    assert meta["grader"]["calibration_status"] == "low_agreement"


def test_calibrate_skips_case_missing_from_golden(calibrate_env):
    paths, cfg = calibrate_env["paths"], calibrate_env["cfg"]
    _write_human_labels(
        paths.human_labels,
        [{"case_id": "case-9999", "model_label": "haiku45", "output_raw": "x", "human_verdict": "pass"}],
    )
    _write_run_output(paths, "run-3", [])

    result = calibrate_mod.calibrate(cfg, paths, run_id="run-3")

    assert result.n_compared == 0
    assert result.n_skipped == 1
    assert result.status == "no_data"


def test_calibrate_skips_case_missing_judge_verdict(calibrate_env):
    paths, cfg = calibrate_env["paths"], calibrate_env["cfg"]
    _write_human_labels(
        paths.human_labels,
        [
            {"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"},
            {"case_id": "case-0002", "model_label": "haiku45", "output_raw": "障害報告", "human_verdict": "pass"},
        ],
    )
    # only case-0001 has a matching judge verdict in the run
    _write_run_output(paths, "run-4", [_row("case-0001", "haiku45", True)])

    result = calibrate_mod.calibrate(cfg, paths, run_id="run-4")

    assert result.n_compared == 1
    assert result.n_skipped == 1


def test_calibrate_missing_run_raises(calibrate_env):
    paths, cfg = calibrate_env["paths"], calibrate_env["cfg"]
    _write_human_labels(
        paths.human_labels,
        [{"case_id": "case-0001", "model_label": "haiku45", "output_raw": "x", "human_verdict": "pass"}],
    )
    with pytest.raises(calibrate_mod.CalibrateError):
        calibrate_mod.calibrate(cfg, paths, run_id="does-not-exist")


def test_calibrate_empty_human_labels_raises(calibrate_env):
    paths, cfg = calibrate_env["paths"], calibrate_env["cfg"]
    paths.human_labels.write_text("", encoding="utf-8")
    with pytest.raises(calibrate_mod.CalibrateError):
        calibrate_mod.calibrate(cfg, paths, run_id="whatever")


def test_judge_verdicts_from_run_majority_vote_across_repeats(calibrate_env):
    paths = calibrate_env["paths"]
    _write_run_output(
        paths,
        "run-5",
        [
            _row("case-0001", "haiku45", True),
            _row("case-0001", "haiku45", True),
            _row("case-0001", "haiku45", False),
        ],
    )
    verdicts = calibrate_mod._judge_verdicts_from_run("run-5", paths)
    assert verdicts[("case-0001", "haiku45")] is True


def test_fresh_mode_two_models_same_case_counted_independently(calibrate_env, monkeypatch):
    """Regression for issue #6: fresh mode matched echo-replay verdicts back by
    case_id alone, so two models labeled on the same case silently overwrote
    each other (the last result row won for both keys). Verdicts must join on
    the (case_id, model_label) composite key.
    """
    paths, cfg = calibrate_env["paths"], calibrate_env["cfg"]
    _write_human_labels(
        paths.human_labels,
        [
            # 同一case-0001を2モデルでラベル: haiku45は正解(pass)、qwen7bは不正解(fail)
            {"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"},
            {"case_id": "case-0001", "model_label": "qwen7b", "output_raw": "その他", "human_verdict": "fail"},
        ],
    )

    def fake_eval(config_path, output_path, **kwargs):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # echo provider replays output_raw; judge passes only the correct answer
        rows = []
        for test in config["tests"]:
            passed = test["vars"]["output_raw"] == "契約照会"
            rows.append(
                {
                    "vars": dict(test["vars"]),
                    "provider": {"id": "echo", "label": "echo"},
                    "response": {"output": test["vars"]["output_raw"]},
                    "gradingResult": {"pass": passed, "score": 1 if passed else 0},
                    "success": passed,
                }
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    result = calibrate_mod.calibrate(cfg, paths, run_id=None)

    # 2ラベルとも独立に判定される: judge=pass/human=pass と judge=fail/human=fail で一致率100%
    assert result.n_compared == 2
    assert result.agreement_rate == pytest.approx(1.0)
    by_key = {(c.case_id, c.alias): c for c in result.cases}
    assert by_key[("case-0001", "haiku45")].judge_pass is True
    assert by_key[("case-0001", "qwen7b")].judge_pass is False


def test_run_id_mode_two_models_same_case_counted_independently(calibrate_env):
    """Cross-check mode already joins on (case_id, alias); pin that behavior."""
    paths, cfg = calibrate_env["paths"], calibrate_env["cfg"]
    _write_human_labels(
        paths.human_labels,
        [
            {"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"},
            {"case_id": "case-0001", "model_label": "qwen7b", "output_raw": "その他", "human_verdict": "fail"},
        ],
    )
    _write_run_output(
        paths,
        "run-6",
        [
            _row("case-0001", "haiku45", True),
            _row("case-0001", "qwen7b", False),
        ],
    )

    result = calibrate_mod.calibrate(cfg, paths, run_id="run-6")

    assert result.n_compared == 2
    assert result.agreement_rate == pytest.approx(1.0)


def test_fresh_judge_config_uses_echo_provider_and_pinned_judge(calibrate_env, monkeypatch):
    """Structural check of the 'no run_id' path: build.py's iron-rule-#2 judge
    pin must flow into the throwaway echo-replay config, without ever calling
    a real API (run_promptfoo_eval is monkeypatched out).
    """
    paths, cfg = calibrate_env["paths"], calibrate_env["cfg"]
    _write_human_labels(
        paths.human_labels,
        [{"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"}],
    )

    captured = {}

    def fake_eval(config_path, output_path, **kwargs):
        captured["config"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "results": {
                        "results": [
                            {
                                "vars": {"case_id": "case-0001", "model_label": "haiku45"},
                                "provider": {"id": "echo", "label": "echo"},
                                "response": {"output": "契約照会"},
                                "gradingResult": {"pass": True, "score": 1},
                                "success": True,
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    result = calibrate_mod.calibrate(cfg, paths, run_id=None)

    assert captured["config"]["providers"] == [{"id": "echo", "label": "echo"}]
    assert captured["config"]["prompts"] == ["{{output_raw}}"]
    rubric_assert = captured["config"]["defaultTest"]["assert"][0]
    assert rubric_assert["type"] == "llm-rubric"
    assert rubric_assert["provider"] == "anthropic:messages:claude-sonnet-4-6"
    assert result.n_compared == 1
    assert result.agreement_rate == pytest.approx(1.0)
    # the throwaway config must not be left behind
    assert not list(paths.promptfoo_dir.glob("_calibrate_*.yaml"))


def test_fresh_mode_missing_rubric_raises_clear_error(calibrate_env):
    paths, cfg = calibrate_env["paths"], calibrate_env["cfg"]
    _write_human_labels(
        paths.human_labels,
        [{"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"}],
    )
    paths.rubric_file.unlink()

    with pytest.raises(calibrate_mod.CalibrateError, match=r"rubric file not found: [\s\S]*--run-id"):
        calibrate_mod.calibrate(cfg, paths, run_id=None)
