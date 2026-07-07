"""Prompt optimization orchestration: golden.jsonl split=='train' -> optimized
prompt -> promptfoo variant config -> automatic run/report/compare.

Method-specific code lives in the evalloop.optimizers package: the shared
contract in optimizers/base.py, the GEPA implementation in optimizers/gepa.py,
and the deterministic proxy metrics + template round-trip helpers in
optimizers/metrics.py (see its module docstring for why training uses a proxy
metric instead of the final promptfoo judge). This module keeps optimizer
selection (currently GEPA only), variant generation, and `compare`.

Iron rules enforced here:
    1. split separation: this module reads ONLY split=='train' cases, and
       re-asserts (independently of build.py) that the train IDs it is about
       to train on are disjoint from data/build/tests_test.yaml's case IDs
       before spending a single GEPA rollout.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import dspy
import yaml

from evalloop import report as report_mod
from evalloop import run as run_mod
from evalloop.optimizers.base import OptimizeError, PromptOptimizer
from evalloop.optimizers.copro import (
    CoproOptimizer,
    run_copro,  # noqa: F401 -- monkeypatch target by convention; CoproOptimizer calls it through this module
)
from evalloop.optimizers.gepa import (
    GepaOptimizer,
    run_gepa,  # noqa: F401 -- historical monkeypatch target; GepaOptimizer calls it through this module
)

# Backward-compatible re-exports: calibrate.py and the test suite import these
# metric functions from evalloop.optimize; the implementations moved to
# evalloop.optimizers.metrics in the APO-03 refactor.
from evalloop.optimizers.metrics import (  # noqa: F401
    _normalize_label,
    _score_fn_for,
    extract_instructions_from_template,
    json_score_and_feedback,
    label_score_and_feedback,
    render_optimized_template,
    text_score_and_feedback,
)
from evalloop.optimizers.miprov2 import (
    MiproV2Optimizer,
    run_miprov2,  # noqa: F401 -- monkeypatch target by convention; MiproV2Optimizer calls it through this module
)
from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import Config, assert_split_disjoint, load_golden_jsonl, parse_promptfoo_output

# ---------------------------------------------------------------------------
# promptfoo provider id -> dspy/litellm model string
# ---------------------------------------------------------------------------


def promptfoo_provider_to_dspy_lm(provider: str) -> str:
    if provider.startswith("anthropic:messages:"):
        return "anthropic/" + provider.split(":", 2)[2]
    if provider.startswith("ollama:chat:"):
        return "ollama_chat/" + provider.split(":", 2)[2]
    # TODO: add a case here (and verify against https://dspy.ai/ provider docs)
    # before using any provider prefix other than the two above in config.yaml.
    raise OptimizeError(
        f"don't know how to translate promptfoo provider {provider!r} into a dspy LM string "
        "(only anthropic:messages: and ollama:chat: are mapped so far) -- add a case to "
        "promptfoo_provider_to_dspy_lm() in optimize.py"
    )


def _dspy_temperature(supports_sampling_params: bool, temperature: float) -> float | None:
    """claude-opus-4-8 / claude-fable-5 reject sampling params with HTTP 400 on
    the dspy/litellm path too, not just through promptfoo. litellm drops
    None-valued params from the request (verified against the installed
    litellm: get_optional_params(temperature=None) omits the key), so None is
    how we avoid sending temperature to those models.
    """
    return temperature if supports_sampling_params else None


def _reflection_supports_sampling(cfg: Config) -> bool:
    """optimize.reflection_provider is a dspy/litellm string; if it corresponds
    to a registry model marked supports_sampling_params=false, temperature must
    not be sent to it either (the bundled configs point reflection at
    anthropic/claude-opus-4-8, which 400s on it). Providers with no registry
    match default to True (send temperature, the historical behavior).
    """
    for m in cfg.models:
        try:
            if promptfoo_provider_to_dspy_lm(m.provider) == cfg.optimize.reflection_provider:
                return m.supports_sampling_params
        except OptimizeError:
            continue  # registry entries with unmapped provider prefixes can't match
    return True


# ---------------------------------------------------------------------------
# variant config generation (reroots every file:// reference one level
# deeper, since promptfoo/variants/{name}.yaml lives one directory below
# promptfoo/promptfooconfig.yaml)
# ---------------------------------------------------------------------------


def _reroot_file_refs(obj, prefix: str):
    if isinstance(obj, dict):
        return {k: _reroot_file_refs(v, prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_reroot_file_refs(v, prefix) for v in obj]
    if isinstance(obj, str) and obj.startswith("file://"):
        return "file://" + prefix + obj[len("file://") :]
    return obj


def to_variant_relpath(target: Path, variants_dir: Path) -> str:
    rel = os.path.relpath(target, start=variants_dir)
    return rel.replace(os.sep, "/")


def build_variant_config(target_alias: str, task_path: Path, paths: TaskPaths) -> dict:
    if not paths.promptfoo_config.exists():
        raise OptimizeError(f"{paths.promptfoo_config} not found; run `evalloop build --task {paths.task}` first")
    base_config = yaml.safe_load(paths.promptfoo_config.read_text(encoding="utf-8"))
    variant_config = _reroot_file_refs(base_config, prefix="../")
    variant_config["prompts"] = [f"file://{to_variant_relpath(task_path, paths.variants_dir)}"]
    variant_config["description"] = f"{base_config.get('description', '')} (optimized: {target_alias})"
    return variant_config


# ---------------------------------------------------------------------------
# optimize orchestration
# ---------------------------------------------------------------------------


@dataclass
class OptimizeOutcome:
    variant_name: str
    task_path: Path
    variant_path: Path
    run_id: str
    base_run_id: str | None
    compare_path: Path | None


def _load_test_ids(paths: TaskPaths) -> set[str]:
    if not paths.tests_test.exists():
        raise OptimizeError(f"{paths.tests_test} not found; run `evalloop build --task {paths.task}` first")
    entries = yaml.safe_load(paths.tests_test.read_text(encoding="utf-8")) or []
    return {e["vars"]["case_id"] for e in entries}


def _find_latest_base_run(task_name: str, paths: TaskPaths) -> str | None:
    if not paths.index.exists():
        return None
    candidates = []
    with paths.index.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if (
                entry.get("task_name") == task_name
                and not entry.get("variant")
                and entry.get("promptfoo_exit_code") == 0
            ):
                candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda e: e["created_at"])
    return candidates[-1]["run_id"]


def optimize(config: Config, paths: TaskPaths, *, force: bool = False) -> OptimizeOutcome:
    cfg = config
    score_fn = _score_fn_for(cfg)  # resolve the training metric first: fail fast on unsupported types

    test_ids = _load_test_ids(paths)
    cases = load_golden_jsonl(paths.golden)
    train_cases = [c for c in cases if c.split == "train"]
    if not train_cases:
        raise OptimizeError("golden.jsonl has no split=='train' cases; nothing to optimize against")
    train_ids = {c.id for c in train_cases}
    assert_split_disjoint(train_ids, test_ids)  # iron rule #1, re-checked independently of build.py

    # APO-09: preflight checks (train size, label coverage, holdout presence).
    # Runs after split separation is confirmed and before any LM call. Errors
    # abort unless force=True; warnings are always printed.
    from rich.console import Console

    from evalloop import preflight as preflight_mod

    preflight_result = preflight_mod.run_preflight(
        cfg, train_cases, len(test_ids), force=force
    )
    console = Console()  # format_preflight() lines carry rich markup; plain print would show the tags
    for line in preflight_mod.format_preflight(preflight_result):
        console.print(line)
    preflight_mod.check_or_raise(preflight_result, force=force)

    target_model = cfg.model_by_alias(cfg.optimize.target_alias)
    task_lm = dspy.LM(
        promptfoo_provider_to_dspy_lm(target_model.provider),
        temperature=_dspy_temperature(target_model.supports_sampling_params, cfg.run.temperature),
        max_tokens=cfg.run.max_tokens,
    )
    reflection_lm = dspy.LM(
        cfg.optimize.reflection_provider,
        temperature=_dspy_temperature(_reflection_supports_sampling(cfg), 1.0),
        max_tokens=32000,
    )

    original_template = (REPO_ROOT / cfg.task.prompt_file).read_text(encoding="utf-8")
    base_instructions = extract_instructions_from_template(original_template)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        score, feedback = score_fn(getattr(pred, "output", ""), gold.expected)
        return dspy.Prediction(score=score, feedback=feedback)

    trainset = [dspy.Example(input=c.input, expected=c.expected).with_inputs("input") for c in train_cases]

    # optimizer selection by cfg.optimize.method (validated against
    # KNOWN_OPTIMIZE_METHODS at config load, so this lookup cannot miss)
    optimizer_classes: dict[str, type] = {
        GepaOptimizer.name: GepaOptimizer,
        MiproV2Optimizer.name: MiproV2Optimizer,
        CoproOptimizer.name: CoproOptimizer,
    }
    optimizer: PromptOptimizer = optimizer_classes[cfg.optimize.method]()
    started = time.monotonic()
    result = optimizer.optimize(
        base_instructions=base_instructions,
        trainset=trainset,
        metric=metric,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        cfg=cfg,
    )
    duration_seconds = round(time.monotonic() - started, 3)
    optimized_template = render_optimized_template(result.optimized_instructions, original_template)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # method identity comes from the optimizer that actually ran (APO-05):
    # it names the output dir, the variant, and the log record, so
    # cross-method comparisons (APO-13) can tell runs apart
    out_dir = paths.optimized_dir / cfg.optimize.target_alias / f"{result.method}-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_dir / "task.txt"
    task_path.write_text(optimized_template, encoding="utf-8")
    log_path = out_dir / "optimize_log.json"
    log_path.write_text(
        json.dumps(
            {
                "target_alias": cfg.optimize.target_alias,
                "reflection_provider": cfg.optimize.reflection_provider,
                "auto": cfg.optimize.auto,
                "method": result.method,
                # effective params: the raw method params with the resolved
                # auto (params.auto precedence is applied at config load)
                "params": {**cfg.optimize.params, "auto": cfg.optimize.auto},
                "duration_seconds": duration_seconds,
                "train_case_count": len(train_cases),
                "train_case_ids": sorted(train_ids),
                "base_instructions": base_instructions,
                "optimized_instructions": result.optimized_instructions,
                "created_at": datetime.now(timezone.utc).isoformat(),
                # method-specific extras (currently empty for GEPA)
                **result.extra_log,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[optimize] wrote {task_path}")
    print(f"[optimize] wrote {log_path}")

    variant_name = f"{cfg.optimize.target_alias}_{result.method}_{ts}"
    variant_config = build_variant_config(cfg.optimize.target_alias, task_path, paths)
    paths.variants_dir.mkdir(parents=True, exist_ok=True)
    variant_path = paths.variants_dir / f"{variant_name}.yaml"
    variant_path.write_text(yaml.safe_dump(variant_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[optimize] wrote {variant_path}")

    outcome = run_mod.run(cfg, paths, variant=variant_name)
    report_mod.report(outcome.run_id, paths)

    base_run_id = _find_latest_base_run(cfg.task.name, paths)
    compare_path = None
    if base_run_id:
        compare_path = compare(base_run_id, outcome.run_id, paths)
    else:
        print(f"[optimize] no prior base run found in {paths.index}; skipping compare")

    return OptimizeOutcome(
        variant_name=variant_name,
        task_path=task_path,
        variant_path=variant_path,
        run_id=outcome.run_id,
        base_run_id=base_run_id,
        compare_path=compare_path,
    )


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _fmt_pct(v):
    return f"{v:.1%}" if v is not None else "n/a"


def _fmt_pct_signed(v):
    return f"{v:+.1%}" if v is not None else "n/a"


def _fmt_usd(v):
    return f"${v:.4f}" if v is not None else "n/a"


def _fmt_usd_signed(v):
    return f"{'+' if v >= 0 else ''}${v:.4f}" if v is not None else "n/a"


def compare(run_a: str, run_b: str, paths: TaskPaths) -> Path:
    output_a = paths.runs_dir / run_a / "output.json"
    output_b = paths.runs_dir / run_b / "output.json"
    if not output_a.exists():
        raise OptimizeError(f"run {run_a!r} not found ({output_a})")
    if not output_b.exists():
        raise OptimizeError(f"run {run_b!r} not found ({output_b})")

    stats_a = {s.alias: s for s in report_mod.compute_alias_stats(parse_promptfoo_output(output_a).results)}
    stats_b = {s.alias: s for s in report_mod.compute_alias_stats(parse_promptfoo_output(output_b).results)}
    aliases = sorted(set(stats_a) | set(stats_b))

    lines = [
        f"# Compare: {run_a} (A, before) vs {run_b} (B, after)",
        "",
        "| alias | pass_rate A | pass_rate B | delta | beyond_95ci | cost A | cost B | cost delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in aliases:
        a, b = stats_a.get(alias), stats_b.get(alias)
        pa = a.pass_rate if a else None
        pb = b.pass_rate if b else None
        delta = (pb - pa) if (pa is not None and pb is not None) else None
        # issue #11: flag whether the delta clears the noise floor -- "yes"
        # only when the two Wilson 95% intervals do not overlap at all
        if a and b and a.pass_ci_low is not None and b.pass_ci_low is not None:
            non_overlap = b.pass_ci_low > a.pass_ci_high or b.pass_ci_high < a.pass_ci_low
            beyond_ci = "yes" if non_overlap else "no"
        else:
            beyond_ci = "n/a"
        ca = a.total_cost_usd if a else None
        cb = b.total_cost_usd if b else None
        cdelta = (cb - ca) if (ca is not None and cb is not None) else None
        lines.append(
            f"| {alias} | {_fmt_pct(pa)} | {_fmt_pct(pb)} | {_fmt_pct_signed(delta)} | {beyond_ci} | "
            f"{_fmt_usd(ca)} | {_fmt_usd(cb)} | {_fmt_usd_signed(cdelta)} |"
        )
    lines.append("")
    lines.append(
        "> beyond_95ci: yes when the Wilson 95% intervals of A and B do not overlap "
        "(a conservative significance check; overlapping intervals mean the delta may be noise)."
    )
    lines.append("")

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    path = paths.reports_dir / f"compare_{run_a}_{run_b}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare] wrote {path}")
    return path
