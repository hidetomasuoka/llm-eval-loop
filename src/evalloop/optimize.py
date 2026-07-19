"""Prompt optimization orchestration: golden.jsonl split=='train' -> optimized
prompt -> promptfoo variant config -> automatic run/report/compare.

Method-specific code lives in the evalloop.optimizers package: the shared
contract in optimizers/base.py, the GEPA implementation in optimizers/gepa.py,
and the deterministic proxy metrics + template round-trip helpers in
optimizers/metrics.py (see its module docstring for why training uses a proxy
metric instead of the final promptfoo judge). Supporting concerns extracted in
the same spirit: evalloop.dspy_lm (provider mapping / LM history costs),
evalloop.optimize_cost (pre-run estimate), evalloop.variants (variant config,
slug/summary, optimized index), and evalloop.compare (run comparison reports).
This module keeps optimizer selection and the optimize() orchestration, and
re-exports the moved symbols for backward compatibility.

Iron rules enforced here:
    1. split separation: this module reads ONLY split=='train' cases, and
       re-asserts (independently of build.py) that the train IDs it is about
       to train on are disjoint from data/build/tests_test.yaml's case IDs
       before spending a single GEPA rollout.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import dspy
import yaml

from evalloop import report as report_mod
from evalloop import run as run_mod

# Backward-compatible re-exports: cli.py, blog.py, and the test suite import
# these names from evalloop.optimize; the implementations moved to
# evalloop.compare in the same refactor that split this module.
from evalloop.compare import (  # noqa: F401
    _COMPARE_MULTI_DISCLAIMER,
    COMPARE_TRADEOFF_COST_INCREASE_RATIO,
    COMPARE_TRADEOFF_OUTPUT_TOKENS_INCREASE_RATIO,
    COMPARE_TRADEOFF_PROMPT_LENGTH_INCREASE_RATIO,
    _compare_matrix,
    _compare_pair,
    _compare_report_filename,
    _compare_tradeoff_notes,
    _fmt_explore_cost,
    _fmt_explore_duration,
    _fmt_int,
    _fmt_num,
    _fmt_pct,
    _fmt_pct_signed,
    _fmt_usd,
    _fmt_usd_signed,
    _iter_optimized_index,
    _load_optimize_log_for_run,
    _load_run_meta,
    _method_for_variant,
    _read_optimize_log_from_index_entry,
    _relative_increase,
    compare,
)
from evalloop.demos import (
    DEMOS_PLACEHOLDER,
    DemoCase,
    DemoError,
    assert_demos_do_not_leak_test,
    expand_demos_in_template,
    save_demos_jsonl,
)

# Backward-compatible re-exports: moved to evalloop.dspy_lm.
from evalloop.dspy_lm import (  # noqa: F401
    SearchCostSummary,
    _cost_from_history_entry,
    _dspy_temperature,
    _history_entries,
    _reflection_registry_model,
    _reflection_supports_sampling,
    _tokens_from_usage,
    promptfoo_provider_to_dspy_lm,
    summarize_lm_search_cost,
)

# Backward-compatible re-exports: moved to evalloop.optimize_cost.
from evalloop.optimize_cost import (  # noqa: F401
    _AUTO_ROLLOUT_FACTORS,
    REFLECTION_INPUT_TOKENS_ESTIMATE,
    REFLECTION_OUTPUT_TOKENS_ESTIMATE,
    OptimizeCostEstimate,
    _rollout_factor,
    estimate_optimize_cost,
)
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
    _split_template,
    extract_instructions_from_template,
    json_score_and_feedback,
    label_score_and_feedback,
    render_optimized_template,
    text_score_and_feedback,
)
from evalloop.optimizers.miprov2 import (
    OPTIMIZED_DEMOS_LOG_KEY,
    MiproV2Optimizer,
    run_miprov2,  # noqa: F401 -- monkeypatch target by convention; MiproV2Optimizer calls it through this module
)
from evalloop.optimizers.schedulers import select_eval_subset
from evalloop.optimizers.tapo import (
    TapoOptimizer,
    run_tapo,  # noqa: F401 -- monkeypatch target by convention; TapoOptimizer calls it through this module
)
from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import (
    Config,
    assert_split_disjoint,
    load_golden_jsonl,
    parse_promptfoo_output,
)

# Backward-compatible re-exports: moved to evalloop.variants.
from evalloop.variants import (  # noqa: F401
    _OPTIMIZED_DIR_RE,
    _PARAM_KEY_SHORT,
    _SLUG_MAX_LEN,
    _append_optimized_index,
    _format_param_token,
    _instructions_hash,
    _make_variant_slug,
    _make_variant_summary,
    _occupied_slugs,
    _reroot_file_refs,
    _sanitize_slug_part,
    _short_param_key,
    _slug_from_dir_name,
    build_variant_config,
    to_variant_relpath,
)

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


@dataclass
class GeneralizationRecord:
    train_score: float | None
    holdout_score: float | None
    base_holdout_score: float | None
    holdout_delta: float | None
    generalization: str | None  # "pass" | "fail" when baseline exists; else None


def _alias_pass_rate(run_id: str, alias: str, paths: TaskPaths) -> float | None:
    output_path = paths.runs_dir / run_id / "output.json"
    if not output_path.exists():
        return None
    stats = {s.alias: s for s in report_mod.compute_alias_stats(parse_promptfoo_output(output_path).results)}
    stat = stats.get(alias)
    return stat.pass_rate if stat else None


def evaluate_generalization_gate(
    *,
    train_score: float | None,
    optimized_run_id: str,
    base_run_id: str | None,
    target_alias: str,
    paths: TaskPaths,
) -> GeneralizationRecord:
    """Compare train proxy score vs holdout pass rate; gate on baseline holdout delta."""
    holdout_score = _alias_pass_rate(optimized_run_id, target_alias, paths)
    base_holdout_score = None
    holdout_delta = None
    generalization = None
    if base_run_id:
        base_holdout_score = _alias_pass_rate(base_run_id, target_alias, paths)
        if holdout_score is not None and base_holdout_score is not None:
            holdout_delta = holdout_score - base_holdout_score
            generalization = "pass" if holdout_delta > 0 else "fail"
    return GeneralizationRecord(
        train_score=train_score,
        holdout_score=holdout_score,
        base_holdout_score=base_holdout_score,
        holdout_delta=holdout_delta,
        generalization=generalization,
    )


def _generalization_record_to_log(record: GeneralizationRecord) -> dict:
    payload = {
        "train_score": record.train_score,
        "holdout_score": record.holdout_score,
        "base_holdout_score": record.base_holdout_score,
        "holdout_delta": record.holdout_delta,
        "generalization": record.generalization,
    }
    return {k: v for k, v in payload.items() if v is not None}


def _patch_optimize_log(log_path: Path, record: GeneralizationRecord) -> None:
    data = json.loads(log_path.read_text(encoding="utf-8"))
    data.update(_generalization_record_to_log(record))
    log_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_generalization_gate(console, record: GeneralizationRecord) -> None:
    console.print("[optimize] generalization gate (train proxy vs holdout pass rate):")
    console.print(
        f"[optimize]   train_score={_fmt_pct(record.train_score)} holdout_score={_fmt_pct(record.holdout_score)}"
    )
    if record.base_holdout_score is not None:
        console.print(
            f"[optimize]   baseline holdout={_fmt_pct(record.base_holdout_score)} "
            f"delta={_fmt_pct_signed(record.holdout_delta)}"
        )
    if record.generalization == "pass":
        console.print("[optimize]   generalization: pass (holdout improved vs baseline)")
    elif record.generalization == "fail":
        console.print("[optimize]   [red]不合格: 過学習の疑い[/red] (holdout did not improve vs baseline)")
    elif record.base_holdout_score is None and record.holdout_score is not None:
        console.print("[optimize]   generalization: n/a (no baseline run to compare against)")


def _load_holdout_from_build(paths: TaskPaths) -> tuple[set[str], set[str]]:
    """Return (case_ids, inputs) from the last build's tests_test.yaml.

    promptfoo eval uses this YAML, so demo leak checks must cover it even when
    golden.jsonl was edited without a rebuild.
    """
    if not paths.tests_test.exists():
        raise OptimizeError(f"{paths.tests_test} not found; run `evalloop build --task {paths.task}` first")
    entries = yaml.safe_load(paths.tests_test.read_text(encoding="utf-8")) or []
    ids: set[str] = set()
    inputs: set[str] = set()
    for entry in entries:
        vars_ = entry.get("vars") or {}
        case_id = vars_.get("case_id")
        if case_id is not None:
            ids.add(str(case_id))
        inp = vars_.get("input")
        if inp is not None:
            inputs.add(str(inp))
    return ids, inputs


def _load_test_ids(paths: TaskPaths) -> set[str]:
    ids, _inputs = _load_holdout_from_build(paths)
    return ids


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


def optimize(
    config: Config, paths: TaskPaths, *, force: bool = False, yes: bool = False, confirm_fn=None
) -> OptimizeOutcome:
    cfg = config
    score_fn = _score_fn_for(cfg)  # resolve the training metric first: fail fast on unsupported types

    test_ids, yaml_test_inputs = _load_holdout_from_build(paths)
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

    preflight_result = preflight_mod.run_preflight(cfg, train_cases, len(test_ids), force=force)
    console = Console()  # format_preflight() lines carry rich markup; plain print would show the tags
    for line in preflight_mod.format_preflight(preflight_result):
        console.print(line)
    preflight_mod.check_or_raise(preflight_result, force=force)

    raw_template = (REPO_ROOT / cfg.task.prompt_file).read_text(encoding="utf-8")
    # Expand {{demos}} the same way build does, so dspy trains on the prompt
    # promptfoo will evaluate (APO-16 / issue #75 Bugbot finding).
    # Leak check unions golden test split with build YAML holdout: promptfoo
    # still evaluates the last build's tests_test.yaml even if golden drifted.
    golden_test_cases = [c for c in cases if c.split == "test"]
    demos_test_ids = test_ids | {c.id for c in golden_test_cases}
    demos_test_inputs = yaml_test_inputs | {c.input for c in golden_test_cases}
    miprov2_demo_search = cfg.optimize.method == MiproV2Optimizer.name and (
        int(cfg.optimize.params.get("max_bootstrapped_demos", 0) or 0) > 0
        or int(cfg.optimize.params.get("max_labeled_demos", 0) or 0) > 0
    )
    # Keep raw_template (with {{demos}}) for APO-17 variant re-injection.
    original_template = raw_template
    if miprov2_demo_search:
        # MIPROv2 will choose demos; strip placeholder so it is not baked into instructions.
        original_template = raw_template.replace(DEMOS_PLACEHOLDER, "")
    elif paths.demos.exists() and DEMOS_PLACEHOLDER not in original_template:
        print(f"[optimize] WARN: {paths.demos} exists but prompt has no {DEMOS_PLACEHOLDER}; demos are ignored")
    elif DEMOS_PLACEHOLDER in original_template:
        try:
            original_template, n_demos = expand_demos_in_template(
                original_template,
                paths.demos,
                test_ids=demos_test_ids,
                test_inputs=demos_test_inputs,
            )
        except DemoError as e:
            raise OptimizeError(str(e)) from e
        print(f"[optimize] embedded {n_demos} demos into the training template")

    # APO-10: order-of-magnitude cost warning + confirmation BEFORE the first
    # rollout is spent (mirrors build.py's --yes pattern)
    scheduler_strategy = str(cfg.optimize.params.get("eval_scheduler", "full"))
    eval_budget_raw = cfg.optimize.params.get("eval_budget")
    eval_budget = int(eval_budget_raw) if eval_budget_raw is not None else None
    scheduler_seed = int(cfg.optimize.params.get("seed", 0))
    optimize_cases = select_eval_subset(
        train_cases, strategy=scheduler_strategy, budget=eval_budget, seed=scheduler_seed
    )
    if cfg.optimize.method == MiproV2Optimizer.name and len(optimize_cases) < 2:
        raise OptimizeError(
            "miprov2 needs at least 2 cases after eval scheduling to carve out a validation set; "
            f"scheduler {scheduler_strategy!r} selected {len(optimize_cases)} case(s)"
        )
    optimize_ids = {c.id for c in optimize_cases}
    if scheduler_strategy != "full":
        print(
            f"[optimize] eval scheduler {scheduler_strategy!r}: "
            f"using {len(optimize_cases)}/{len(train_cases)} train cases for candidate evaluation"
        )

    estimate = estimate_optimize_cost(cfg, optimize_cases, original_template)
    print(f"[optimize] estimated cost (rough, order-of-magnitude only -- method={estimate.method}):")
    print(
        f"[optimize]   target {cfg.optimize.target_alias}: ~{estimate.rollout_count} rollouts "
        f"({estimate.train_case_count} train cases x factor {estimate.rollout_factor}) = ${estimate.target_usd:.4f}"
    )
    print(
        f"[optimize]     ~{estimate.target_input_tokens} input tokens/rollout; "
        f"method={estimate.target_token_count_method}"
    )
    reflection_cost = (
        f"${estimate.reflection_usd:.4f}"
        if estimate.reflection_usd is not None
        else "price unknown (provider not in the config.yaml model registry)"
    )
    print(
        f"[optimize]   reflection {cfg.optimize.reflection_provider}: "
        f"~{estimate.reflection_call_count} calls = {reflection_cost}"
    )
    print(f"[optimize]   TOTAL: ~${estimate.total_usd:.4f} (excludes the post-optimize eval run)")
    print(
        "[optimize] tip: 初回は小さいデータ・軽い設定から始めることを推奨 "
        "(auto: light / 小さめの train split / 評価は `evalloop run --limit N`)"
    )

    if estimate.total_usd > cfg.run.cost_warn_usd and not yes:
        confirm = confirm_fn or (lambda msg: input(f"{msg} [y/N] ").strip().lower() == "y")
        if not confirm(
            f"Estimated optimize cost ${estimate.total_usd:.4f} exceeds cost_warn_usd "
            f"(${cfg.run.cost_warn_usd:.2f}). Continue?"
        ):
            raise OptimizeError("aborted by user: optimize cost estimate exceeded cost_warn_usd")

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

    base_instructions = extract_instructions_from_template(original_template)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # gold.input is the case's source document -- the text metric uses it
        # for the verbatim-quote check (improvement plan #3)
        score, feedback = score_fn(getattr(pred, "output", ""), gold.expected, getattr(gold, "input", None))
        return dspy.Prediction(score=score, feedback=feedback)

    trainset = [
        dspy.Example(input=c.input, expected=c.expected, case_id=c.id).with_inputs("input") for c in optimize_cases
    ]

    # optimizer selection by cfg.optimize.method (validated against
    # KNOWN_OPTIMIZE_METHODS at config load, so this lookup cannot miss)
    optimizer_classes: dict[str, type] = {
        GepaOptimizer.name: GepaOptimizer,
        MiproV2Optimizer.name: MiproV2Optimizer,
        CoproOptimizer.name: CoproOptimizer,
        TapoOptimizer.name: TapoOptimizer,
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
    search_cost = summarize_lm_search_cost(task_lm, reflection_lm, cfg)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # effective params: the raw method params with the resolved auto
    # (params.auto precedence is applied at config load)
    effective_params = {**cfg.optimize.params, "auto": cfg.optimize.auto}
    # method identity comes from the optimizer that actually ran (APO-05);
    # slug encodes auto/params/train size so variants are distinguishable
    # without opening optimize_log.json
    occupied_slugs = _occupied_slugs(paths.optimized_dir / cfg.optimize.target_alias)
    slug = _make_variant_slug(
        auto=cfg.optimize.auto,
        params=effective_params,
        train_case_count=len(optimize_cases),
        base_instructions=base_instructions,
        optimized_instructions=result.optimized_instructions,
        occupied=occupied_slugs,
    )
    summary = _make_variant_summary(
        method=result.method,
        auto=cfg.optimize.auto,
        params=effective_params,
        train_case_count=len(optimize_cases),
        base_instructions=base_instructions,
        optimized_instructions=result.optimized_instructions,
    )
    dir_name = f"{result.method}-{ts}-{slug}"
    out_dir = paths.optimized_dir / cfg.optimize.target_alias / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # APO-17: persist MIPROv2 few-shot demos and re-expand {{demos}} into variant prompt.
    extra_log = dict(result.extra_log)
    optimized_demo_rows = extra_log.pop(OPTIMIZED_DEMOS_LOG_KEY, None)
    created_at = datetime.now(timezone.utc).isoformat()
    if miprov2_demo_search:
        if not optimized_demo_rows:
            raise OptimizeError(
                "miprov2 demo search was enabled but the compiled program produced 0 demos; "
                "refusing to write a demos-less variant (check trainset size and max_*_demos)"
            )
        demos_with_origin = [
            (
                DemoCase(input=row["input"], output=row["output"], id=row.get("id")),
                str(row.get("origin") or "labeled"),
            )
            for row in optimized_demo_rows
        ]
        demos_only = [d for d, _o in demos_with_origin]
        try:
            assert_demos_do_not_leak_test(demos_only, test_ids=demos_test_ids, test_inputs=demos_test_inputs)
        except DemoError as e:
            raise OptimizeError(str(e)) from e
        demos_path = out_dir / "demos.jsonl"
        save_demos_jsonl(
            demos_path,
            demos_with_origin,
            provenance={
                "source": "miprov2-optimize",
                "method": result.method,
                "max_bootstrapped_demos": int(cfg.optimize.params.get("max_bootstrapped_demos", 0) or 0),
                "max_labeled_demos": int(cfg.optimize.params.get("max_labeled_demos", 0) or 0),
                "seed": int(cfg.optimize.params.get("seed", 0) or 0),
                "created_at": created_at,
            },
        )
        # render_optimized_template rewrites the pre-{{input}} region, which can
        # drop a {{demos}} paragraph that lived outside the input trailer. Restore it.
        shell = render_optimized_template(result.optimized_instructions, raw_template)
        if DEMOS_PLACEHOLDER not in shell:
            if DEMOS_PLACEHOLDER not in raw_template:
                raise OptimizeError(f"miprov2 produced demos but {cfg.task.prompt_file} has no {DEMOS_PLACEHOLDER}")
            _instr, trailer = _split_template(raw_template)
            shell = f"{result.optimized_instructions.strip()}\n\n{DEMOS_PLACEHOLDER}\n\n{trailer}\n"
        try:
            optimized_template, n_demos = expand_demos_in_template(
                shell,
                demos_path,
                test_ids=demos_test_ids,
                test_inputs=demos_test_inputs,
            )
        except DemoError as e:
            raise OptimizeError(str(e)) from e
        if n_demos is None:
            raise OptimizeError(f"miprov2 produced demos but {cfg.task.prompt_file} has no {DEMOS_PLACEHOLDER}")
        extra_log["demos_path"] = f"{cfg.optimize.target_alias}/{dir_name}/demos.jsonl"
        extra_log["demo_ids"] = [d.id for d in demos_only if d.id]
        print(f"[optimize] wrote {demos_path} ({len(demos_only)} demos)")
    else:
        optimized_template = render_optimized_template(result.optimized_instructions, original_template)

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
                "params": effective_params,
                "slug": slug,
                "summary": summary,
                "duration_seconds": duration_seconds,
                "search_cost_usd": search_cost.search_cost_usd,
                "search_lm_call_count": search_cost.search_lm_call_count,
                "train_case_count": len(optimize_cases),
                "train_case_ids": sorted(optimize_ids),
                "full_train_case_count": len(train_cases),
                "eval_scheduler": scheduler_strategy,
                "eval_budget": eval_budget,
                "base_instructions": base_instructions,
                "optimized_instructions": result.optimized_instructions,
                "created_at": created_at,
                # method-specific extras (currently empty for GEPA)
                **extra_log,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[optimize] wrote {task_path}")
    print(f"[optimize] wrote {log_path}")

    variant_name = f"{cfg.optimize.target_alias}_{result.method}_{ts}_{slug}"
    variant_config = build_variant_config(cfg.optimize.target_alias, task_path, paths)
    paths.variants_dir.mkdir(parents=True, exist_ok=True)
    variant_path = paths.variants_dir / f"{variant_name}.yaml"
    variant_path.write_text(yaml.safe_dump(variant_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[optimize] wrote {variant_path}")

    outcome = run_mod.run(cfg, paths, variant=variant_name)
    report_mod.report(outcome.run_id, paths)

    base_run_id = _find_latest_base_run(cfg.task.name, paths)
    generalization = evaluate_generalization_gate(
        train_score=result.extra_log.get("train_score"),
        optimized_run_id=outcome.run_id,
        base_run_id=base_run_id,
        target_alias=cfg.optimize.target_alias,
        paths=paths,
    )
    _print_generalization_gate(console, generalization)
    _patch_optimize_log(log_path, generalization)

    compare_path = None
    if base_run_id:
        compare_path = compare([base_run_id, outcome.run_id], paths)
    else:
        print(f"[optimize] no prior base run found in {paths.index}; skipping compare")

    rel_dir = f"{cfg.optimize.target_alias}/{dir_name}"
    _append_optimized_index(
        paths,
        {
            "variant_name": variant_name,
            "created_at": created_at,
            "method": result.method,
            "target_alias": cfg.optimize.target_alias,
            "slug": slug,
            "dir": rel_dir,
            "summary": summary,
            "params": effective_params,
            "train_case_count": len(optimize_cases),
            "run_id": outcome.run_id,
            "base_run_id": base_run_id,
            "optimize_log": f"{rel_dir}/optimize_log.json",
        },
    )
    print(f"[optimize] appended {paths.optimized_index}")

    return OptimizeOutcome(
        variant_name=variant_name,
        task_path=task_path,
        variant_path=variant_path,
        run_id=outcome.run_id,
        base_run_id=base_run_id,
        compare_path=compare_path,
    )
