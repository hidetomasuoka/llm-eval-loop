"""Run comparison reports: the 2-run before/after delta table and the 3+-run
model x run matrix (APO-13 / APO-21).

Extracted from evalloop.optimize (kept there as re-exports for backward
compatibility; `evalloop compare` still enters through evalloop.optimize).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evalloop import report as report_mod
from evalloop import stats as stats_mod
from evalloop.optimizers.base import OptimizeError
from evalloop.paths import TaskPaths
from evalloop.schemas import parse_promptfoo_output


def _fmt_pct(v):
    return f"{v:.1%}" if v is not None else "n/a"


def _fmt_pct_signed(v):
    return f"{v:+.1%}" if v is not None else "n/a"


def _fmt_usd(v):
    return f"${v:.4f}" if v is not None else "n/a"


def _fmt_usd_signed(v):
    return f"{'+' if v >= 0 else ''}${v:.4f}" if v is not None else "n/a"


# Conditionality disclaimer for multi-run method matrices (APO-13 / issue #72).
_COMPARE_MULTI_DISCLAIMER = (
    "手法の優劣はデータセット・meta-LLM・タスク形式に条件依存する。この結果は本タスク・本設定に限る"
)

# APO-21 / issue #80: flag accuracy gains that come with large cost/length spikes.
COMPARE_TRADEOFF_COST_INCREASE_RATIO = 0.50
COMPARE_TRADEOFF_OUTPUT_TOKENS_INCREASE_RATIO = 0.50
COMPARE_TRADEOFF_PROMPT_LENGTH_INCREASE_RATIO = 0.50


def _load_run_meta(run_id: str, paths: TaskPaths) -> dict:
    meta_path = paths.runs_dir / run_id / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _iter_optimized_index(paths: TaskPaths):
    index_path = paths.optimized_index
    if not index_path.exists():
        return
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if isinstance(entry, dict):
                yield entry
    except (OSError, json.JSONDecodeError, TypeError):
        return


def _method_for_variant(variant: str | None, paths: TaskPaths) -> str | None:
    """Resolve optimizer method from optimized/index.jsonl or the variant slug."""
    if not variant:
        return None
    for entry in _iter_optimized_index(paths):
        if entry.get("variant_name") == variant and entry.get("method"):
            return str(entry["method"])
    # variant naming: {alias}_{method}_{timestamp}_{slug}
    parts = variant.split("_")
    if len(parts) >= 2 and parts[1] in {"gepa", "miprov2", "copro", "tapo"}:
        return parts[1]
    return None


def _read_optimize_log_from_index_entry(entry: dict, paths: TaskPaths) -> dict | None:
    rel = entry.get("optimize_log")
    if not isinstance(rel, str) or not rel:
        return None
    log_path = paths.optimized_dir / rel
    if not log_path.exists():
        return None
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _load_optimize_log_for_run(run_id: str, paths: TaskPaths, *, variant: str | None = None) -> dict | None:
    """Load optimize_log.json via optimized/index.jsonl (APO-14).

    Prefer an exact ``run_id`` match; fall back to ``variant_name`` so a
    re-eval of the same variant (new run_id) still resolves explore cost/time.
    """
    entries = list(_iter_optimized_index(paths))
    for entry in entries:
        if entry.get("run_id") == run_id:
            return _read_optimize_log_from_index_entry(entry, paths)
    if variant:
        # Last matching index row wins (newest append) if names collide.
        match: dict | None = None
        for entry in entries:
            if entry.get("variant_name") == variant:
                match = entry
        if match is not None:
            return _read_optimize_log_from_index_entry(match, paths)
    return None


def _fmt_explore_cost(variant: str | None, log: dict | None) -> str:
    """Base runs → ``-``; missing log/field → ``n/a``; else USD."""
    if not variant:
        return "-"
    if not log or log.get("search_cost_usd") is None:
        return "n/a"
    try:
        return _fmt_usd(float(log["search_cost_usd"]))
    except (TypeError, ValueError):
        return "n/a"


def _fmt_explore_duration(variant: str | None, log: dict | None) -> str:
    """Base runs → ``-``; missing log/field → ``n/a``; else seconds."""
    if not variant:
        return "-"
    if not log or log.get("duration_seconds") is None:
        return "n/a"
    try:
        return f"{float(log['duration_seconds']):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def _compare_report_filename(run_ids: list[str]) -> str:
    """Keep readable names for <=3 runs; hash-shorten at 4+ (APO-13)."""
    if len(run_ids) <= 3:
        return "compare_" + "_".join(run_ids) + ".md"
    digest = hashlib.sha1("|".join(run_ids).encode("utf-8")).hexdigest()[:10]
    return f"compare_{len(run_ids)}runs_{digest}.md"


def _relative_increase(before: float | None, after: float | None) -> float | None:
    """(after - before) / before when before > 0; else None."""
    if before is None or after is None or before <= 0:
        return None
    return (after - before) / before


def _fmt_num(v: float | None, spec: str = ".1f") -> str:
    return format(v, spec) if v is not None else "n/a"


def _fmt_int(v: int | None) -> str:
    return str(v) if v is not None else "n/a"


def _compare_tradeoff_notes(
    *,
    alias: str,
    accuracy_delta: float | None,
    cost_ratio: float | None,
    output_tok_ratio: float | None,
    prompt_len_ratio: float | None,
) -> list[str]:
    """Warn when accuracy improves but cost/tokens/prompt length spike (APO-21)."""
    if accuracy_delta is None or accuracy_delta <= 0:
        return []
    spikes: list[str] = []
    if cost_ratio is not None and cost_ratio > COMPARE_TRADEOFF_COST_INCREASE_RATIO:
        spikes.append(f"コスト +{cost_ratio:.0%}")
    if output_tok_ratio is not None and output_tok_ratio > COMPARE_TRADEOFF_OUTPUT_TOKENS_INCREASE_RATIO:
        spikes.append(f"出力トークン +{output_tok_ratio:.0%}")
    if prompt_len_ratio is not None and prompt_len_ratio > COMPARE_TRADEOFF_PROMPT_LENGTH_INCREASE_RATIO:
        spikes.append(f"プロンプト長 +{prompt_len_ratio:.0%}")
    if not spikes:
        return []
    return [f"> ⚠ トレードオフ注意 ({alias}): 精度 {_fmt_pct_signed(accuracy_delta)} 改善に対し、" + "、".join(spikes)]


def _compare_pair(run_a: str, run_b: str, paths: TaskPaths) -> list[str]:
    """Before/after delta table with APO-21 tradeoff columns."""
    results_a = parse_promptfoo_output(paths.runs_dir / run_a / "output.json").results
    results_b = parse_promptfoo_output(paths.runs_dir / run_b / "output.json").results
    stats_a = {s.alias: s for s in report_mod.compute_alias_stats(results_a)}
    stats_b = {s.alias: s for s in report_mod.compute_alias_stats(results_b)}
    meta_a = _load_run_meta(run_a, paths)
    meta_b = _load_run_meta(run_b, paths)
    prompt_a = report_mod.prompt_file_char_len(meta_a, root=paths.root)
    prompt_b = report_mod.prompt_file_char_len(meta_b, root=paths.root)
    prompt_delta = (prompt_b - prompt_a) if (prompt_a is not None and prompt_b is not None) else None
    prompt_ratio = _relative_increase(
        float(prompt_a) if prompt_a is not None else None,
        float(prompt_b) if prompt_b is not None else None,
    )
    aliases = sorted(set(stats_a) | set(stats_b))

    lines = [
        f"# Compare: {run_a} (A, before) vs {run_b} (B, after)",
        "",
        "| alias | pass_rate A | pass_rate B | delta | beyond_95ci | b/c | mcnemar_p | "
        "cost A | cost B | cost delta | cost delta % | "
        "avg_out_tok A | avg_out_tok B | out_tok delta | "
        "prompt_len A | prompt_len B | prompt_len delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tradeoff_notes: list[str] = []
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
        # paired McNemar exact test on the shared case set (improvement plan #2):
        # the same cases are graded in both runs, so the transition table has far
        # more power than the independent-sample Wilson check above
        transition = stats_mod.paired_transition(results_a, results_b, alias)
        if transition.n_paired == 0:
            bc_cell, p_cell = "n/a", "n/a"
        else:
            bc_cell = f"{transition.b}/{transition.c}"
            p_value = transition.p_value
            p_cell = f"{p_value:.3f}" if p_value is not None else "n/a"
        ca = a.total_cost_usd if a else None
        cb = b.total_cost_usd if b else None
        cdelta = (cb - ca) if (ca is not None and cb is not None) else None
        cost_ratio = _relative_increase(ca, cb)
        ta = a.avg_model_completion_tokens if a else None
        tb = b.avg_model_completion_tokens if b else None
        tdelta = (tb - ta) if (ta is not None and tb is not None) else None
        tok_ratio = _relative_increase(ta, tb)
        lines.append(
            f"| {alias} | {_fmt_pct(pa)} | {_fmt_pct(pb)} | {_fmt_pct_signed(delta)} | {beyond_ci} | "
            f"{bc_cell} | {p_cell} | "
            f"{_fmt_usd(ca)} | {_fmt_usd(cb)} | {_fmt_usd_signed(cdelta)} | {_fmt_pct_signed(cost_ratio)} | "
            f"{_fmt_num(ta)} | {_fmt_num(tb)} | {_fmt_num(tdelta, '+.1f')} | "
            f"{_fmt_int(prompt_a)} | {_fmt_int(prompt_b)} | {_fmt_num(prompt_delta, '+.0f')} |"
        )
        tradeoff_notes.extend(
            _compare_tradeoff_notes(
                alias=alias,
                accuracy_delta=delta,
                cost_ratio=cost_ratio,
                output_tok_ratio=tok_ratio,
                prompt_len_ratio=prompt_ratio,
            )
        )
    lines.append("")
    lines.append(
        "> beyond_95ci: yes when the Wilson 95% intervals of A and B do not overlap "
        "(a conservative significance check; overlapping intervals mean the delta may be noise)."
    )
    lines.append(
        "> b/c, mcnemar_p: paired McNemar exact test on the case set graded in BOTH runs "
        "(b = cases improved fail→pass, c = regressed pass→fail). Because the same cases are "
        "compared, this has more power than the independent-sample beyond_95ci check; "
        "mcnemar_p < 0.05 means the flip pattern is unlikely to be noise. n/a when no case is "
        "graded in both runs or no case flipped."
    )
    lines.append(
        "> cost delta % / out_tok / prompt_len: tradeoff axes vs A (APO-21). "
        f"A tradeoff warning is emitted when accuracy improves but cost/tokens/prompt grow "
        f"> {COMPARE_TRADEOFF_COST_INCREASE_RATIO:.0%}."
    )
    lines.append("")
    lines.extend(tradeoff_notes)
    if tradeoff_notes:
        lines.append("")
    return lines


def _compare_matrix(run_ids: list[str], paths: TaskPaths) -> list[str]:
    """Model × run matrix for 3+ runs (accuracy / cost / latency / explore)."""
    per_run_stats: list[dict[str, report_mod.AliasStats]] = []
    headers: list[str] = []
    explore_costs: list[str] = []
    explore_durations: list[str] = []
    for run_id in run_ids:
        meta = _load_run_meta(run_id, paths)
        variant_raw = meta.get("variant")
        variant = variant_raw if isinstance(variant_raw, str) else None
        method = _method_for_variant(variant, paths)
        variant_label = variant or "(base)"
        method_label = method or "n/a"
        headers.append(f"`{run_id}` (variant=`{variant_label}`, method=`{method_label}`)")
        opt_log = _load_optimize_log_for_run(run_id, paths, variant=variant) if variant else None
        explore_costs.append(_fmt_explore_cost(variant, opt_log))
        explore_durations.append(_fmt_explore_duration(variant, opt_log))
        per_run_stats.append(
            {
                s.alias: s
                for s in report_mod.compute_alias_stats(
                    parse_promptfoo_output(paths.runs_dir / run_id / "output.json").results
                )
            }
        )

    aliases = sorted({alias for stats in per_run_stats for alias in stats})
    # Compact column labels R1..Rn; full run identity lives in the Runs list.
    col_labels = [f"R{i + 1}" for i in range(len(run_ids))]

    lines = [
        "# Compare: " + " vs ".join(run_ids),
        "",
        "## Runs",
        "",
    ]
    for label, header in zip(col_labels, headers, strict=True):
        lines.append(f"- {label}: {header}")
    lines.append("")

    header_cells = ["alias"]
    align_cells = ["---"]
    for label in col_labels:
        header_cells.extend(
            [
                f"pass_rate {label}",
                f"cost {label}",
                f"p50_ms {label}",
                f"search_cost {label}",
                f"duration_s {label}",
            ]
        )
        align_cells.extend(["---:", "---:", "---:", "---:", "---:"])
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(align_cells) + "|")

    for alias in aliases:
        cells = [alias]
        for stats, search_cost, duration_s in zip(per_run_stats, explore_costs, explore_durations, strict=True):
            s = stats.get(alias)
            cells.append(_fmt_pct(s.pass_rate if s else None))
            cells.append(_fmt_usd(s.total_cost_usd if s else None))
            cells.append(report_mod.fmt(s.p50_latency_ms if s else None, ".0f"))
            cells.append(search_cost)
            cells.append(duration_s)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "> search_cost / duration_s: optimize exploration (from optimize_log.json); "
        "base runs show `-`, missing logs show `n/a`. `cost` is the holdout eval cost."
    )
    lines.append("")
    lines.append(f"> {_COMPARE_MULTI_DISCLAIMER}")
    lines.append("")
    return lines


def compare(run_ids: list[str], paths: TaskPaths) -> Path:
    """Compare 2+ runs into ``results/<task>/reports/compare_*.md``.

    Two runs keep the legacy before/after delta table. Three or more emit a
    model×run matrix with variant/method headers (APO-13 / issue #72).
    """
    cleaned = [r.strip() for r in run_ids if r and r.strip()]
    if len(cleaned) < 2:
        raise OptimizeError("compare requires at least 2 run_ids")
    if len(cleaned) != len(set(cleaned)):
        raise OptimizeError("compare run_ids must be unique")

    for run_id in cleaned:
        output = paths.runs_dir / run_id / "output.json"
        if not output.exists():
            raise OptimizeError(f"run {run_id!r} not found ({output})")

    lines = _compare_pair(cleaned[0], cleaned[1], paths) if len(cleaned) == 2 else _compare_matrix(cleaned, paths)

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    path = paths.reports_dir / _compare_report_filename(cleaned)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare] wrote {path}")
    return path
