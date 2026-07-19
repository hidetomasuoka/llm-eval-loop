"""results/runs/{run_id}/output.json -> results/reports/{run_id}.md

A per-model matrix: pass rate, cost, latency, cache rate. This is the
"モデル×精度×コスト×レイテンシ" half of the two final deliverables in
README.md section 1; the failure x category pivot lives in analyze.py.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import CaseResult, parse_promptfoo_output


class ReportError(RuntimeError):
    pass


@dataclass
class AliasStats:
    alias: str
    n: int
    pass_rate: float | None
    total_cost_usd: float
    avg_cost_usd: float
    p50_latency_ms: float | None
    cache_rate: float
    error_count: int
    # token usage: prompt/completion tokens actually consumed. model_tokens
    # are the evaluated model's own output tokens; judge_tokens are the llm-rubric
    # grader's tokens (0 for non-judged answer_types). Useful when prices are $0
    # (e.g. local/Ollama models) -- cost alone is uninformative there.
    model_prompt_tokens: int = 0
    model_completion_tokens: int = 0
    avg_model_prompt_tokens: float | None = None
    avg_model_completion_tokens: float | None = None
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    # uncertainty additions (issue #11): a single pass rate hides both the
    # binomial sampling error and the run-to-run noise repeat runs reveal
    pass_ci_low: float | None = None
    pass_ci_high: float | None = None
    repeat_pass_rates: list[float] = field(default_factory=list)  # one entry per repeat_index; [] when repeat=1
    repeat_stddev: float | None = None  # stddev across repeat_pass_rates; None when <2 repeats
    flip_case_ids: list[str] = field(default_factory=list)  # cases graded both pass AND fail across repeats
    flip_rate: float | None = None  # flips / cases-with->=2-repeats; None when no case has repeats


def _percentile50(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. NOTE: with
    repeat>1 the graded rows are not independent samples (the same cases are
    resampled), so this interval is optimistic -- repeat_stddev is the
    empirical run-to-run noise measure; this is the sampling-error floor.
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = successes / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_alias_stats(results: list[CaseResult]) -> list[AliasStats]:
    by_alias: dict[str, list[CaseResult]] = {}
    for r in results:
        by_alias.setdefault(r.alias or "unknown", []).append(r)

    stats: list[AliasStats] = []
    for alias, rows in sorted(by_alias.items()):
        graded = [r for r in rows if r.passed is not None]
        successes = sum(1 for r in graded if r.passed)
        pass_rate = (successes / len(graded)) if graded else None
        ci_low, ci_high = wilson_interval(successes, len(graded)) if graded else (None, None)

        # per-repeat pass rates (repeat_index is per (case, alias) -- schemas.py)
        by_repeat: dict[int, list[bool]] = {}
        for r in graded:
            by_repeat.setdefault(r.repeat_index, []).append(bool(r.passed))
        repeat_pass_rates = (
            [sum(flags) / len(flags) for _idx, flags in sorted(by_repeat.items())] if len(by_repeat) > 1 else []
        )
        repeat_stddev = statistics.stdev(repeat_pass_rates) if len(repeat_pass_rates) >= 2 else None

        # flip = a case graded more than once whose verdict is not constant
        verdicts_by_case: dict[str, list[bool]] = {}
        for r in graded:
            if r.case_id:
                verdicts_by_case.setdefault(r.case_id, []).append(bool(r.passed))
        multi_repeat_cases = {cid: flags for cid, flags in verdicts_by_case.items() if len(flags) > 1}
        flip_case_ids = sorted(cid for cid, flags in multi_repeat_cases.items() if len(set(flags)) > 1)
        flip_rate = (len(flip_case_ids) / len(multi_repeat_cases)) if multi_repeat_cases else None

        costs = [r.cost or 0.0 for r in rows]
        latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
        cache_hits = sum(1 for r in rows if r.cached)
        errors = sum(1 for r in rows if r.error)
        # token usage: model tokens come from CaseResult.token_usage (response
        # tokenUsage, parsed in schemas.py); judge tokens live under
        # gradingResult.tokensUsed which schemas.py does not surface, so pull
        # them from CaseResult.raw here (the only place that needs them).
        model_prompt = sum((r.token_usage or {}).get("prompt", 0) or 0 for r in rows)
        model_completion = sum((r.token_usage or {}).get("completion", 0) or 0 for r in rows)
        judge_prompt = 0
        judge_completion = 0
        for r in rows:
            grading = (r.raw or {}).get("gradingResult") or {}
            jt = grading.get("tokensUsed") or {}
            judge_prompt += jt.get("prompt", 0) or 0
            judge_completion += jt.get("completion", 0) or 0
        if rows and (model_prompt or model_completion):
            avg_model_prompt = model_prompt / len(rows)
            avg_model_completion = model_completion / len(rows)
        else:
            avg_model_prompt = None
            avg_model_completion = None
        stats.append(
            AliasStats(
                alias=alias,
                n=len(rows),
                pass_rate=pass_rate,
                total_cost_usd=sum(costs),
                avg_cost_usd=(sum(costs) / len(rows)) if rows else 0.0,
                p50_latency_ms=_percentile50(latencies),
                cache_rate=(cache_hits / len(rows)) if rows else 0.0,
                error_count=errors,
                model_prompt_tokens=model_prompt,
                model_completion_tokens=model_completion,
                avg_model_prompt_tokens=avg_model_prompt,
                avg_model_completion_tokens=avg_model_completion,
                judge_prompt_tokens=judge_prompt,
                judge_completion_tokens=judge_completion,
                pass_ci_low=ci_low,
                pass_ci_high=ci_high,
                repeat_pass_rates=repeat_pass_rates,
                repeat_stddev=repeat_stddev,
                flip_case_ids=flip_case_ids,
                flip_rate=flip_rate,
            )
        )
    return stats


def fmt(value, spec) -> str:
    """Shared numeric-or-'n/a' formatter, also used by blog.py's tables/conditions rendering."""
    return format(value, spec) if value is not None else "n/a"


def prompt_file_char_len(meta: dict, *, root: Path | None = None) -> int | None:
    """Character length of ``meta['prompt_file']`` (any run; APO-21)."""
    prompt_file = meta.get("prompt_file")
    if not prompt_file:
        return None
    path = (root or REPO_ROOT) / prompt_file
    if not path.is_file():
        return None
    return len(path.read_text(encoding="utf-8"))


def prompt_template_char_len(meta: dict, *, root: Path | None = None) -> int | None:
    """Return prompt template length for variant runs (APO-20 / issue #79)."""
    if not meta.get("variant"):
        return None
    return prompt_file_char_len(meta, root=root)


def render_markdown(run_id: str, meta: dict, stats: list[AliasStats], warnings_lines: list[str]) -> str:
    lines = [f"# Report: {run_id}", ""]
    lines.append(f"- task: `{meta.get('task_name')}`")
    lines.append(f"- answer_type: `{meta.get('answer_type')}`")
    lines.append(f"- variant: `{meta.get('variant') or '(base)'}`")
    lines.append(f"- created_at: {meta.get('created_at')}")
    lines.append(f"- repeat: {meta.get('repeat')}  limit: {meta.get('limit')}")
    lines.append(f"- promptfoo config: `{meta.get('promptfoo_config_path')}`")
    lines.append(f"- promptfoo version: `{meta.get('promptfoo_version')}`")
    lines.append("")

    for w in warnings_lines:
        lines.append(f"> ⚠ {w}")
    if warnings_lines:
        lines.append("")

    lines.append(
        "| alias | n | pass_rate | pass_95ci | repeat_stddev | total_cost_usd | avg_cost_usd "
        "| avg_prompt_tokens | avg_output_tokens | model_tokens | judge_tokens "
        "| p50_latency_ms | cache_rate | errors |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        ci = (
            f"[{format(s.pass_ci_low, '.1%')}, {format(s.pass_ci_high, '.1%')}]"
            if s.pass_ci_low is not None and s.pass_ci_high is not None
            else "n/a"
        )
        model_tok = s.model_prompt_tokens + s.model_completion_tokens
        judge_tok = s.judge_prompt_tokens + s.judge_completion_tokens
        lines.append(
            "| {alias} | {n} | {pass_rate} | {ci} | {stddev} | {total_cost} | {avg_cost} "
            "| {avg_prompt} | {avg_output} | {model_tok} | {judge_tok} | {p50} | {cache} | {errors} |".format(
                alias=s.alias,
                n=s.n,
                pass_rate=fmt(s.pass_rate, ".1%"),
                ci=ci,
                stddev=fmt(s.repeat_stddev, ".1%"),
                total_cost=fmt(s.total_cost_usd, ".4f"),
                avg_cost=fmt(s.avg_cost_usd, ".6f"),
                avg_prompt=fmt(s.avg_model_prompt_tokens, ".1f"),
                avg_output=fmt(s.avg_model_completion_tokens, ".1f"),
                model_tok=model_tok,
                judge_tok=judge_tok,
                p50=fmt(s.p50_latency_ms, ".0f"),
                cache=fmt(s.cache_rate, ".1%"),
                errors=s.error_count,
            )
        )
    lines.append("")
    lines.append(
        "> pass_95ci: Wilson score interval over all graded rows (optimistic when repeat>1 -- "
        "repeats of the same case are not independent samples). repeat_stddev: stddev of the "
        "per-repeat pass rates; n/a when repeat=1."
    )
    lines.append(
        "> avg_prompt_tokens / avg_output_tokens: per-row average model prompt/completion tokens "
        "from promptfoo response.tokenUsage; n/a when the provider reports no token usage "
        "(e.g. Ollama or missing tokenUsage)."
    )
    lines.append(
        "> model_tokens / judge_tokens: total (prompt+completion) tokens consumed by the "
        "evaluated model / llm-rubric grader respectively. judge_tokens is 0 for answer_types "
        "without an LLM judge. Useful when unit prices are $0 (e.g. local/Ollama models)."
    )
    prompt_len = prompt_template_char_len(meta)
    if prompt_len is not None:
        lines.append(f"> variant prompt template: {prompt_len} characters (`{meta.get('prompt_file')}`)")
    lines.append("")

    # repeat stability section, only when at least one alias actually has repeats
    repeat_stats = [s for s in stats if s.repeat_pass_rates]
    if repeat_stats:
        lines.append("## Repeat stability (run-to-run variance)")
        lines.append("")
        lines.append('A case "flips" when it is graded both pass and fail across repeats of the same run.')
        lines.append("")
        for s in repeat_stats:
            rates = ", ".join(format(r, ".1%") for r in s.repeat_pass_rates)
            lines.append(
                f"- **{s.alias}**: per-repeat pass rates [{rates}], stddev {fmt(s.repeat_stddev, '.1%')}, "
                f"flip rate {fmt(s.flip_rate, '.1%')} ({len(s.flip_case_ids)} case(s))"
            )
            if s.flip_case_ids:
                shown = ", ".join(s.flip_case_ids[:20])
                more = f" (+{len(s.flip_case_ids) - 20} more)" if len(s.flip_case_ids) > 20 else ""
                lines.append(f"  - flipped: {shown}{more}")
        lines.append("")
    return "\n".join(lines)


def report(run_id: str, paths: TaskPaths) -> Path:
    run_dir = paths.runs_dir / run_id
    output_path = run_dir / "output.json"
    meta_path = run_dir / "meta.json"
    if not output_path.exists() or not meta_path.exists():
        raise ReportError(f"run {run_id!r} not found under {paths.runs_dir} (expected output.json and meta.json)")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parsed = parse_promptfoo_output(output_path)
    stats = compute_alias_stats(parsed.results)

    warnings_lines: list[str] = []
    grader_meta = meta.get("grader") or {}
    judge_meta = meta.get("judge") or {}
    uses_judge = grader_meta.get("type") == "llm-rubric" or (not grader_meta and meta.get("answer_type") == "text")
    calibration_status = grader_meta.get("calibration_status", judge_meta.get("calibration_status"))
    judge_provider = grader_meta.get("provider", judge_meta.get("provider"))
    if uses_judge and calibration_status != "calibrated":
        # Fall back to task-level calibration.json when meta is still uncalibrated (issue #100).
        from evalloop.calibrate import apply_calibration_to_meta, load_task_calibration

        snap = load_task_calibration(paths, judge_provider=judge_provider)
        if snap and snap.get("calibration_status") == "calibrated":
            if apply_calibration_to_meta(meta, snap):
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            calibration_status = "calibrated"
            grader_meta = meta.get("grader") or grader_meta
            judge_meta = meta.get("judge") or judge_meta
        else:
            warnings_lines.append(
                "uncalibrated/low-agreement judge: run `evalloop calibrate` before trusting these pass rates "
                f"(judge={judge_provider})"
            )
    warnings_lines.extend(f"promptfoo output.json parser warning: {w}" for w in parsed.warnings)

    markdown = render_markdown(run_id, meta, stats, warnings_lines)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = paths.reports_dir / f"{run_id}.md"
    report_path.write_text(markdown, encoding="utf-8")
    print(f"[report] wrote {report_path}")
    return report_path
