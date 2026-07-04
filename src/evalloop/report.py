"""results/runs/{run_id}/output.json -> results/reports/{run_id}.md

A per-model matrix: pass rate, cost, latency, cache rate. This is the
"モデル×精度×コスト×レイテンシ" half of the two final deliverables in
README.md section 1; the failure x category pivot lives in analyze.py.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from evalloop.schemas import CaseResult, parse_promptfoo_output

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "results" / "runs"
REPORTS_DIR = REPO_ROOT / "results" / "reports"


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


def _percentile50(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def compute_alias_stats(results: list[CaseResult]) -> list[AliasStats]:
    by_alias: dict[str, list[CaseResult]] = {}
    for r in results:
        by_alias.setdefault(r.alias or "unknown", []).append(r)

    stats: list[AliasStats] = []
    for alias, rows in sorted(by_alias.items()):
        passed_flags = [r.passed for r in rows if r.passed is not None]
        pass_rate = (sum(1 for p in passed_flags if p) / len(passed_flags)) if passed_flags else None
        costs = [r.cost or 0.0 for r in rows]
        latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
        cache_hits = sum(1 for r in rows if r.cached)
        errors = sum(1 for r in rows if r.error)
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
            )
        )
    return stats


def fmt(value, spec) -> str:
    """Shared numeric-or-'n/a' formatter, also used by blog.py's tables/conditions rendering."""
    return format(value, spec) if value is not None else "n/a"


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

    lines.append("| alias | n | pass_rate | total_cost_usd | avg_cost_usd | p50_latency_ms | cache_rate | errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        lines.append(
            "| {alias} | {n} | {pass_rate} | {total_cost} | {avg_cost} | {p50} | {cache} | {errors} |".format(
                alias=s.alias,
                n=s.n,
                pass_rate=fmt(s.pass_rate, ".1%"),
                total_cost=fmt(s.total_cost_usd, ".4f"),
                avg_cost=fmt(s.avg_cost_usd, ".6f"),
                p50=fmt(s.p50_latency_ms, ".0f"),
                cache=fmt(s.cache_rate, ".1%"),
                errors=s.error_count,
            )
        )
    lines.append("")
    return "\n".join(lines)


def report(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    output_path = run_dir / "output.json"
    meta_path = run_dir / "meta.json"
    if not output_path.exists() or not meta_path.exists():
        raise ReportError(f"run {run_id!r} not found under {RUNS_DIR} (expected output.json and meta.json)")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parsed = parse_promptfoo_output(output_path)
    stats = compute_alias_stats(parsed.results)

    warnings_lines: list[str] = []
    judge_meta = meta.get("judge") or {}
    uses_judge = meta.get("answer_type") == "text"
    if uses_judge and judge_meta.get("calibration_status") != "calibrated":
        warnings_lines.append(
            "uncalibrated/low-agreement judge: run `evalloop calibrate` before trusting these pass rates "
            f"(judge={judge_meta.get('provider')})"
        )
    warnings_lines.extend(f"promptfoo output.json parser warning: {w}" for w in parsed.warnings)

    markdown = render_markdown(run_id, meta, stats, warnings_lines)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{run_id}.md"
    report_path.write_text(markdown, encoding="utf-8")
    print(f"[report] wrote {report_path}")
    return report_path
