"""Publish-guarded blog export.

    evalloop blog --runs A[,B] [--slug NAME] -> blog/{YYYYMMDD}_{slug}/
        fig01_accuracy_by_model.{png,svg}
        fig02_cost_vs_accuracy.{png,svg}
        fig03_failure_heatmap.{png,svg}   (skipped if data/taxonomy.yaml is missing/empty)
        tables.md
        conditions.md
        article_draft.md

Iron rule #7 / spec section 9.3: nothing is written into blog/ unless every
guard below passes. Generation happens in a staging directory first; it is
only moved into blog/ after the secret/path scan succeeds, so a failed guard
never leaves partial output behind for someone to accidentally publish.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from evalloop import analyze as analyze_mod
from evalloop import build as build_mod
from evalloop import report as report_mod
from evalloop import run as run_mod
from evalloop.schemas import load_config, load_golden_jsonl, parse_promptfoo_output

REPO_ROOT = build_mod.REPO_ROOT
BLOG_DIR = REPO_ROOT / "blog"
REVIEW_COMMENT = "<!-- 公開前に固有情報がないか目視確認 -->"

ALLOWED_SOURCES_DEFAULT = {"self-made"}

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

_CJK_FONT_CANDIDATES = [
    # macOS (this project's primary target)
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Hiragino Maru Gothic ProN",
    # cross-platform / Linux
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "TakaoPGothic",
    # Windows
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
]


class BlogGuardError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# guard 1: every golden.jsonl case must be self-made / explicitly licensed
# ---------------------------------------------------------------------------


def check_source_guard(golden_cases, allowed_sources: set[str] = frozenset(ALLOWED_SOURCES_DEFAULT)) -> None:
    violations = sorted(c.id for c in golden_cases if c.source not in allowed_sources)
    if violations:
        raise BlogGuardError(
            "publish guard failed: the following golden.jsonl case(s) have a "
            f"meta.source not in {sorted(allowed_sources)}: {violations}. "
            "Fix the source field or add the license to config before publishing."
        )


# ---------------------------------------------------------------------------
# guard 2: no secrets / home-directory absolute paths in generated output
# ---------------------------------------------------------------------------


def check_secret_guard(staging_dir: Path) -> None:
    home = str(Path.home())
    findings: list[str] = []
    for path in sorted(staging_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".png"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path.relative_to(staging_dir)}: matches secret pattern {pattern.pattern!r}")
        if home and home in text:
            findings.append(f"{path.relative_to(staging_dir)}: contains home directory path {home!r}")

    if findings:
        raise BlogGuardError("publish guard failed: potential secret/local-path leak(s):\n" + "\n".join(findings))


# ---------------------------------------------------------------------------
# font handling (no tofu boxes)
# ---------------------------------------------------------------------------


def find_cjk_font() -> str | None:
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return None


@dataclass
class Labels:
    """Figure text in Japanese if a CJK font was found, else an English
    fallback -- see README.md section 9.1 ("豆腐の混入防止").
    """

    accuracy: str
    cost: str
    model: str
    category: str
    unassigned: str


def _labels(has_cjk_font: bool) -> Labels:
    if has_cjk_font:
        return Labels(accuracy="精度", cost="コスト (USD/件, 対数軸)", model="モデル", category="失敗カテゴリ", unassigned="未割当")
    return Labels(accuracy="Accuracy", cost="Cost (USD/case, log scale)", model="Model", category="Failure category", unassigned="unassigned")


# ---------------------------------------------------------------------------
# per-run data loading
# ---------------------------------------------------------------------------


@dataclass
class RunData:
    run_id: str
    meta: dict
    stats: list  # list[report_mod.AliasStats]


def _load_run_data(run_id: str) -> RunData:
    run_dir = run_mod.RUNS_DIR / run_id
    output_path = run_dir / "output.json"
    meta_path = run_dir / "meta.json"
    if not output_path.exists() or not meta_path.exists():
        raise BlogGuardError(f"run {run_id!r} not found under {run_mod.RUNS_DIR}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parsed = parse_promptfoo_output(output_path)
    stats = report_mod.compute_alias_stats(parsed.results)
    tier_order = {m["alias"]: i for i, m in enumerate(meta.get("models", []))}
    stats.sort(key=lambda s: tier_order.get(s.alias, 999))
    return RunData(run_id=run_id, meta=meta, stats=stats)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _save_fig(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


_TIER_COLORS = {"local": "#8c8c8c", "small": "#4c72b0", "mid": "#55a868", "large": "#c44e52", "frontier": "#8172b2"}


def _tier_for_alias(meta: dict, alias: str) -> str:
    for m in meta.get("models", []):
        if m["alias"] == alias:
            return m.get("tier", "unknown")
    return "unknown"


def make_fig01_accuracy_by_model(runs: list[RunData], out_dir: Path, labels: Labels) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    aliases = [s.alias for s in runs[-1].stats]
    width = 0.8 / len(runs)
    for i, run in enumerate(runs):
        by_alias = {s.alias: s for s in run.stats}
        values = [(by_alias[a].pass_rate or 0.0) if a in by_alias else 0.0 for a in aliases]
        colors = [_TIER_COLORS.get(_tier_for_alias(run.meta, a), "#333333") for a in aliases]
        xs = [x + i * width for x in range(len(aliases))]
        alpha = 1.0 if i == len(runs) - 1 else 0.5

        # Wilson 95% CI error bars (issue #11) -- zero-length when a stat has
        # no CI (e.g. nothing graded), so the bar still renders
        def _err(alias: str, side: str) -> float:
            s = by_alias.get(alias)
            if s is None or s.pass_rate is None or s.pass_ci_low is None or s.pass_ci_high is None:
                return 0.0
            return (s.pass_rate - s.pass_ci_low) if side == "low" else (s.pass_ci_high - s.pass_rate)

        yerr = [[_err(a, "low") for a in aliases], [_err(a, "high") for a in aliases]]
        ax.bar(
            xs, values, width=width, color=colors, alpha=alpha, label=run.run_id,
            yerr=yerr, capsize=3, error_kw={"ecolor": "#333333", "alpha": 0.7},
        )
    ax.set_xticks([x + width * (len(runs) - 1) / 2 for x in range(len(aliases))])
    ax.set_xticklabels(aliases, rotation=30, ha="right")
    ax.set_ylabel(labels.accuracy)
    ax.set_ylim(0, 1.05)
    if len(runs) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig01_accuracy_by_model")


def make_fig02_cost_vs_accuracy(runs: list[RunData], out_dir: Path, labels: Labels) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    positions: dict[str, tuple[float, float]] = {}
    for run_idx, run in enumerate(runs):
        for s in run.stats:
            cost_per_case = s.avg_cost_usd if s.avg_cost_usd > 0 else 1e-6
            acc = s.pass_rate or 0.0
            color = _TIER_COLORS.get(_tier_for_alias(run.meta, s.alias), "#333333")
            marker = "o" if run_idx == len(runs) - 1 else "x"
            ax.scatter(cost_per_case, acc, color=color, marker=marker, s=60, zorder=3)
            ax.annotate(s.alias, (cost_per_case, acc), fontsize=8, xytext=(4, 4), textcoords="offset points")
            if run_idx == 0:
                positions[s.alias] = (cost_per_case, acc)
            elif s.alias in positions:
                x0, y0 = positions[s.alias]
                ax.annotate(
                    "", xy=(cost_per_case, acc), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.6},
                )
    ax.set_xscale("log")
    ax.set_xlabel(labels.cost)
    ax.set_ylabel(labels.accuracy)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig02_cost_vs_accuracy")


def make_fig03_failure_heatmap(run: RunData, out_dir: Path, labels: Labels) -> bool:
    try:
        taxonomy = analyze_mod.load_taxonomy()
    except analyze_mod.AnalyzeError:
        return False
    if not taxonomy.get("categories"):
        return False

    parsed = parse_promptfoo_output(run_mod.RUNS_DIR / run.run_id / "output.json")
    assignments = taxonomy["assignments"]
    category_names = {c["id"]: c.get("name", c["id"]) for c in taxonomy["categories"]}
    aliases = sorted({s.alias for s in run.stats})
    categories = sorted({assignments.get(r.case_id or "", "unassigned") for r in parsed.results if r.passed is False})
    if not categories:
        return False

    matrix = [[0 for _ in aliases] for _ in categories]
    for r in parsed.results:
        if r.passed is not False:
            continue
        cat = assignments.get(r.case_id or "", "unassigned")
        if cat not in categories or r.alias not in aliases:
            continue
        matrix[categories.index(cat)][aliases.index(r.alias)] += 1

    fig, ax = plt.subplots(figsize=(1.2 * len(aliases) + 2, 0.6 * len(categories) + 2))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(aliases)))
    ax.set_xticklabels(aliases, rotation=30, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([category_names.get(c, labels.unassigned if c == "unassigned" else c) for c in categories])
    for i in range(len(categories)):
        for j in range(len(aliases)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label=labels.category)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig03_failure_heatmap")
    return True


# ---------------------------------------------------------------------------
# text outputs
# ---------------------------------------------------------------------------


def render_tables_md(runs: list[RunData]) -> str:
    lines = ["# Tables", ""]
    for run in runs:
        lines.append(f"## {run.run_id}")
        lines.append("")
        lines.append("| model | tier | accuracy | total_cost_usd | p50_latency_ms | cache_rate |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for s in run.stats:
            tier = _tier_for_alias(run.meta, s.alias)
            lines.append(
                f"| {s.alias} | {tier} | {report_mod.fmt(s.pass_rate, '.1%')} | "
                f"{report_mod.fmt(s.total_cost_usd, '.4f')} | {report_mod.fmt(s.p50_latency_ms, '.0f')} | "
                f"{report_mod.fmt(s.cache_rate, '.1%')} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_conditions_md(runs: list[RunData], config, fig03_written: bool) -> str:
    primary = runs[-1]
    prompt_path = REPO_ROOT / primary.meta.get("prompt_file", config.task.prompt_file)
    prompt_sha8 = primary.meta.get("prompt_sha256", "")[:8] if primary.meta.get("prompt_sha256") else (
        hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:8] if prompt_path.exists() else "unknown"
    )
    total_cost = sum(s.total_cost_usd for s in primary.stats)
    jpy = config.blog.jpy_per_usd

    lines = [
        "# Reproducibility conditions",
        "",
        f"- experiment date: {datetime.now().strftime('%Y-%m-%d')}",
        "- models:",
    ]
    for m in primary.meta.get("models", []):
        lines.append(f"  - `{m['provider']}` (alias: {m['alias']}, tier: {m['tier']})")
    n_test_cases = max((s.n for s in primary.stats), default=0)
    lines += [
        f"- test cases (approx, n per model): {n_test_cases}",
        f"- repeat: {primary.meta.get('repeat')}",
        f"- temperature: {config.run.temperature}",
        f"- prompt sha256 (first 8): `{prompt_sha8}`",
        f"- judge: `{primary.meta.get('judge', {}).get('provider')}`"
        f" (calibration: {primary.meta.get('judge', {}).get('calibration_status', 'uncalibrated')},"
        f" agreement: {report_mod.fmt(primary.meta.get('judge', {}).get('agreement_rate'), '.1%')})",
        f"- total cost: ${total_cost:.4f}" + (f" (~{'{:,.0f}'.format(total_cost * jpy)} JPY)" if jpy else ""),
        f"- promptfoo version: `{primary.meta.get('promptfoo_version')}`",
        f"- dspy version: `{__import__('dspy').__version__}`",
        f"- fig03 (failure heatmap): {'included' if fig03_written else 'skipped (data/taxonomy.yaml not defined yet)'}",
        "",
        "## reproduce",
        "```bash",
    ]
    config_flag = f" --config {primary.meta['config_path']}" if primary.meta.get("config_path", "config.yaml") != "config.yaml" else ""
    # mirror build.py's iron-rule-#2 check: for a same-judge text config the
    # copy-pasted command aborts unless --allow-same-judge is included.
    # Use primary.meta (the run snapshot) so that the flag matches the actual
    # config that was used for the run, not the config passed to blog().
    _meta_judge_provider = primary.meta.get("judge", {}).get("provider", "")
    _meta_models = primary.meta.get("models", [])
    same_judge = (
        primary.meta.get("answer_type") == "text"
        and any(m.get("provider") == _meta_judge_provider for m in _meta_models)
    )
    same_judge_flag = " --allow-same-judge" if same_judge else ""
    lines.append(f"evalloop build{config_flag}{same_judge_flag}")
    for run in runs:
        variant_flag = f" --variant {run.meta.get('variant')}" if run.meta.get("variant") else ""
        run_config_flag = (
            f" --config {run.meta['config_path']}" if run.meta.get("config_path", "config.yaml") != "config.yaml" else ""
        )
        lines.append(f"evalloop run{run_config_flag}{variant_flag} --repeat {run.meta.get('repeat')}")
    lines += ["evalloop report " + primary.run_id, "```", ""]
    return "\n".join(lines)


def render_article_draft(runs: list[RunData], config, fig03_written: bool) -> str:
    primary = runs[-1]
    best = max((s for s in primary.stats if s.pass_rate is not None), key=lambda s: s.pass_rate, default=None)
    cheapest_passing = min(
        (s for s in primary.stats if s.pass_rate and s.pass_rate >= config.judge.threshold),
        key=lambda s: s.total_cost_usd,
        default=None,
    )

    lines = [
        f"# {config.task.name}: どのモデルが必要精度を満たすか、それはいくらか",
        "",
        REVIEW_COMMENT,
        "",
        "## 背景",
        "",
        "TODO: このタスクを評価することにした背景・動機を記述する。",
        "",
        "## 手法",
        "",
        "TODO: 構成図（前処理→promptfoo実行→分析→GEPA→再評価→ブログ化）をここに挿入する。",
        f"評価はpromptfooで実行し、判定は{'決定的アサート' if config.task.answer_type == 'label' else 'LLMジャッジ'}を使用した。",
        "",
        "## 結果",
        "",
        "![モデル別精度](./fig01_accuracy_by_model.png)",
        "",
        "![コスト対精度](./fig02_cost_vs_accuracy.png)",
        "",
    ]
    if fig03_written:
        lines += ["![失敗カテゴリ×モデル](./fig03_failure_heatmap.png)", ""]
    else:
        lines += ["(fig03: 失敗タクソノミー未確定のため未生成)", ""]

    if best:
        lines.append(f"最も精度が高かったのは `{best.alias}`（精度 {best.pass_rate:.1%}）だった。")
    if cheapest_passing:
        lines.append(
            f"精度しきい値 {config.judge.threshold:.0%} 相当を満たした中で最も安価だったのは "
            f"`{cheapest_passing.alias}`（1件あたり ${cheapest_passing.avg_cost_usd:.6f}）だった。"
        )
    lines += [
        "",
        "詳細な数値は [tables.md](./tables.md) を参照。",
        "",
        "## 考察",
        "",
        "TODO: 結果から何が言えるか、意外だった点、今後の改善余地を記述する。",
        "",
        "## 限界と注意",
        "",
        "TODO: サンプルサイズ・タスクの一般化可能性・ジャッジ校正状況などの限界を記述する。",
        f"（このデータセットは {config.task.name} 用の自作データであり、"
        f"ジャッジは {'未校正/低一致率' if primary.meta.get('judge', {}).get('calibration_status') != 'calibrated' else '校正済み'} である点に注意）",
        "",
        "## 再現手順",
        "",
        "[conditions.md](./conditions.md) の内容を参照。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def blog(
    run_ids: list[str],
    slug: str | None = None,
    config_path: str | Path = REPO_ROOT / "config.yaml",
) -> Path:
    if not run_ids or len(run_ids) > 2:
        raise BlogGuardError("--runs must name exactly 1 or 2 run_ids")

    config = load_config(config_path)
    golden_cases = load_golden_jsonl(build_mod.GOLDEN_PATH)
    check_source_guard(golden_cases, allowed_sources=frozenset(config.blog.allowed_sources))  # guard 1

    runs = [_load_run_data(rid) for rid in run_ids]
    found_font = find_cjk_font()
    has_cjk_font = found_font is not None
    if has_cjk_font:
        # Merely finding the font isn't enough -- matplotlib still defaults to
        # DejaVu Sans (no CJK glyphs) unless font.family is set explicitly.
        # Without this, Japanese labels silently render as missing-glyph boxes.
        plt.rcParams["font.family"] = found_font
        plt.rcParams["axes.unicode_minus"] = False  # most CJK fonts lack the unicode minus glyph
    else:
        print("[blog] WARNING: no CJK-capable font found; figure text will fall back to English to avoid tofu boxes")
    labels = _labels(has_cjk_font)

    with tempfile.TemporaryDirectory(prefix="evalloop-blog-") as tmp:
        staging_dir = Path(tmp)

        make_fig01_accuracy_by_model(runs, staging_dir, labels)
        make_fig02_cost_vs_accuracy(runs, staging_dir, labels)
        fig03_written = make_fig03_failure_heatmap(runs[-1], staging_dir, labels)
        if not fig03_written:
            print("[blog] fig03 skipped: data/taxonomy.yaml not defined yet (run `evalloop cluster` then merge it)")

        (staging_dir / "tables.md").write_text(render_tables_md(runs), encoding="utf-8")
        (staging_dir / "conditions.md").write_text(render_conditions_md(runs, config, fig03_written), encoding="utf-8")
        (staging_dir / "article_draft.md").write_text(render_article_draft(runs, config, fig03_written), encoding="utf-8")

        check_secret_guard(staging_dir)  # guard 2, only after everything is written

        slug_name = slug or config.blog.slug_prefix
        date_str = datetime.now().strftime("%Y%m%d")
        final_dir = BLOG_DIR / f"{date_str}_{slug_name}"
        if final_dir.exists():
            shutil.rmtree(final_dir)  # re-running blog for the same day/slug regenerates, doesn't accumulate stale files
        BLOG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging_dir, final_dir)

    print(f"[blog] wrote {final_dir}")
    print("[blog] reminder: `promptfoo share` is never used by this project; review the article draft before publishing")
    return final_dir
