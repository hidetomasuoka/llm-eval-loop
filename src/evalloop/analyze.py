"""Failure analysis: failures / cluster / pivot.

    failures --run RUN_ID   output.json -> results/runs/{run_id}/failures.jsonl
                                         -> appends open-coding rows to data/notes.csv
    cluster  --notes ...    data/notes.csv -> data/taxonomy.draft.yaml (never touches taxonomy.yaml)
    pivot    --run RUN_ID   output.json + data/taxonomy.yaml -> reports/pivot_{run_id}.md

`cluster`'s taxonomy proposal still goes through promptfoo (a throwaway
single-provider eval using judge.provider), per the architecture rule that
Python never calls a model provider directly -- see README.md section 2.
"""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from evalloop import build as build_mod
from evalloop import run as run_mod
from evalloop.schemas import load_config, parse_promptfoo_output

REPO_ROOT = build_mod.REPO_ROOT
NOTES_PATH = REPO_ROOT / "data" / "notes.csv"
TAXONOMY_PATH = REPO_ROOT / "data" / "taxonomy.yaml"
TAXONOMY_DRAFT_PATH = REPO_ROOT / "data" / "taxonomy.draft.yaml"
REPORTS_DIR = REPO_ROOT / "results" / "reports"

NOTES_COLUMNS = ["case_id", "model", "input_head", "output_head", "expected", "note"]
HEAD_LEN = 50


class AnalyzeError(RuntimeError):
    pass


def _head(value, n=HEAD_LEN) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "..."


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------


def failures(run_id: str) -> tuple[Path, Path]:
    output_path = run_mod.RUNS_DIR / run_id / "output.json"
    if not output_path.exists():
        raise AnalyzeError(f"run {run_id!r} has no output.json at {output_path}")
    parsed = parse_promptfoo_output(output_path)

    failing = [r for r in parsed.results if r.passed is False or r.error]
    failures_path = run_mod.RUNS_DIR / run_id / "failures.jsonl"
    with failures_path.open("w", encoding="utf-8") as f:
        for r in failing:
            f.write(
                json.dumps(
                    {
                        "case_id": r.case_id,
                        "alias": r.alias,
                        "category": r.category,
                        "expected": r.expected,
                        "output": r.output,
                        "reason": r.reason,
                        "error": r.error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    existing_keys: set[tuple[str, str]] = set()
    notes_exists = NOTES_PATH.exists()
    if notes_exists:
        with NOTES_PATH.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row.get("case_id", ""), row.get("model", "")))

    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_rows = [
        {
            "case_id": r.case_id or "",
            "model": r.alias or "",
            "input_head": "",  # output.json doesn't carry the original input text; fill from golden.jsonl if needed
            "output_head": _head(r.output or r.error or ""),
            "expected": _head(r.expected) if r.expected is not None else "",
            "note": "",
        }
        for r in failing
        if (r.case_id or "", r.alias or "") not in existing_keys
    ]
    with NOTES_PATH.open("a" if notes_exists else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NOTES_COLUMNS)
        if not notes_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"[failures] {len(failing)} failing result(s) -> {failures_path}")
    print(f"[failures] appended {len(new_rows)} new row(s) to {NOTES_PATH} (fill in the `note` column by hand)")
    return failures_path, NOTES_PATH


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------


_CLUSTER_PROMPT_TEMPLATE = """あなたはLLM評価の失敗分析を行うアナリストです。
以下は、モデルが誤答したケースのメモ一覧です（CSVのnotes.csvから抽出、JSON配列）。

{{notes_json}}

これらのメモを読み、失敗の原因ごとにカテゴリ分けするタクソノミー案を作成してください。
出力は次のスキーマに厳密に従うJSON一つだけとしてください（説明文やコードブロック記法は不要です）。

{
  "categories": [{"id": "短い英数字ID", "name": "カテゴリ名（日本語）", "definition": "定義（日本語、1-2文）"}],
  "assignments": {"case_id": "categories[].idのいずれか"}
}

すべてのメモのcase_idについて必ずassignmentsに1件割り当ててください。
"""


def _run_cluster_llm(notes_rows: list[dict], judge_provider: str, judge_supports_sampling: bool = True) -> dict:
    # mirror build.py: providers with supports_sampling_params=false reject
    # temperature with HTTP 400
    provider_config: dict = {}
    if judge_supports_sampling:
        provider_config["temperature"] = 0.2
    provider_config["max_tokens"] = 2048
    promptfoo_config = {
        "description": "evalloop cluster (taxonomy draft)",
        "providers": [{"id": judge_provider, "label": "cluster_judge", "config": provider_config}],
        "prompts": [_CLUSTER_PROMPT_TEMPLATE],
        "tests": [{"vars": {"notes_json": json.dumps(notes_rows, ensure_ascii=False)}}],
        "defaultTest": {"assert": [{"type": "is-json"}]},
    }

    tmp_name = "_cluster_tmp.yaml"
    tmp_config_path = build_mod.PROMPTFOO_DIR / tmp_name
    build_mod.PROMPTFOO_DIR.mkdir(parents=True, exist_ok=True)
    tmp_config_path.write_text(yaml.safe_dump(promptfoo_config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    try:
        with tempfile.TemporaryDirectory(prefix="evalloop-cluster-") as tmp_dir:
            output_path = Path(tmp_dir) / "cluster_output.json"
            proc = run_mod.run_promptfoo_eval(tmp_config_path, output_path, repeat=1, no_cache=True, timeout_s=600)
            if not output_path.exists():
                raise AnalyzeError(f"cluster LLM call failed (exit {proc.returncode}).\nstderr:\n{proc.stderr}")
            parsed = parse_promptfoo_output(output_path)
    finally:
        tmp_config_path.unlink(missing_ok=True)

    if not parsed.results or not parsed.results[0].output:
        raise AnalyzeError("cluster LLM call returned no output")

    raw_output = parsed.results[0].output
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise AnalyzeError(f"cluster LLM output was not valid JSON: {e}\nraw output:\n{raw_output}") from e


def cluster(notes_path: str | Path | None = None, config_path: str | Path = REPO_ROOT / "config.yaml") -> Path:
    notes_path = Path(notes_path) if notes_path is not None else NOTES_PATH
    if not notes_path.exists():
        raise AnalyzeError(f"{notes_path} not found; run `evalloop failures --run RUN_ID` first")

    with notes_path.open(encoding="utf-8", newline="") as f:
        notes_rows = list(csv.DictReader(f))
    if not notes_rows:
        raise AnalyzeError(f"{notes_path} has no rows to cluster")

    cfg = load_config(config_path)
    # config.yaml only carries supports_sampling_params on models[]; if the
    # judge provider also appears there, honor that entry's flag (empty match
    # -> True, i.e. keep sending temperature as before)
    judge_supports_sampling = all(m.supports_sampling_params for m in cfg.models if m.provider == cfg.judge.provider)
    taxonomy = _run_cluster_llm(notes_rows, cfg.judge.provider, judge_supports_sampling)

    if "categories" not in taxonomy or "assignments" not in taxonomy:
        raise AnalyzeError(f"cluster LLM output missing 'categories'/'assignments' keys: {taxonomy}")

    TAXONOMY_DRAFT_PATH.write_text(
        yaml.safe_dump(taxonomy, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"[cluster] wrote {TAXONOMY_DRAFT_PATH} ({len(taxonomy['categories'])} categories, "
          f"{len(taxonomy['assignments'])} assignments)")
    print(f"[cluster] this NEVER overwrites {TAXONOMY_PATH} -- merge by hand, then `evalloop pivot` reads that file")
    return TAXONOMY_DRAFT_PATH


# ---------------------------------------------------------------------------
# pivot
# ---------------------------------------------------------------------------


@dataclass
class PivotCell:
    category: str
    alias: str
    count: int


def load_taxonomy(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else TAXONOMY_PATH
    if not path.exists():
        raise AnalyzeError(
            f"{path} not found. Run `evalloop cluster` then hand-merge data/taxonomy.draft.yaml into {path}"
        )
    with path.open(encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f) or {}
    taxonomy.setdefault("categories", [])
    taxonomy.setdefault("assignments", {})
    return taxonomy


def pivot(run_id: str, taxonomy_path: str | Path | None = None) -> Path:
    output_path = run_mod.RUNS_DIR / run_id / "output.json"
    if not output_path.exists():
        raise AnalyzeError(f"run {run_id!r} has no output.json at {output_path}")
    parsed = parse_promptfoo_output(output_path)
    taxonomy = load_taxonomy(taxonomy_path)
    assignments: dict[str, str] = taxonomy["assignments"]
    category_names = {c["id"]: c.get("name", c["id"]) for c in taxonomy["categories"]}

    failing = [r for r in parsed.results if r.passed is False]
    counts: dict[tuple[str, str], int] = {}
    aliases: set[str] = set()
    categories_seen: set[str] = set()
    for r in failing:
        category_id = assignments.get(r.case_id or "", "unassigned")
        alias = r.alias or "unknown"
        aliases.add(alias)
        categories_seen.add(category_id)
        counts[(category_id, alias)] = counts.get((category_id, alias), 0) + 1

    sorted_aliases = sorted(aliases)
    sorted_categories = sorted(categories_seen, key=lambda c: (c == "unassigned", c))

    lines = [f"# Failure pivot: {run_id}", ""]
    lines.append(f"- total failing results: {len(failing)}")
    lines.append(f"- taxonomy: `{taxonomy_path}`")
    lines.append("")
    header = "| category | " + " | ".join(sorted_aliases) + " | total |"
    sep = "|---|" + "---:|" * (len(sorted_aliases) + 1)
    lines.append(header)
    lines.append(sep)
    for cat in sorted_categories:
        name = category_names.get(cat, "unassigned (未割当)" if cat == "unassigned" else cat)
        row_counts = [counts.get((cat, alias), 0) for alias in sorted_aliases]
        lines.append(f"| {name} | " + " | ".join(str(c) for c in row_counts) + f" | {sum(row_counts)} |")
    lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"pivot_{run_id}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[pivot] wrote {report_path}")
    return report_path
