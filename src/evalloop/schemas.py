"""Typed data models for every file format evalloop reads or writes.

Centralizing schemas here means the rest of the codebase (build/run/report/
calibrate/analyze/optimize/blog) never touches raw dicts from config.yaml,
golden.jsonl, human_labels.jsonl or promptfoo's output.json directly.

promptfoo's output.json shape has known version-to-version variance, so
`parse_promptfoo_output` is defensive: it tries multiple known layouts and
emits a warning (rather than raising) when an expected key is missing.
"""

from __future__ import annotations

import json
import os
import warnings as _warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from evalloop import paths as paths_mod

VALID_SPLITS = {"train", "test"}
VALID_ANSWER_TYPES = {"label", "json", "text"}


class SchemaError(ValueError):
    """Raised when an input file violates its expected schema."""


# ---------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    name: str
    answer_type: str
    prompt_file: str
    labels: list[str] = field(default_factory=list)
    json_schema_file: str | None = None

    def __post_init__(self) -> None:
        if self.answer_type not in VALID_ANSWER_TYPES:
            raise SchemaError(
                f"task.answer_type must be one of {sorted(VALID_ANSWER_TYPES)}, got {self.answer_type!r}"
            )
        if self.answer_type == "label" and not self.labels:
            raise SchemaError("task.answer_type=label requires a non-empty task.labels list")
        if self.answer_type == "json" and not self.json_schema_file:
            raise SchemaError("task.answer_type=json requires task.json_schema_file")


@dataclass
class ModelConfig:
    provider: str
    alias: str
    tier: str
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    # claude-opus-4-8 / claude-fable-5 など、samplingパラメータ(temperature等)を
    # HTTP 400で拒否するモデルは false にする。build時にprovider configへ
    # temperatureを出力しない（max_tokensはどのモデルも受け付けるので常に出力）
    supports_sampling_params: bool = True


@dataclass
class RunConfig:
    repeat: int = 1
    temperature: float = 0.0
    max_tokens: int = 1024
    cost_warn_usd: float = 3.0


@dataclass
class JudgeConfig:
    provider: str
    threshold: float = 0.8
    agreement_threshold: float = 0.85
    rubric_file: str = "prompts/base/judge_rubric.txt"


@dataclass
class OptimizeConfig:
    target_alias: str
    reflection_provider: str
    auto: str = "light"


@dataclass
class BlogConfig:
    jpy_per_usd: float = 150.0
    slug_prefix: str = "llm-eval"
    # README.md 9.3.1: "self-made" またはこのリストで許可したライセンス値のみ公開可
    allowed_sources: list[str] = field(default_factory=lambda: ["self-made"])


@dataclass
class Config:
    task: TaskConfig
    models: list[ModelConfig]
    run: RunConfig
    judge: JudgeConfig
    optimize: OptimizeConfig
    blog: BlogConfig
    path: Path

    def alias_by_provider(self) -> dict[str, str]:
        return {m.provider: m.alias for m in self.models}

    def model_by_alias(self, alias: str) -> ModelConfig:
        for m in self.models:
            if m.alias == alias:
                return m
        raise SchemaError(f"unknown model alias {alias!r}; known aliases: {[m.alias for m in self.models]}")


@dataclass
class GlobalConfig:
    """Root config.yaml: the model zoo + run defaults + default_task.
    Everything task-specific lives in tasks/<name>/task.yaml (issue #47)."""

    default_task: str | None
    models: list[ModelConfig]
    run: RunConfig
    path: Path


def _parse_models(models_raw, source: str) -> list[ModelConfig]:
    models = [
        ModelConfig(
            provider=m["provider"],
            alias=m["alias"],
            tier=m.get("tier", "unknown"),
            price_in_per_mtok=float(m.get("price_in_per_mtok", 0.0)),
            price_out_per_mtok=float(m.get("price_out_per_mtok", 0.0)),
            supports_sampling_params=bool(m.get("supports_sampling_params", True)),
        )
        for m in models_raw
    ]
    aliases = [m.alias for m in models]
    if len(aliases) != len(set(aliases)):
        raise SchemaError(f"{source} models[].alias must be unique, got: {aliases}")
    return models


def _parse_run(run_raw, defaults: RunConfig | None = None) -> RunConfig:
    base = defaults or RunConfig()
    run_raw = run_raw or {}
    return RunConfig(
        repeat=int(run_raw.get("repeat", base.repeat)),
        temperature=float(run_raw.get("temperature", base.temperature)),
        max_tokens=int(run_raw.get("max_tokens", base.max_tokens)),
        cost_warn_usd=float(run_raw.get("cost_warn_usd", base.cost_warn_usd)),
    )


def load_global_config(path: str | Path) -> GlobalConfig:
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"global config not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    models_raw = raw.get("models")
    if not models_raw:
        raise SchemaError(f"{path}: must define a non-empty models[] registry")
    models = _parse_models(models_raw, str(path))
    run = _parse_run(raw.get("run"))
    return GlobalConfig(default_task=raw.get("default_task"), models=models, run=run, path=path)


def resolve_task_name(task: str | None, global_config: GlobalConfig) -> str:
    """--task flag > EVALLOOP_TASK env > config.yaml default_task."""
    name = task or os.environ.get("EVALLOOP_TASK") or global_config.default_task
    if not name:
        raise SchemaError(
            "no task specified: pass --task NAME, set EVALLOOP_TASK, or set default_task in config.yaml"
        )
    return name


def load_task(task: str | None = None, root: Path | None = None) -> tuple[Config, paths_mod.TaskPaths]:
    """Resolve a task into (merged Config, TaskPaths).

    The merged Config keeps the same shape the codebase always used, so
    downstream modules don't care about the global/task file split:
    prompt_file / rubric_file / json_schema_file come back as ABSOLUTE paths
    (path convention: tasks/<name>/prompts/task.txt, judge_rubric.txt).
    """
    root = root or paths_mod.REPO_ROOT
    global_config = load_global_config(root / "config.yaml")
    name = resolve_task_name(task, global_config)
    try:
        tp = paths_mod.for_task(name, root)
    except paths_mod.TaskNotFoundError as e:
        raise SchemaError(str(e)) from e

    with tp.task_config.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        task_raw = raw["task"]
        judge_raw = raw["judge"]
        optimize_raw = raw["optimize"]
    except KeyError as e:
        raise SchemaError(f"{tp.task_config} missing required top-level key: {e}") from e

    json_schema_file = task_raw.get("json_schema_file")
    task_cfg = TaskConfig(
        name=name,  # the directory name is the canonical task name
        answer_type=task_raw["answer_type"],
        prompt_file=str(tp.prompt_file),
        labels=task_raw.get("labels") or [],
        json_schema_file=str(tp.task_dir / json_schema_file) if json_schema_file else None,
    )

    # task.yaml's models: is an alias subset of the global registry (omitted = all)
    selection = raw.get("models")
    if selection:
        by_alias = {m.alias: m for m in global_config.models}
        unknown = [a for a in selection if a not in by_alias]
        if unknown:
            raise SchemaError(
                f"{tp.task_config}: models {unknown} not in the global registry "
                f"(known: {sorted(by_alias)})"
            )
        models = [by_alias[a] for a in selection]
    else:
        models = list(global_config.models)

    run = _parse_run(raw.get("run"), defaults=global_config.run)
    judge = JudgeConfig(
        provider=judge_raw["provider"],
        threshold=float(judge_raw.get("threshold", 0.8)),
        agreement_threshold=float(judge_raw.get("agreement_threshold", 0.85)),
        rubric_file=str(tp.rubric_file),  # path convention, not configurable
    )
    optimize = OptimizeConfig(
        target_alias=optimize_raw["target_alias"],
        reflection_provider=optimize_raw["reflection_provider"],
        auto=optimize_raw.get("auto", "light"),
    )
    blog_raw = raw.get("blog") or {}
    blog = BlogConfig(
        jpy_per_usd=float(blog_raw.get("jpy_per_usd", 150.0)),
        slug_prefix=blog_raw.get("slug_prefix", "llm-eval"),
        allowed_sources=blog_raw.get("allowed_sources") or ["self-made"],
    )

    config = Config(task=task_cfg, models=models, run=run, judge=judge, optimize=optimize, blog=blog, path=tp.task_config)
    return config, tp


def restrict_models(config: Config, aliases: list[str]) -> Config:
    """CLI --models: narrow the resolved task's model list (e.g. CI smoke)."""
    known = {m.alias for m in config.models}
    unknown = [a for a in aliases if a not in known]
    if unknown:
        raise SchemaError(f"--models {unknown} not in this task's model list (known: {sorted(known)})")
    picked = [m for m in config.models if m.alias in set(aliases)]
    return replace(config, models=picked)


# ---------------------------------------------------------------------------
# data/golden.jsonl
# ---------------------------------------------------------------------------


@dataclass
class GoldenCase:
    id: str
    input: str
    expected: object  # str for label/text, dict for json
    split: str
    category: str
    difficulty: str | None
    source: str
    raw_meta: dict = field(default_factory=dict)


def load_golden_jsonl(path: str | Path) -> list[GoldenCase]:
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"golden dataset not found: {path}")

    cases: list[GoldenCase] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SchemaError(f"{path}:{lineno}: invalid JSON: {e}") from e

            for required in ("id", "input", "expected", "split", "meta"):
                if required not in row:
                    raise SchemaError(f"{path}:{lineno}: missing required field {required!r}")

            case_id = row["id"]
            if case_id in seen_ids:
                raise SchemaError(f"{path}:{lineno}: duplicate case id {case_id!r}")
            seen_ids.add(case_id)

            split = row["split"]
            if split not in VALID_SPLITS:
                raise SchemaError(f"{path}:{lineno}: split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

            meta = row["meta"]
            if "category" not in meta:
                raise SchemaError(f"{path}:{lineno}: meta.category is required")
            if "source" not in meta:
                raise SchemaError(f"{path}:{lineno}: meta.source is required")

            cases.append(
                GoldenCase(
                    id=case_id,
                    input=row["input"],
                    expected=row["expected"],
                    split=split,
                    category=meta["category"],
                    difficulty=meta.get("difficulty"),
                    source=meta["source"],
                    raw_meta=meta,
                )
            )
    return cases


def assert_split_disjoint(train_ids: set[str], test_ids: set[str]) -> None:
    """Iron rule #1: split separation must hold. Raise loudly if it doesn't."""
    overlap = train_ids & test_ids
    if overlap:
        raise SchemaError(
            "train/test split ID overlap detected (must never happen): " f"{sorted(overlap)}"
        )


# ---------------------------------------------------------------------------
# data/human_labels.jsonl
# ---------------------------------------------------------------------------


@dataclass
class HumanLabel:
    """One human verdict on one model's output for one case.

    (case_id, model_label) is the composite primary key: the same case may be
    labeled once per model, and calibrate joins judge verdicts back on that
    pair -- never on case_id alone.
    """

    case_id: str
    model_label: str
    output_raw: str
    human_verdict: str  # "pass" | "fail"


def load_human_labels(path: str | Path) -> list[HumanLabel]:
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"human labels file not found: {path}")

    labels: list[HumanLabel] = []
    seen_keys: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for required in ("case_id", "model_label", "output_raw", "human_verdict"):
                if required not in row:
                    raise SchemaError(f"{path}:{lineno}: missing required field {required!r}")
            if row["human_verdict"] not in ("pass", "fail"):
                raise SchemaError(
                    f"{path}:{lineno}: human_verdict must be 'pass' or 'fail', got {row['human_verdict']!r}"
                )
            key = (row["case_id"], row["model_label"])
            if key in seen_keys:
                raise SchemaError(
                    f"{path}:{lineno}: duplicate (case_id, model_label) pair {key!r}; "
                    "each case may be labeled at most once per model"
                )
            seen_keys.add(key)
            labels.append(
                HumanLabel(
                    case_id=row["case_id"],
                    model_label=row["model_label"],
                    output_raw=row["output_raw"],
                    human_verdict=row["human_verdict"],
                )
            )
    return labels


# ---------------------------------------------------------------------------
# promptfoo output.json
#
# Confirmed against a real promptfoo output.json (promptfoo/examples/simple-cli,
# schema "version": 3): top-level shape is
#   {"evalId": ..., "results": {"version": 3, "prompts": [...], "results": [...]}}
# and each entry of the inner results[] list carries (at minimum):
#   cost, gradingResult, id, latencyMs, namedScores, prompt, promptId, promptIdx,
#   provider, response, score, success, testCase, testIdx, vars
# gradingResult itself carries: pass, score, reason, tokensUsed{total,prompt,
# completion,cached}, componentResults?, assertion.
#
# promptfoo has changed this shape across major versions before (the "version"
# field exists precisely because of that), so this parser is defensive: it
# tries the known nested layout first, falls back to a flat `results: [...]`
# list, and otherwise warns and returns an empty result set rather than
# raising. Every per-row field access uses .get() with a fallback path so a
# renamed/missing key degrades to a warning, not a crash.
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str | None
    alias: str | None  # provider label; build.py sets label=config alias so this is stable
    provider_id: str | None
    expected: object | None
    category: str | None
    output: str | None
    passed: bool | None
    score: float | None
    reason: str | None
    cost: float | None
    latency_ms: float | None
    cached: bool
    token_usage: dict
    error: str | None
    repeat_index: int
    raw: dict = field(repr=False, default_factory=dict)


@dataclass
class ParsedRun:
    eval_id: str | None
    results: list[CaseResult]
    prompts_meta: list[dict]
    warnings: list[str]


def _nested_get(d: dict, *path: str, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _extract_result_rows(raw: dict) -> tuple[list[dict], list[str]]:
    warns: list[str] = []
    results_field = raw.get("results")

    if isinstance(results_field, dict) and isinstance(results_field.get("results"), list):
        return results_field["results"], warns

    if isinstance(results_field, list):
        warns.append(
            "output.json: 'results' was a flat list rather than the nested "
            "{results:{results:[...]}} shape; using it directly. Verify against "
            "the pinned promptfoo version (run.PROMPTFOO_VERSION) if this looks wrong."
        )
        return results_field, warns

    if isinstance(raw.get("table"), dict) and isinstance(raw["table"].get("body"), list):
        warns.append("output.json: falling back to legacy 'table.body' layout; fields may be incomplete.")
        return raw["table"]["body"], warns

    warns.append("output.json: no recognizable results array found in any known promptfoo output layout.")
    return [], warns


def parse_promptfoo_output(path: str | Path) -> ParsedRun:
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"promptfoo output file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    rows, warns = _extract_result_rows(raw)
    results: list[CaseResult] = []
    # keyed by (case_id, alias): a multi-provider run interleaves providers for
    # the same case, so counting per case_id alone would spread one provider's
    # repeats across indices and break per-alias repeat aggregation (report.py)
    repeat_counters: dict[tuple[str, str | None], int] = {}

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            warns.append(f"output.json results[{i}] is not an object; skipping")
            continue

        row_vars = row.get("vars") or _nested_get(row, "testCase", "vars", default={}) or {}
        case_id = row_vars.get("case_id")
        if case_id is None:
            warns.append(f"output.json results[{i}]: vars.case_id missing; per-case grouping will be incomplete")

        provider_raw = row.get("provider")
        alias: str | None
        provider_id: str | None
        if isinstance(provider_raw, dict):
            provider_id = provider_raw.get("id")
            alias = provider_raw.get("label") or provider_id
        elif isinstance(provider_raw, str):
            provider_id = provider_raw
            alias = provider_raw
        else:
            provider_id = None
            alias = None
            warns.append(f"output.json results[{i}]: no usable 'provider' field")

        response = row.get("response") or {}
        output_text = response.get("output", row.get("output"))

        grading = row.get("gradingResult") or {}
        # token_usage means the MODEL's own consumption (response.tokenUsage).
        # Never fall back to gradingResult.tokensUsed here: that is the
        # llm-rubric judge's consumption, which report.py surfaces separately
        # -- mixing them double-counted judge tokens in the model column
        # whenever a provider omitted response.tokenUsage (issue #85).
        token_usage = response.get("tokenUsage") or {}
        cached = bool(response.get("cached", False))

        counter_key = (case_id, alias)
        repeat_index = repeat_counters.get(counter_key, 0) if case_id else 0
        if case_id:
            repeat_counters[counter_key] = repeat_index + 1

        passed = row.get("success")
        if passed is None:
            passed = grading.get("pass")

        results.append(
            CaseResult(
                case_id=case_id,
                alias=alias,
                provider_id=provider_id,
                expected=row_vars.get("expected"),
                category=row_vars.get("category"),
                output=output_text,
                passed=passed,
                score=row.get("score", grading.get("score")),
                reason=grading.get("reason"),
                cost=row.get("cost", response.get("cost")),
                latency_ms=row.get("latencyMs", response.get("latencyMs")),
                cached=cached,
                token_usage=token_usage,
                error=row.get("error") or response.get("error"),
                repeat_index=repeat_index,
                raw=row,
            )
        )

    eval_id = raw.get("evalId")
    prompts_meta = _nested_get(raw, "results", "prompts", default=[])
    if not isinstance(prompts_meta, list):
        prompts_meta = []

    for w in warns:
        _warnings.warn(w, stacklevel=2)

    return ParsedRun(eval_id=eval_id, results=results, prompts_meta=prompts_meta, warnings=warns)
