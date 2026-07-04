"""golden.jsonl + config.yaml -> data/build/tests_*.yaml + promptfoo/promptfooconfig.yaml.

Fixed project-convention paths (not configurable — see README.md section 4):
    data/golden.jsonl                      single source of truth for cases
    data/build/tests_test.yaml             promptfoo tests, split=="test" only
    data/build/tests_train.yaml            GEPA-only, NEVER referenced by promptfoo
    promptfoo/promptfooconfig.yaml         generated eval config

Iron rules enforced here (README.md section 11):
    1. split separation: tests_train.yaml can never end up referenced by the
       generated promptfooconfig.yaml, and train/test ids are asserted disjoint.
    2. an llm-rubric judge must never silently grade the same model it judges;
       this raises unless --allow-same-judge is passed.
    5. cost is estimated *before* running and the user is asked to confirm if
       the estimate exceeds run.cost_warn_usd (unless --yes is passed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from evalloop.schemas import Config, GoldenCase, SchemaError, assert_split_disjoint, load_config, load_golden_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "data" / "golden.jsonl"
BUILD_DIR = REPO_ROOT / "data" / "build"
TESTS_TEST_PATH = BUILD_DIR / "tests_test.yaml"
TESTS_TRAIN_PATH = BUILD_DIR / "tests_train.yaml"
PROMPTFOO_DIR = REPO_ROOT / "promptfoo"
PROMPTFOO_CONFIG_PATH = PROMPTFOO_DIR / "promptfooconfig.yaml"
ASSERTS_DIR = REPO_ROOT / "src" / "evalloop" / "asserts"
LABEL_MATCH_JS = ASSERTS_DIR / "label_match.js"
JSON_FIELD_MATCH_JS = ASSERTS_DIR / "json_field_match.js"

# Rough chars-per-token heuristic for cost pre-estimation only (mixed
# Japanese/English business text). TODO: swap for a real tokenizer
# (e.g. anthropic's token counting API) if estimates prove too far off from
# the actual costs recorded in meta.json after real runs.
_CHARS_PER_TOKEN_ESTIMATE = 2.0
_ESTIMATED_OUTPUT_TOKENS = {"label": 12, "json": 120, "text": 200}


class BuildError(RuntimeError):
    pass


def to_promptfoo_relpath(target: Path) -> str:
    """promptfoo resolves file:// paths relative to the config file's own directory."""
    rel = os.path.relpath(target, start=PROMPTFOO_DIR)
    return rel.replace(os.sep, "/")


@dataclass
class CostEstimate:
    per_model_usd: dict[str, float]
    total_usd: float


def estimate_cost(config: Config, test_cases: list[GoldenCase], prompt_template: str) -> CostEstimate:
    avg_input_chars = len(prompt_template) + (
        sum(len(c.input) for c in test_cases) / len(test_cases) if test_cases else 0
    )
    est_input_tokens = max(1, int(avg_input_chars / _CHARS_PER_TOKEN_ESTIMATE))
    est_output_tokens = _ESTIMATED_OUTPUT_TOKENS.get(config.task.answer_type, 100)

    per_model: dict[str, float] = {}
    for model in config.models:
        cost_per_call = (
            est_input_tokens / 1_000_000 * model.price_in_per_mtok
            + est_output_tokens / 1_000_000 * model.price_out_per_mtok
        )
        per_model[model.alias] = cost_per_call * len(test_cases) * config.run.repeat

    return CostEstimate(per_model_usd=per_model, total_usd=sum(per_model.values()))


def _case_to_test_entry(case: GoldenCase) -> dict:
    return {
        "description": case.id,
        "vars": {
            "case_id": case.id,
            "input": case.input,
            "expected": case.expected,
            "category": case.category,
        },
    }


def _write_tests_yaml(path: Path, cases: list[GoldenCase]) -> None:
    entries = [_case_to_test_entry(c) for c in cases]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(entries, f, allow_unicode=True, sort_keys=False)


def _build_default_test(config: Config, allow_same_judge: bool) -> dict:
    answer_type = config.task.answer_type
    default_test: dict = {"vars": {}, "assert": []}

    if answer_type == "label":
        default_test["vars"]["labels"] = config.task.labels
        default_test["assert"] = [
            {"type": "javascript", "value": f"file://{to_promptfoo_relpath(LABEL_MATCH_JS)}"}
        ]

    elif answer_type == "json":
        is_json_assert: dict = {"type": "is-json"}
        if config.task.json_schema_file:
            schema_path = REPO_ROOT / config.task.json_schema_file
            is_json_assert["value"] = f"file://{to_promptfoo_relpath(schema_path)}"
        default_test["assert"] = [
            is_json_assert,
            {"type": "javascript", "value": f"file://{to_promptfoo_relpath(JSON_FIELD_MATCH_JS)}"},
        ]

    elif answer_type == "text":
        judge_provider = config.judge.provider
        same_judge_models = [m for m in config.models if m.provider == judge_provider]
        if same_judge_models and not allow_same_judge:
            aliases = ", ".join(m.alias for m in same_judge_models)
            raise BuildError(
                "iron rule #2 violation: judge.provider "
                f"({judge_provider!r}) is identical to evaluated model(s) [{aliases}]. "
                "Re-run with --allow-same-judge to override this on purpose."
            )
        rubric_path = REPO_ROOT / config.judge.rubric_file
        # Empirically (promptfoo 0.121.17): a `file://` value on llm-rubric is
        # NOT run through Nunjucks templating -- {{input}}/{{expected}} come
        # through to the grading prompt as literal, unsubstituted text (see
        # `renderedAssertionValue` in a real output.json). Inline string
        # values *do* get templated (matches the promptfoo docs' own inline
        # example), so read the rubric file's content here and embed it
        # directly instead of referencing it by file://.
        default_test["assert"] = [
            {
                "type": "llm-rubric",
                "value": rubric_path.read_text(encoding="utf-8"),
                "provider": judge_provider,
                "threshold": config.judge.threshold,
            }
        ]

    else:  # pragma: no cover - guarded by TaskConfig.__post_init__
        raise BuildError(f"unknown answer_type {answer_type!r}")

    return default_test


def _build_promptfoo_config(config: Config, allow_same_judge: bool) -> dict:
    prompt_path = REPO_ROOT / config.task.prompt_file
    providers = [
        {
            "id": m.provider,
            "label": m.alias,
            "config": {
                "temperature": config.run.temperature,
                "max_tokens": config.run.max_tokens,
            },
        }
        for m in config.models
    ]

    return {
        "description": config.task.name,
        "providers": providers,
        "prompts": [f"file://{to_promptfoo_relpath(prompt_path)}"],
        "defaultTest": _build_default_test(config, allow_same_judge),
        "tests": f"file://{to_promptfoo_relpath(TESTS_TEST_PATH)}",
    }


def _assert_config_never_references_train(promptfoo_config_text: str) -> None:
    """Iron rule #1, defense in depth: fail loudly if tests_train ever leaks in."""
    if "tests_train" in promptfoo_config_text:
        raise BuildError(
            "generated promptfooconfig.yaml references tests_train.yaml - this must "
            "never happen (train data must never reach promptfoo eval). This is a bug in build.py."
        )


def build(
    config_path: str | Path = REPO_ROOT / "config.yaml",
    allow_same_judge: bool = False,
    yes: bool = False,
    confirm_fn=None,
) -> CostEstimate:
    """Run the full build pipeline. Returns the cost estimate that was shown to the user.

    `confirm_fn` is injectable for tests; defaults to `typer.confirm` at the CLI layer.
    """
    config = load_config(config_path)
    cases = load_golden_jsonl(GOLDEN_PATH)

    if config.task.answer_type == "label":
        bad = sorted(
            {c.id for c in cases if isinstance(c.expected, str) and c.expected not in config.task.labels}
        )
        if bad:
            raise BuildError(
                f"golden.jsonl has case(s) with `expected` not in task.labels {config.task.labels}: {bad}"
            )

    train_cases = [c for c in cases if c.split == "train"]
    test_cases = [c for c in cases if c.split == "test"]
    if not test_cases:
        raise BuildError("golden.jsonl has no split=='test' cases; promptfoo eval would run 0 tests")

    train_ids = {c.id for c in train_cases}
    test_ids = {c.id for c in test_cases}
    assert_split_disjoint(train_ids, test_ids)

    _write_tests_yaml(TESTS_TEST_PATH, test_cases)
    _write_tests_yaml(TESTS_TRAIN_PATH, train_cases)

    promptfoo_config = _build_promptfoo_config(config, allow_same_judge)
    PROMPTFOO_DIR.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(promptfoo_config, allow_unicode=True, sort_keys=False)
    _assert_config_never_references_train(config_text)
    PROMPTFOO_CONFIG_PATH.write_text(config_text, encoding="utf-8")

    prompt_template = (REPO_ROOT / config.task.prompt_file).read_text(encoding="utf-8")
    estimate = estimate_cost(config, test_cases, prompt_template)

    print(f"[build] {len(train_cases)} train / {len(test_cases)} test cases from {GOLDEN_PATH}")
    print(f"[build] wrote {TESTS_TEST_PATH} and {TESTS_TRAIN_PATH}")
    print(f"[build] wrote {PROMPTFOO_CONFIG_PATH}")
    print("[build] estimated pre-run cost (repeat=%d):" % config.run.repeat)
    for alias, usd in estimate.per_model_usd.items():
        print(f"[build]   {alias}: ${usd:.4f}")
    print(f"[build]   TOTAL: ${estimate.total_usd:.4f}  (rough estimate, not a real tokenizer - see build.py)")

    if estimate.total_usd > config.run.cost_warn_usd and not yes:
        confirm = confirm_fn or (lambda msg: input(f"{msg} [y/N] ").strip().lower() == "y")
        if not confirm(
            f"Estimated cost ${estimate.total_usd:.4f} exceeds cost_warn_usd "
            f"(${config.run.cost_warn_usd:.2f}). Continue?"
        ):
            raise BuildError("aborted by user: cost estimate exceeded cost_warn_usd")

    return estimate
