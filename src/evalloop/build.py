"""tasks/<name>/golden.jsonl + task.yaml -> data/build/<name>/tests_*.yaml +
promptfoo/<name>/promptfooconfig.yaml.

All task-scoped paths come from paths.TaskPaths (issue #47) -- this module
holds no per-task path constants.

Iron rules enforced here:
    1. split separation: tests_train.yaml can never end up referenced by the
       generated promptfooconfig.yaml, and train/test ids are asserted disjoint.
    2. an llm-rubric judge must never silently grade the same model it judges;
       this raises unless --allow-same-judge is passed.
    5. cost is estimated *before* running and the user is asked to confirm if
       the estimate exceeds run.cost_warn_usd (unless --yes is passed).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from evalloop.demos import DEMOS_PLACEHOLDER, DemoError, expand_demos_in_template
from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import Config, GoldenCase, assert_split_disjoint, load_golden_jsonl
from evalloop.token_counting import average_input_tokens, render_case_prompts

ASSERTS_DIR = Path(__file__).resolve().parent / "asserts"
LABEL_MATCH_JS = ASSERTS_DIR / "label_match.js"
JSON_FIELD_MATCH_JS = ASSERTS_DIR / "json_field_match.js"

ESTIMATED_OUTPUT_TOKENS = {"label": 12, "json": 120, "text": 200}


class BuildError(RuntimeError):
    pass


def to_promptfoo_relpath(target: Path, start: Path) -> str:
    """promptfoo resolves file:// paths relative to the config file's own directory."""
    rel = os.path.relpath(target, start=start)
    return rel.replace(os.sep, "/")


@dataclass
class CostEstimate:
    per_model_usd: dict[str, float]
    per_model_input_tokens: dict[str, int]
    token_count_methods: dict[str, str]
    total_usd: float


def estimate_cost(config: Config, test_cases: list[GoldenCase], prompt_template: str) -> CostEstimate:
    rendered_prompts = render_case_prompts(prompt_template, [c.input for c in test_cases])
    est_output_tokens = ESTIMATED_OUTPUT_TOKENS.get(config.task.answer_type, 100)

    per_model: dict[str, float] = {}
    per_model_input_tokens: dict[str, int] = {}
    token_count_methods: dict[str, str] = {}
    for model in config.models:
        token_count = average_input_tokens(model.provider, rendered_prompts)
        est_input_tokens = token_count.average_input_tokens
        cost_per_call = (
            est_input_tokens / 1_000_000 * model.price_in_per_mtok
            + est_output_tokens / 1_000_000 * model.price_out_per_mtok
        )
        per_model[model.alias] = cost_per_call * len(test_cases) * config.run.repeat
        per_model_input_tokens[model.alias] = est_input_tokens
        token_count_methods[model.alias] = token_count.method

    return CostEstimate(
        per_model_usd=per_model,
        per_model_input_tokens=per_model_input_tokens,
        token_count_methods=token_count_methods,
        total_usd=sum(per_model.values()),
    )


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


def _build_default_test(config: Config, allow_same_judge: bool, paths: TaskPaths) -> dict:
    answer_type = config.task.answer_type
    promptfoo_dir = paths.promptfoo_dir
    default_test: dict = {"vars": {}, "assert": []}

    if answer_type == "label":
        # NOTE: must be a JSON-encoded string, not a real YAML/JS array. promptfoo
        # treats any array-valued var (including ones merged in from defaultTest)
        # as a test matrix dimension and expands one test per element -- passing
        # the 9-label list directly turned 5 cases into 45 duplicated rows (one
        # per label) instead of 5. label_match.js JSON.parses this back to an array.
        default_test["vars"]["labels"] = json.dumps(config.task.labels, ensure_ascii=False)
        default_test["assert"] = [
            {"type": "javascript", "value": f"file://{to_promptfoo_relpath(LABEL_MATCH_JS, promptfoo_dir)}"}
        ]

    elif answer_type == "json":
        is_json_assert: dict = {"type": "is-json"}
        if config.task.json_schema_file:
            schema_path = REPO_ROOT / config.task.json_schema_file
            is_json_assert["value"] = f"file://{to_promptfoo_relpath(schema_path, promptfoo_dir)}"
        default_test["assert"] = [
            is_json_assert,
            {"type": "javascript", "value": f"file://{to_promptfoo_relpath(JSON_FIELD_MATCH_JS, promptfoo_dir)}"},
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


def _build_promptfoo_config(
    config: Config,
    allow_same_judge: bool,
    paths: TaskPaths,
    *,
    prompt_path: Path | None = None,
    tests_path: Path | None = None,
) -> dict:
    resolved_prompt = prompt_path if prompt_path is not None else (REPO_ROOT / config.task.prompt_file)
    tests_target = tests_path if tests_path is not None else paths.tests_test
    providers = []
    for m in config.models:
        # claude-opus-4-8 / claude-fable-5 はsamplingパラメータ(temperature等)を
        # HTTP 400で拒否する。supports_sampling_params: false のモデルには
        # temperatureを出力しない（max_tokensは全モデルで受け付ける）
        provider_config: dict = {}
        if m.supports_sampling_params:
            provider_config["temperature"] = config.run.temperature
        provider_config["max_tokens"] = config.run.max_tokens
        providers.append({"id": m.provider, "label": m.alias, "config": provider_config})

    return {
        "description": config.task.name,
        "providers": providers,
        "prompts": [f"file://{to_promptfoo_relpath(resolved_prompt, paths.promptfoo_dir)}"],
        "defaultTest": _build_default_test(config, allow_same_judge, paths),
        "tests": f"file://{to_promptfoo_relpath(tests_target, paths.promptfoo_dir)}",
    }


def _resolve_prompt_template(config: Config, paths: TaskPaths, test_cases: list[GoldenCase]) -> tuple[str, Path]:
    """Return (template_text, path_for_promptfoo) with optional ``{{demos}}`` expansion."""
    prompt_path = REPO_ROOT / config.task.prompt_file
    template = prompt_path.read_text(encoding="utf-8")
    demos_exist = paths.demos.exists()

    if demos_exist and DEMOS_PLACEHOLDER not in template:
        print(f"[build] WARN: {paths.demos} exists but prompt has no {DEMOS_PLACEHOLDER}; demos are ignored")
        return template, prompt_path

    try:
        resolved_text, n_demos = expand_demos_in_template(
            template,
            paths.demos,
            test_ids={c.id for c in test_cases},
            test_inputs={c.input for c in test_cases},
        )
    except DemoError as e:
        raise BuildError(str(e)) from e

    if n_demos is None:
        return template, prompt_path

    paths.resolved_prompt.parent.mkdir(parents=True, exist_ok=True)
    paths.resolved_prompt.write_text(resolved_text, encoding="utf-8")
    print(f"[build] wrote {paths.resolved_prompt} ({n_demos} demos embedded)")
    return resolved_text, paths.resolved_prompt


def _assert_config_never_references_train(promptfoo_config_text: str) -> None:
    """Iron rule #1, defense in depth: fail loudly if tests_train ever leaks in."""
    if "tests_train" in promptfoo_config_text:
        raise BuildError(
            "generated promptfooconfig.yaml references tests_train.yaml - this must "
            "never happen (train data must never reach promptfoo eval). This is a bug in build.py."
        )


def build(
    config: Config,
    paths: TaskPaths,
    allow_same_judge: bool = False,
    yes: bool = False,
    confirm_fn=None,
    shuffle_demos: int | None = None,
) -> CostEstimate:
    """Run the full build pipeline. Returns the cost estimate that was shown to the user.

    `confirm_fn` is injectable for tests; defaults to `typer.confirm` at the CLI layer.
    When ``shuffle_demos`` is a positive int, also write N demoshuffle variants (APO-19).
    """
    if not paths.golden.exists():
        raise BuildError(
            f"task {paths.task!r} has no dataset at {paths.golden}. Task data is not tracked in git "
            f"(issue #47 data policy) -- see {paths.task_dir / 'PROVENANCE.md'} for how to obtain it."
        )
    cases = load_golden_jsonl(paths.golden)

    if config.task.answer_type == "label":
        bad = sorted({c.id for c in cases if isinstance(c.expected, str) and c.expected not in config.task.labels})
        if bad:
            raise BuildError(f"golden.jsonl has case(s) with `expected` not in task.labels {config.task.labels}: {bad}")

    train_cases = [c for c in cases if c.split == "train"]
    dev_cases = [c for c in cases if c.split == "dev"]
    test_cases = [c for c in cases if c.split == "test"]
    if not test_cases:
        raise BuildError("golden.jsonl has no split=='test' cases; promptfoo eval would run 0 tests")

    train_ids = {c.id for c in train_cases}
    dev_ids = {c.id for c in dev_cases}
    test_ids = {c.id for c in test_cases}
    assert_split_disjoint(train_ids, test_ids)
    assert_split_disjoint(train_ids, dev_ids, label="train/dev")
    assert_split_disjoint(dev_ids, test_ids, label="dev/test")

    # Resolve demos / promptfoo config before writing build artifacts so a failed
    # demo leak check cannot leave fresh tests_*.yaml next to a stale config.
    # Demos must leak into neither test nor dev: dev is the shipping-gate holdout.
    holdout_cases = test_cases + dev_cases
    prompt_template, prompt_path_for_eval = _resolve_prompt_template(config, paths, holdout_cases)
    promptfoo_config = _build_promptfoo_config(config, allow_same_judge, paths, prompt_path=prompt_path_for_eval)
    config_text = yaml.safe_dump(promptfoo_config, allow_unicode=True, sort_keys=False)
    _assert_config_never_references_train(config_text)
    dev_config_text = None
    if dev_cases:
        dev_config = _build_promptfoo_config(
            config, allow_same_judge, paths, prompt_path=prompt_path_for_eval, tests_path=paths.tests_dev
        )
        dev_config_text = yaml.safe_dump(dev_config, allow_unicode=True, sort_keys=False)
        _assert_config_never_references_train(dev_config_text)

    _write_tests_yaml(paths.tests_test, test_cases)
    _write_tests_yaml(paths.tests_train, train_cases)
    paths.promptfoo_dir.mkdir(parents=True, exist_ok=True)
    paths.promptfoo_config.write_text(config_text, encoding="utf-8")
    if dev_cases:
        _write_tests_yaml(paths.tests_dev, dev_cases)
        paths.promptfoo_config_dev.write_text(dev_config_text, encoding="utf-8")
    else:
        # a stale dev config from a removed dev split must not stay runnable
        paths.tests_dev.unlink(missing_ok=True)
        paths.promptfoo_config_dev.unlink(missing_ok=True)
        # ... and neither must any optimized variant's dev config: it still
        # references the tests_dev.yaml just deleted above, so `evalloop run
        # --variant X --split dev` would otherwise hit a confusing promptfoo
        # error instead of resolve_config_path's clear "add dev cases" one.
        if paths.variants_dir.exists():
            for stale in paths.variants_dir.glob("*.dev.yaml"):
                stale.unlink()

    estimate = estimate_cost(config, test_cases, prompt_template)

    print(f"[build] {len(train_cases)} train / {len(dev_cases)} dev / {len(test_cases)} test cases from {paths.golden}")
    print(f"[build] wrote {paths.tests_test} and {paths.tests_train}")
    print(f"[build] wrote {paths.promptfoo_config}")
    if dev_cases:
        print(f"[build] wrote {paths.tests_dev} and {paths.promptfoo_config_dev}")
    print("[build] estimated pre-run cost (repeat=%d):" % config.run.repeat)
    for alias, usd in estimate.per_model_usd.items():
        print(
            f"[build]   {alias}: ${usd:.4f} "
            f"(~{estimate.per_model_input_tokens[alias]} input tokens/call; "
            f"method={estimate.token_count_methods[alias]})"
        )
    print(f"[build]   TOTAL: ${estimate.total_usd:.4f}  (pre-run estimate)")

    if estimate.total_usd > config.run.cost_warn_usd and not yes:
        confirm = confirm_fn or (lambda msg: input(f"{msg} [y/N] ").strip().lower() == "y")
        if not confirm(
            f"Estimated cost ${estimate.total_usd:.4f} exceeds cost_warn_usd "
            f"(${config.run.cost_warn_usd:.2f}). Continue?"
        ):
            raise BuildError("aborted by user: cost estimate exceeded cost_warn_usd")

    if shuffle_demos is not None:
        from evalloop import sensitivity as sensitivity_mod

        try:
            sensitivity_mod.build_demoshuffle_variants(config, paths, shuffle_demos)
        except sensitivity_mod.SensitivityError as e:
            raise BuildError(str(e)) from e

    return estimate
