import json
import shutil

import pytest
import yaml

from evalloop import build as build_mod
from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import (
    BlogConfig,
    Config,
    GoldenCase,
    JudgeConfig,
    ModelConfig,
    OptimizeConfig,
    RunConfig,
    TaskConfig,
    load_golden_jsonl,
    load_task,
)


def _make_config(
    answer_type="label",
    models=None,
    judge_provider="anthropic:messages:claude-sonnet-4-6",
    rubric_file="tasks/sample-inquiry/prompts/judge_rubric.txt",
):
    return Config(
        task=TaskConfig(
            name="unit-test-task",
            answer_type=answer_type,
            prompt_file="tasks/sample-inquiry/prompts/task.txt",
            labels=["契約照会", "障害報告", "機能要望", "その他"] if answer_type == "label" else [],
            json_schema_file="schema.json" if answer_type == "json" else None,
        ),
        models=models
        or [
            ModelConfig(provider="ollama:chat:qwen2.5:7b", alias="qwen7b", tier="local", price_in_per_mtok=0, price_out_per_mtok=0),
            ModelConfig(provider="anthropic:messages:claude-haiku-4-5-20251001", alias="haiku45", tier="small", price_in_per_mtok=1, price_out_per_mtok=5),
        ],
        run=RunConfig(repeat=1, temperature=0.0, max_tokens=1024, cost_warn_usd=3.0),
        judge=JudgeConfig(provider=judge_provider, threshold=0.8, agreement_threshold=0.85, rubric_file=rubric_file),
        optimize=OptimizeConfig(target_alias="qwen7b", reflection_provider="anthropic/claude-opus-4-8"),
        blog=BlogConfig(),
        path=REPO_ROOT / "config.yaml",
    )


def _write_rubric(root):
    rubric_path = root / "judge_rubric.txt"
    rubric_path.write_text("問い合わせ: {{input}}\n期待: {{expected}}\nで採点してください。\n", encoding="utf-8")
    return rubric_path


# ---------------------------------------------------------------------------
# _build_default_test / iron rule #2 (judge must not grade itself unnoticed)
# ---------------------------------------------------------------------------


def test_default_test_label_uses_label_match_js(isolated_root):
    cfg = _make_config(answer_type="label")
    paths = TaskPaths(root=isolated_root, task="t1")
    default_test = build_mod._build_default_test(cfg, allow_same_judge=False, paths=paths)
    # labels must be a JSON-encoded STRING, not a real array: promptfoo expands
    # array-valued vars into a test matrix (one test per element), silently
    # multiplying every case by len(labels)
    assert isinstance(default_test["vars"]["labels"], str)
    assert json.loads(default_test["vars"]["labels"]) == cfg.task.labels
    assert default_test["assert"][0]["type"] == "javascript"
    assert "label_match.js" in default_test["assert"][0]["value"]


def test_default_test_json_uses_is_json_and_field_match(isolated_root):
    cfg = _make_config(answer_type="json")
    paths = TaskPaths(root=isolated_root, task="t1")
    default_test = build_mod._build_default_test(cfg, allow_same_judge=False, paths=paths)
    types = [a["type"] for a in default_test["assert"]]
    assert types == ["is-json", "javascript"]
    assert "json_field_match.js" in default_test["assert"][1]["value"]


def test_default_test_text_pins_judge_provider_and_threshold(isolated_root):
    rubric_path = _write_rubric(isolated_root)
    cfg = _make_config(
        answer_type="text",
        judge_provider="anthropic:messages:claude-sonnet-4-6",
        rubric_file=str(rubric_path),
    )
    paths = TaskPaths(root=isolated_root, task="t1")
    default_test = build_mod._build_default_test(cfg, allow_same_judge=False, paths=paths)
    rubric_assert = default_test["assert"][0]
    assert rubric_assert["type"] == "llm-rubric"
    assert rubric_assert["provider"] == "anthropic:messages:claude-sonnet-4-6"
    assert rubric_assert["threshold"] == 0.8
    # inline file content, NOT a file:// reference -- promptfoo 0.121.17 does not
    # substitute {{input}}/{{expected}} inside a file://-loaded llm-rubric value,
    # only inline string values get Nunjucks-templated (confirmed via a live run)
    assert not rubric_assert["value"].startswith("file://")
    assert "{{input}}" in rubric_assert["value"]
    assert "{{expected}}" in rubric_assert["value"]
    assert rubric_assert["value"] == rubric_path.read_text(encoding="utf-8")


def test_iron_rule_2_same_judge_raises_by_default(isolated_root):
    same_provider = "anthropic:messages:claude-sonnet-4-6"
    cfg = _make_config(
        answer_type="text",
        judge_provider=same_provider,
        models=[ModelConfig(provider=same_provider, alias="sonnet46", tier="mid", price_in_per_mtok=3, price_out_per_mtok=15)],
    )
    paths = TaskPaths(root=isolated_root, task="t1")
    try:
        build_mod._build_default_test(cfg, allow_same_judge=False, paths=paths)
        assert False, "expected BuildError"
    except build_mod.BuildError as e:
        assert "allow_same_judge" in str(e) or "--allow-same-judge" in str(e)


def test_iron_rule_2_same_judge_allowed_with_override(isolated_root):
    rubric_path = _write_rubric(isolated_root)
    same_provider = "anthropic:messages:claude-sonnet-4-6"
    cfg = _make_config(
        answer_type="text",
        judge_provider=same_provider,
        models=[ModelConfig(provider=same_provider, alias="sonnet46", tier="mid", price_in_per_mtok=3, price_out_per_mtok=15)],
        rubric_file=str(rubric_path),
    )
    paths = TaskPaths(root=isolated_root, task="t1")
    default_test = build_mod._build_default_test(cfg, allow_same_judge=True, paths=paths)
    assert default_test["assert"][0]["provider"] == same_provider


# ---------------------------------------------------------------------------
# supports_sampling_params: temperature omission for models that 400 on it
# ---------------------------------------------------------------------------


def test_provider_config_includes_temperature_by_default(isolated_root):
    cfg = _make_config()
    paths = TaskPaths(root=isolated_root, task="t1")
    promptfoo_config = build_mod._build_promptfoo_config(cfg, allow_same_judge=False, paths=paths)
    for provider in promptfoo_config["providers"]:
        assert provider["config"]["temperature"] == 0.0
        assert provider["config"]["max_tokens"] == 1024


def test_provider_config_omits_temperature_when_sampling_unsupported(isolated_root):
    # claude-opus-4-8 / claude-fable-5 はtemperature指定をHTTP 400で拒否する
    cfg = _make_config(
        models=[
            ModelConfig(provider="anthropic:messages:claude-haiku-4-5-20251001", alias="haiku45", tier="small", price_in_per_mtok=1, price_out_per_mtok=5),
            ModelConfig(provider="anthropic:messages:claude-opus-4-8", alias="opus48", tier="large", price_in_per_mtok=5, price_out_per_mtok=25, supports_sampling_params=False),
            ModelConfig(provider="anthropic:messages:claude-fable-5", alias="fable5", tier="frontier", price_in_per_mtok=10, price_out_per_mtok=50, supports_sampling_params=False),
        ]
    )
    paths = TaskPaths(root=isolated_root, task="t1")
    promptfoo_config = build_mod._build_promptfoo_config(cfg, allow_same_judge=False, paths=paths)
    by_alias = {p["label"]: p["config"] for p in promptfoo_config["providers"]}
    assert "temperature" in by_alias["haiku45"]
    assert "temperature" not in by_alias["opus48"]
    assert "temperature" not in by_alias["fable5"]
    # max_tokensは全モデルで受け付けるので常に出力される
    for provider_config in by_alias.values():
        assert provider_config["max_tokens"] == 1024


# ---------------------------------------------------------------------------
# iron rule #1 defense-in-depth check
# ---------------------------------------------------------------------------


def test_assert_config_never_references_train_raises():
    try:
        build_mod._assert_config_never_references_train("tests: file://../data/build/t1/tests_train.yaml")
        assert False, "expected BuildError"
    except build_mod.BuildError:
        pass


def test_assert_config_never_references_train_passes_for_test_file():
    build_mod._assert_config_never_references_train("tests: file://../data/build/t1/tests_test.yaml")


# ---------------------------------------------------------------------------
# cost estimate
# ---------------------------------------------------------------------------


def test_estimate_cost_zero_for_free_local_model():
    cfg = _make_config(models=[ModelConfig(provider="ollama:chat:qwen2.5:7b", alias="qwen7b", tier="local", price_in_per_mtok=0, price_out_per_mtok=0)])
    cases = [GoldenCase(id="case-0001", input="hello", expected="契約照会", split="test", category="基本", difficulty="easy", source="self-made")]
    estimate = build_mod.estimate_cost(cfg, cases, prompt_template="{{input}}")
    assert estimate.per_model_usd["qwen7b"] == 0.0
    assert estimate.total_usd == 0.0


def test_estimate_cost_scales_with_case_count_and_repeat():
    cfg = _make_config(
        models=[ModelConfig(provider="anthropic:messages:claude-haiku-4-5-20251001", alias="haiku45", tier="small", price_in_per_mtok=1, price_out_per_mtok=5)]
    )
    cfg.run.repeat = 2
    cases = [
        GoldenCase(id=f"case-{i:04d}", input="x" * 100, expected="契約照会", split="test", category="基本", difficulty="easy", source="self-made")
        for i in range(5)
    ]
    estimate = build_mod.estimate_cost(cfg, cases, prompt_template="{{input}}")
    assert estimate.per_model_usd["haiku45"] > 0
    # doubling repeat should double the estimate
    cfg.run.repeat = 4
    estimate2 = build_mod.estimate_cost(cfg, cases, prompt_template="{{input}}")
    assert estimate2.per_model_usd["haiku45"] == estimate.per_model_usd["haiku45"] * 2


# ---------------------------------------------------------------------------
# to_promptfoo_relpath
# ---------------------------------------------------------------------------


def test_to_promptfoo_relpath_uses_forward_slashes():
    rel = build_mod.to_promptfoo_relpath(
        REPO_ROOT / "tasks" / "sample-inquiry" / "prompts" / "task.txt",
        REPO_ROOT / "promptfoo" / "sample-inquiry",
    )
    assert rel == "../../tasks/sample-inquiry/prompts/task.txt"


# ---------------------------------------------------------------------------
# full build() pipeline against the real tracked sample-inquiry task.
#
# NOTE: tasks/sample-inquiry/ is the only task whose dataset is tracked in git
# (fresh clone / CI smoke). The workspace is copied into the isolated root so
# artifacts never land in the real checkout, and the assertions stay generic
# (derived from the copied golden.jsonl at test time) rather than hardcoding
# case counts/labels.
# ---------------------------------------------------------------------------


def test_build_missing_golden_raises(isolated_root):
    cfg = _make_config()
    paths = TaskPaths(root=isolated_root, task="t1")  # no golden.jsonl anywhere
    with pytest.raises(build_mod.BuildError, match="no dataset"):
        build_mod.build(cfg, paths, yes=True)


def test_build_end_to_end_against_real_sample(isolated_root):
    shutil.copy(REPO_ROOT / "config.yaml", isolated_root / "config.yaml")
    shutil.copytree(REPO_ROOT / "tasks" / "sample-inquiry", isolated_root / "tasks" / "sample-inquiry")
    cfg, paths = load_task("sample-inquiry", root=isolated_root)

    real_cases = load_golden_jsonl(paths.golden)
    expected_test_count = sum(1 for c in real_cases if c.split == "test")
    expected_train_count = sum(1 for c in real_cases if c.split == "train")

    estimate = build_mod.build(cfg, paths, yes=True)
    assert estimate.total_usd >= 0

    test_entries = yaml.safe_load(paths.tests_test.read_text(encoding="utf-8"))
    train_entries = yaml.safe_load(paths.tests_train.read_text(encoding="utf-8"))
    assert len(test_entries) == expected_test_count
    assert len(train_entries) == expected_train_count

    test_ids = {e["vars"]["case_id"] for e in test_entries}
    train_ids = {e["vars"]["case_id"] for e in train_entries}
    assert test_ids.isdisjoint(train_ids)

    promptfoo_config_text = paths.promptfoo_config.read_text(encoding="utf-8")
    assert "tests_train" not in promptfoo_config_text
    promptfoo_config = yaml.safe_load(promptfoo_config_text)
    assert promptfoo_config["tests"].endswith("tests_test.yaml")
    aliases = {p["label"] for p in promptfoo_config["providers"]}
    assert aliases == {m.alias for m in cfg.models}
