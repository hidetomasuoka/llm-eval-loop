import json
import types

import pytest
import yaml

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import run as run_mod
from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import GoldenCase, load_task
from tests.conftest import scaffold_task

# NOTE: tests that exercise the real build/run/report orchestration scaffold a
# task inside `isolated_root` (tests/conftest.py) so nothing is written into
# the real checkout.


# ---------------------------------------------------------------------------
# promptfoo provider -> dspy LM string
# ---------------------------------------------------------------------------


def test_provider_mapping_anthropic():
    assert optimize_mod.promptfoo_provider_to_dspy_lm("anthropic:messages:claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"


def test_provider_mapping_ollama():
    assert optimize_mod.promptfoo_provider_to_dspy_lm("ollama:chat:qwen2.5:7b") == "ollama_chat/qwen2.5:7b"


def test_provider_mapping_unknown_raises():
    with pytest.raises(optimize_mod.OptimizeError):
        optimize_mod.promptfoo_provider_to_dspy_lm("openai:gpt-5")


# ---------------------------------------------------------------------------
# APO-14: post-hoc search cost from dspy lm.history
# ---------------------------------------------------------------------------


def test_summarize_lm_search_cost_from_litellm_cost_and_usage(isolated_root):
    cfg, _paths = scaffold_task(isolated_root, models=["haiku45"])
    # point optimize at a priced registry model
    raw = yaml.safe_load(_paths.task_config.read_text(encoding="utf-8"))
    raw["optimize"]["target_alias"] = "haiku45"
    raw["optimize"]["reflection_provider"] = "anthropic/claude-haiku-4-5-20251001"
    _paths.task_config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    cfg, _paths = load_task("t1", root=isolated_root)

    task_lm = types.SimpleNamespace(
        history=[
            {"cost": 0.01, "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            {
                "cost": None,
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
            },  # $1.0 in @ $1/MTok
        ]
    )
    reflection_lm = types.SimpleNamespace(
        history=[{"cost": 0.002, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}]
    )
    summary = optimize_mod.summarize_lm_search_cost(task_lm, reflection_lm, cfg)
    assert summary.search_lm_call_count == 3
    assert summary.search_cost_usd == pytest.approx(1.012)


def test_summarize_lm_search_cost_empty_history_is_none(isolated_root):
    cfg, _paths = scaffold_task(isolated_root)
    summary = optimize_mod.summarize_lm_search_cost(
        types.SimpleNamespace(history=[]),
        types.SimpleNamespace(history=None),
        cfg,
    )
    assert summary.search_cost_usd is None
    assert summary.search_lm_call_count == 0


def test_summarize_lm_search_cost_unpriced_call_yields_none(isolated_root):
    cfg, _paths = scaffold_task(isolated_root)
    task_lm = types.SimpleNamespace(history=[{"cost": None, "usage": {}}])
    reflection_lm = types.SimpleNamespace(history=[])
    summary = optimize_mod.summarize_lm_search_cost(task_lm, reflection_lm, cfg)
    assert summary.search_cost_usd is None
    assert summary.search_lm_call_count == 1


# ---------------------------------------------------------------------------
# sampling params on the dspy path (opus48/fable5 reject temperature with 400)
# ---------------------------------------------------------------------------


def test_dspy_temperature_none_when_sampling_unsupported():
    # litellm drops None-valued params from the request, so None = "don't send"
    assert optimize_mod._dspy_temperature(True, 0.0) == 0.0
    assert optimize_mod._dspy_temperature(False, 0.0) is None


def test_reflection_supports_sampling_matches_registry_by_dspy_string(isolated_root):
    opus48 = {
        "provider": "anthropic:messages:claude-opus-4-8",
        "alias": "opus48",
        "tier": "large",
        "supports_sampling_params": False,
    }
    cfg, _paths = scaffold_task(
        isolated_root,
        global_models=[
            {"provider": "ollama:chat:qwen2.5:7b", "alias": "qwen7b", "tier": "local"},
            opus48,
        ],
        reflection_provider="anthropic/claude-opus-4-8",
    )
    # the bundled configs point reflection at opus48, which rejects temperature
    assert optimize_mod._reflection_supports_sampling(cfg) is False


def test_reflection_supports_sampling_defaults_true_for_unknown_provider(isolated_root):
    cfg, _paths = scaffold_task(
        isolated_root,
        name="t2",
        global_models=[{"provider": "ollama:chat:qwen2.5:7b", "alias": "qwen7b", "tier": "local"}],
        reflection_provider="openai/gpt-5",  # no registry match -> keep sending temperature
    )
    assert optimize_mod._reflection_supports_sampling(cfg) is True


# ---------------------------------------------------------------------------
# template <-> instructions round-trip
# ---------------------------------------------------------------------------


def test_extract_and_render_round_trip_preserves_trailer():
    template = "これは指示です。\n複数行あります。\n\n問い合わせ文:\n{{input}}\n"
    instructions = optimize_mod.extract_instructions_from_template(template)
    assert "{{input}}" not in instructions
    assert "これは指示です。" in instructions

    rendered = optimize_mod.render_optimized_template("新しい指示", template)
    assert "新しい指示" in rendered
    assert "{{input}}" in rendered
    assert "問い合わせ文:" in rendered


def test_extract_handles_template_without_input_marker():
    template = "no placeholder here"
    instructions = optimize_mod.extract_instructions_from_template(template)
    assert instructions == "no placeholder here"
    rendered = optimize_mod.render_optimized_template("new", template)
    assert "{{input}}" in rendered


def test_round_trip_against_real_sample_prompt():
    real_template = (REPO_ROOT / "tasks" / "sample-inquiry" / "prompts" / "task.txt").read_text(encoding="utf-8")
    instructions = optimize_mod.extract_instructions_from_template(real_template)
    assert "{{input}}" not in instructions
    assert len(instructions) > 0
    rendered = optimize_mod.render_optimized_template(instructions, real_template)
    assert "{{input}}" in rendered


# ---------------------------------------------------------------------------
# label metric (Python port of label_match.js)
# ---------------------------------------------------------------------------


LABELS = ["契約照会", "障害報告", "機能要望", "その他"]


def test_label_metric_exact_match():
    score, feedback = optimize_mod.label_score_and_feedback("契約照会", "契約照会", LABELS)
    assert score == 1.0


def test_label_metric_normalizes_punctuation_and_width():
    score, _ = optimize_mod.label_score_and_feedback("契約照会。", "契約照会", LABELS)
    assert score == 1.0


def test_label_metric_single_contained_label_matches():
    score, _ = optimize_mod.label_score_and_feedback("回答: 障害報告 です", "障害報告", LABELS)
    assert score == 1.0


def test_label_metric_wrong_label_scores_zero_with_feedback():
    score, feedback = optimize_mod.label_score_and_feedback("その他", "契約照会", LABELS)
    assert score == 0.0
    assert "契約照会" in feedback


def test_label_metric_no_known_label_scores_zero():
    score, feedback = optimize_mod.label_score_and_feedback("わかりません", "契約照会", LABELS)
    assert score == 0.0
    assert "契約照会" in feedback


# ---------------------------------------------------------------------------
# text metric (SQuAD-style token F1; the final eval stays llm-rubric)
# ---------------------------------------------------------------------------


GOLD_SPAN = "This Agreement shall be governed by the laws of the State of New York."


def test_text_metric_verbatim_extraction_scores_one():
    score, _ = optimize_mod.text_score_and_feedback(GOLD_SPAN, GOLD_SPAN)
    assert score == 1.0


def test_text_metric_ignores_case_punctuation_and_articles():
    output = "this agreement shall be governed by laws of state of new york"
    score, _ = optimize_mod.text_score_and_feedback(output, GOLD_SPAN)
    assert score == 1.0


def test_text_metric_partial_overlap_scores_between_zero_and_one():
    score, feedback = optimize_mod.text_score_and_feedback("governed by the laws of the State of New York", GOLD_SPAN)
    assert 0.0 < score < 1.0
    assert "recall-weighted" in feedback


def test_text_metric_disjoint_scores_zero():
    score, _ = optimize_mod.text_score_and_feedback("completely unrelated text here", GOLD_SPAN)
    assert score == 0.0


def test_text_metric_empty_output_scores_zero():
    score, _ = optimize_mod.text_score_and_feedback("", GOLD_SPAN)
    assert score == 0.0


def test_text_metric_no_clause_both_sides_scores_one():
    # wrapping quotes / trailing punctuation must not break the sentinel match
    score, _ = optimize_mod.text_score_and_feedback("「該当条項なし。」", "該当条項なし")
    assert score == 1.0


def test_text_metric_no_clause_output_but_gold_has_clause():
    score, feedback = optimize_mod.text_score_and_feedback("該当条項なし", GOLD_SPAN)
    assert score == 0.0
    assert "該当条項なし" in feedback


def test_text_metric_extraction_when_gold_says_no_clause():
    score, feedback = optimize_mod.text_score_and_feedback(GOLD_SPAN, "該当条項なし")
    assert score == 0.0
    assert "該当条項なし" in feedback


def test_text_metric_multi_span_order_insensitive():
    expected = "Span one about indemnity; Span two about termination"
    output = "Span two about termination; Span one about indemnity"
    score, _ = optimize_mod.text_score_and_feedback(output, expected)
    assert score == 1.0


def test_text_metric_spurious_extra_span_is_penalized():
    output = f"{GOLD_SPAN}; Some unrelated extra clause the model invented"
    score, _ = optimize_mod.text_score_and_feedback(output, GOLD_SPAN)
    assert 0.0 < score < 1.0


def test_text_metric_missing_span_is_penalized():
    expected = f"{GOLD_SPAN}; A second clause about termination fees"
    score, _ = optimize_mod.text_score_and_feedback(GOLD_SPAN, expected)
    assert 0.0 < score < 1.0


def test_text_metric_under_extraction_to_single_word_scores_low():
    # the first GEPA run (20260706-075752) degraded to outputting a single
    # heading word like "Effective Date" -- recall-weighted scoring should
    # punish this much harder than over-extraction (rubric tolerates extra
    # text but fails on missing core content).
    score, _ = optimize_mod.text_score_and_feedback("Agreement", GOLD_SPAN)
    assert 0.0 < score < 0.5


def test_text_metric_over_extraction_scores_higher_than_under_extraction():
    # over-extraction (gold + extra) should beat under-extraction (one word)
    # because the rubric tolerates mild over-extraction but fails truncation.
    under, _ = optimize_mod.text_score_and_feedback("Agreement", GOLD_SPAN)
    over, _ = optimize_mod.text_score_and_feedback(
        f"{GOLD_SPAN} and some surrounding context", GOLD_SPAN
    )
    assert over > under


def test_text_metric_full_extraction_with_extra_keeps_high_score():
    # gold fully covered + mild extra context -- recall is 1.0 so the score
    # should stay close to 1.0 (the rubric tolerates this; the old F1 metric
    # would have pulled it down via precision).
    score, _ = optimize_mod.text_score_and_feedback(
        f"{GOLD_SPAN} Additional surrounding sentences.", GOLD_SPAN
    )
    assert score >= 0.8


# ---------------------------------------------------------------------------
# variant config re-rooting
# ---------------------------------------------------------------------------


def test_reroot_file_refs_adds_prefix_only_to_file_uris():
    obj = {"a": "file://../x.txt", "b": ["file://../y.js", "not-a-file-ref"], "c": 3}
    rerooted = optimize_mod._reroot_file_refs(obj, prefix="../")
    assert rerooted["a"] == "file://../../x.txt"
    assert rerooted["b"][0] == "file://../../y.js"
    assert rerooted["b"][1] == "not-a-file-ref"
    assert rerooted["c"] == 3


def test_build_variant_config_reroots_and_swaps_prompt(isolated_root):
    # promptfoo/<task>/promptfooconfig.yaml paths are relative to the task's
    # promptfoo dir; the variant lives one level deeper at
    # promptfoo/<task>/variants/, so every file:// ref must gain one extra
    # "../". Pin down a known (label-type) build first -- label-type still has
    # a file:// javascript assert to check rerooting on (text-type's
    # llm-rubric assert is inline content, not file://, since promptfoo
    # doesn't template {{input}}/{{expected}} in file://-loaded rubric values
    # -- see build.py's comment).
    cfg, paths = scaffold_task(isolated_root)
    build_mod.build(cfg, paths, yes=True)

    fake_task_path = paths.optimized_dir / "qwen7b" / "20260101-000000" / "task.txt"
    variant_config = optimize_mod.build_variant_config("qwen7b", fake_task_path, paths)

    assert variant_config["prompts"] == [
        f"file://{optimize_mod.to_variant_relpath(fake_task_path, paths.variants_dir)}"
    ]
    assert variant_config["defaultTest"]["assert"][0]["value"].startswith("file://../../")
    assert variant_config["tests"].startswith("file://../../")
    assert "optimized" in variant_config["description"]


def test_build_variant_config_missing_base_config_raises(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")  # nothing built here
    with pytest.raises(optimize_mod.OptimizeError):
        optimize_mod.build_variant_config("qwen7b", isolated_root / "task.txt", paths)


# ---------------------------------------------------------------------------
# _find_latest_base_run
# ---------------------------------------------------------------------------


def test_find_latest_base_run_picks_most_recent_successful_base(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    paths.results_dir.mkdir(parents=True)
    entries = [
        {"run_id": "old", "created_at": "2026-01-01T00:00:00Z", "task_name": "t", "variant": None, "promptfoo_exit_code": 0},
        {"run_id": "variant-run", "created_at": "2026-01-02T00:00:00Z", "task_name": "t", "variant": "x", "promptfoo_exit_code": 0},
        {"run_id": "failed", "created_at": "2026-01-03T00:00:00Z", "task_name": "t", "variant": None, "promptfoo_exit_code": 1},
        {"run_id": "newest-base", "created_at": "2026-01-04T00:00:00Z", "task_name": "t", "variant": None, "promptfoo_exit_code": 0},
    ]
    paths.index.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    assert optimize_mod._find_latest_base_run("t", paths) == "newest-base"


def test_find_latest_base_run_returns_none_when_missing(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")  # no index.jsonl
    assert optimize_mod._find_latest_base_run("t", paths) is None


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _write_output(runs_dir, run_id, rows):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "output.json").write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")


def _row(case_id, alias, passed, cost=0.001):
    return {
        "vars": {"case_id": case_id, "expected": "契約照会"},
        "provider": {"id": "p", "label": alias},
        "response": {"output": "契約照会"},
        "gradingResult": {"pass": passed, "score": 1 if passed else 0},
        "success": passed,
        "cost": cost,
    }


def test_compare_computes_deltas(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")

    _write_output(paths.runs_dir, "before", [_row("case-0001", "qwen7b", False, cost=0.0), _row("case-0002", "qwen7b", False, cost=0.0)])
    _write_output(paths.runs_dir, "after", [_row("case-0001", "qwen7b", True, cost=0.0), _row("case-0002", "qwen7b", True, cost=0.0)])

    path = optimize_mod.compare(["before", "after"], paths)
    content = path.read_text(encoding="utf-8")

    assert path.parent == paths.reports_dir
    assert path.name == "compare_before_after.md"
    assert "qwen7b" in content
    assert "+100.0%" in content
    # 2-run path must stay the legacy delta form (no multi-run disclaimer).
    assert "条件依存" not in content


def test_compare_missing_run_raises(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    with pytest.raises(optimize_mod.OptimizeError):
        optimize_mod.compare(["nope-a", "nope-b"], paths)


# ---------------------------------------------------------------------------
# optimize() end-to-end orchestration, with the real GEPA call and the real
# promptfoo subprocess both stubbed out (neither is feasible without live
# API keys / a compatible npx promptfoo install)
# ---------------------------------------------------------------------------


def _label_type_golden_rows():
    # 12 train cases = 3 per label: enough to clear the APO-09 preflight
    # errors (train>=10, every label >=2) so the e2e path runs without --force
    labels = ["契約照会", "障害報告", "機能要望", "その他"]
    rows = []
    for i in range(12):
        rows.append(
            {
                "id": f"case-{i + 1:04d}",
                "input": f"問い合わせ文サンプル{i + 1}",
                "expected": labels[i % len(labels)],
                "split": "train",
                "meta": {"category": "基本", "source": "self-made"},
            }
        )
    for i in range(12):
        rows.append(
            {
                "id": f"case-{i + 100:04d}",
                "input": f"問い合わせ文サンプル{i + 100}",
                "expected": labels[i % len(labels)],
                "split": "test",
                "meta": {"category": "基本", "source": "self-made"},
            }
        )
    return rows


def _stub_run_env(monkeypatch, fake_eval):
    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)
    # keep run()'s post-eval bookkeeping hermetic (no real npx/node subprocesses)
    monkeypatch.setattr(run_mod, "get_promptfoo_version", lambda: "0.0.0-test")
    monkeypatch.setattr(run_mod, "get_node_version", lambda: "v22.22.0")


def test_optimize_end_to_end_with_stubbed_gepa_and_promptfoo(isolated_root, monkeypatch):
    # optimize() trains against the task's own golden.jsonl; scaffold a
    # label-type task whose data matches its own labels.
    cfg, paths = scaffold_task(isolated_root, golden_rows=_label_type_golden_rows())
    build_mod.build(cfg, paths, yes=True)

    fake_optimized = types.SimpleNamespace(
        signature=types.SimpleNamespace(instructions="最適化された新しい指示文です。")
    )
    monkeypatch.setattr(optimize_mod, "run_gepa", lambda *a, **k: fake_optimized)

    def fake_eval(config_path, output_path, **kwargs):
        cfg_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rows = [
            {
                "vars": {"case_id": f"case-{i:04d}", "expected": "契約照会", "category": "基本"},
                "provider": {"id": p["id"], "label": p["label"]},
                "response": {"output": "契約照会"},
                "gradingResult": {"pass": True, "score": 1},
                "success": True,
                "cost": 0.0,
            }
            for i, p in enumerate(cfg_yaml["providers"], start=1)
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    _stub_run_env(monkeypatch, fake_eval)

    # 12 train cases clear preflight, so the default (non-force) path runs
    outcome = optimize_mod.optimize(cfg, paths)

    assert outcome.task_path.exists()
    assert paths.optimized_dir in outcome.task_path.parents
    assert "最適化された新しい指示文です。" in outcome.task_path.read_text(encoding="utf-8")
    assert outcome.variant_path.exists()
    assert outcome.run_id
    assert (paths.runs_dir / outcome.run_id / "output.json").exists()

    # APO-05: the method names the variant and the output directory so
    # cross-method comparisons can tell runs apart; slug encodes auto/n{train}
    assert "_gepa_" in outcome.variant_name
    assert outcome.task_path.parent.name.startswith("gepa-")
    assert outcome.task_path.parent.name.endswith("-light-n12")
    assert outcome.variant_name.endswith("_light-n12")

    # APO-05: pin the optimize_log.json schema (plus slug/summary)
    log = json.loads((outcome.task_path.parent / "optimize_log.json").read_text(encoding="utf-8"))
    assert log["method"] == "gepa"
    assert log["params"]["auto"] == "light"  # effective params include the resolved auto
    assert log["slug"] == "light-n12"
    assert "gepa auto=light train=12" in log["summary"]
    assert "instructions" in log["summary"]
    assert isinstance(log["duration_seconds"], float)
    assert "search_cost_usd" in log  # APO-14; stub LMs leave this null
    assert log["search_cost_usd"] is None
    assert log["search_lm_call_count"] == 0
    assert log["train_case_count"] == len(log["train_case_ids"])

    # optimized/index.jsonl records the variant with run linkage
    assert paths.optimized_index.exists()
    index_lines = [
        json.loads(line) for line in paths.optimized_index.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(index_lines) == 1
    entry = index_lines[0]
    assert entry["variant_name"] == outcome.variant_name
    assert entry["slug"] == "light-n12"
    assert entry["method"] == "gepa"
    assert entry["run_id"] == outcome.run_id
    assert entry["base_run_id"] is None
    assert entry["optimize_log"].endswith("/optimize_log.json")

    # the isolated index.jsonl only holds this variant run (base runs are
    # variant=None), so compare is skipped -- deterministically, unlike when
    # this read the developer's real results/index.jsonl
    assert outcome.base_run_id is None
    assert outcome.compare_path is None


# ---------------------------------------------------------------------------
# variant slug / summary helpers
# ---------------------------------------------------------------------------


def test_make_variant_slug_basic():
    assert optimize_mod._make_variant_slug(auto="light", params={"auto": "light"}, train_case_count=20) == "light-n20"


def test_make_variant_slug_includes_scalar_params():
    slug = optimize_mod._make_variant_slug(
        auto="medium",
        params={"auto": "medium", "val_ratio": 0.2, "seed": 42},
        train_case_count=40,
    )
    assert slug == "medium-seed42-val0.2-n40"


def test_make_variant_slug_skips_nested_and_long_strings():
    slug = optimize_mod._make_variant_slug(
        auto="light",
        params={
            "auto": "light",
            "nested": {"a": 1},
            "long": "x" * 40,
            "breadth": 5,
        },
        train_case_count=10,
    )
    assert slug == "light-br5-n10"
    assert "nested" not in slug
    assert "xxxx" not in slug


def test_make_variant_slug_collision_appends_hash():
    slug = optimize_mod._make_variant_slug(
        auto="light",
        params={"auto": "light"},
        train_case_count=4,
        base_instructions="base",
        optimized_instructions="opt",
        occupied={"light-n4"},
    )
    assert slug.endswith("-n4")
    assert slug != "light-n4"
    # form: light-{4hex}-n4
    parts = slug.split("-")
    assert parts[0] == "light" and parts[-1] == "n4" and len(parts[-2]) == 4


def test_make_variant_slug_truncation_preserves_train_token():
    # many long param tokens would exceed _SLUG_MAX_LEN if truncated from the right
    params = {f"param{i}": f"value{i}xx" for i in range(8)}
    params["auto"] = "light"
    slug = optimize_mod._make_variant_slug(auto="light", params=params, train_case_count=99)
    assert slug.endswith("-n99")
    assert len(slug) <= optimize_mod._SLUG_MAX_LEN


def test_make_variant_summary():
    summary = optimize_mod._make_variant_summary(
        method="gepa",
        auto="light",
        params={"auto": "light"},
        train_case_count=20,
        base_instructions="abcd",
        optimized_instructions="abcdefgh",
    )
    assert summary == "gepa auto=light train=20; instructions 4→8 chars"


def test_make_variant_summary_normalizes_string_newlines():
    summary = optimize_mod._make_variant_summary(
        method="gepa",
        auto="light",
        params={"auto": "light", "note": "a\nb\tc"},
        train_case_count=3,
        base_instructions="x",
        optimized_instructions="yy",
    )
    assert "\n" not in summary
    assert "note=a b c" in summary
    assert summary.count("\n") == 0


def test_slug_from_dir_name():
    assert optimize_mod._slug_from_dir_name("gepa-20260708-094533-light-n20") == "light-n20"
    assert optimize_mod._slug_from_dir_name("gepa-20260708-094533") is None
    assert optimize_mod._slug_from_dir_name("20260706-075752") is None


def test_occupied_slugs(tmp_path):
    alias = tmp_path / "glm52"
    (alias / "gepa-20260708-094533-light-n20").mkdir(parents=True)
    (alias / "gepa-20260708-100000").mkdir()  # no slug
    (alias / "20260706-075752").mkdir()  # legacy
    assert optimize_mod._occupied_slugs(alias) == {"light-n20"}


def _text_type_golden_rows():
    spans = [
        "This Agreement shall be governed by the laws of the State of New York.",
        "Either party may terminate upon thirty days written notice.",
        "該当条項なし",
    ]
    rows = []
    # 12 train cases clear the APO-09 preflight minimum (text tasks have no
    # per-label check, only the size checks)
    for i in range(12):
        rows.append(
            {
                "id": f"case-{i + 1:04d}",
                "input": f"CONTRACT EXCERPT {i + 1} ...",
                "expected": spans[i % len(spans)],
                "split": "train",
                "meta": {"category": "governing-law", "source": "self-made"},
            }
        )
    for i in range(6):
        rows.append(
            {
                "id": f"case-{i + 100:04d}",
                "input": f"CONTRACT EXCERPT {i + 100} ...",
                "expected": spans[i % len(spans)],
                "split": "test",
                "meta": {"category": "governing-law", "source": "self-made"},
            }
        )
    return rows


def test_optimize_end_to_end_with_text_task(isolated_root, monkeypatch):
    """answer_type=text no longer trips a guard (issue #17): a CUAD-style
    text task must reach GEPA training and the downstream run/report.
    """
    # the scaffold's judge provider is distinct from every evaluated model, so
    # no --allow-same-judge override is needed for the llm-rubric build
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="text",
        golden_rows=_text_type_golden_rows(),
        rubric="採点: {{input}} / {{expected}}\n",
    )
    build_mod.build(cfg, paths, yes=True)

    captured = {}

    def fake_gepa(student, trainset, metric, reflection_lm, auto, seed=0):
        # exercise the real metric wiring with one plausible rollout per case
        captured["scores"] = [
            metric(gold, types.SimpleNamespace(output=gold.expected)).score for gold in trainset
        ]
        return types.SimpleNamespace(signature=types.SimpleNamespace(instructions="optimized text instructions"))

    monkeypatch.setattr(optimize_mod, "run_gepa", fake_gepa)

    def fake_eval(config_path, output_path, **kwargs):
        cfg_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rows = [
            {
                "vars": {"case_id": f"case-{i:04d}", "expected": "x", "category": "governing-law"},
                "provider": {"id": p["id"], "label": p["label"]},
                "response": {"output": "x"},
                "gradingResult": {"pass": True, "score": 1},
                "success": True,
                "cost": 0.0,
            }
            for i, p in enumerate(cfg_yaml["providers"], start=1)
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    _stub_run_env(monkeypatch, fake_eval)

    # 12 train cases clear preflight, so the default (non-force) path runs
    outcome = optimize_mod.optimize(cfg, paths)

    assert outcome.task_path.exists()
    assert "optimized text instructions" in outcome.task_path.read_text(encoding="utf-8")
    # a rollout that echoes the gold answer must score 1.0 through the real
    # metric wiring (incl. the 該当条項なし sentinel case)
    assert captured["scores"] == [1.0] * 12


# ---------------------------------------------------------------------------
# pre-run cost estimate (APO-10): price-table math, per-method factors, and
# the --yes confirmation contract
# ---------------------------------------------------------------------------


# scaffold registry variant with real prices so the estimate math is non-zero
PRICED_GLOBAL_MODELS = [
    {"provider": "ollama:chat:qwen2.5:7b", "alias": "qwen7b", "tier": "local"},
    {
        "provider": "anthropic:messages:claude-haiku-4-5",
        "alias": "haiku45",
        "tier": "small",
        "price_in_per_mtok": 1.0,
        "price_out_per_mtok": 5.0,
    },
    {
        "provider": "anthropic:messages:claude-opus-4-8",
        "alias": "opus48",
        "tier": "large",
        "price_in_per_mtok": 15.0,
        "price_out_per_mtok": 75.0,
    },
]


def _train_case(i: int, input_text: str) -> GoldenCase:
    return GoldenCase(
        id=f"cost-{i:04d}",
        input=input_text,
        expected="契約照会",
        split="train",
        category="基本",
        difficulty=None,
        source="self-made",
    )


def _set_optimize_method(paths, root, method, params=None):
    raw = yaml.safe_load(paths.task_config.read_text(encoding="utf-8"))
    raw["optimize"]["method"] = method
    if params is not None:
        raw["optimize"]["params"] = params
    paths.task_config.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_task(paths.task, root=root)


def test_estimate_optimize_cost_gepa_price_table_math(isolated_root):
    cfg, _paths = scaffold_task(
        isolated_root,
        global_models=PRICED_GLOBAL_MODELS,
        optimize_target="haiku45",
        reflection_provider="anthropic/claude-opus-4-8",
    )
    # controlled sizes: a 20-char ASCII prefix + 30-char ASCII input is one
    # 50-character run -> ceil(50 / 4) = 13 fallback tokens.
    train = [_train_case(i, "x" * 30) for i in range(4)]
    est = optimize_mod.estimate_optimize_cost(cfg, train, "p" * 20 + "{{input}}")

    assert est.method == "gepa"
    assert est.rollout_factor == 10  # auto=light
    assert est.rollout_count == 40  # 4 train cases x 10
    assert est.reflection_call_count == 10
    assert est.target_input_tokens == 13
    assert est.target_token_count_method == "heuristic:mixed-text-v1"
    # target: 40 calls x (13 in-tokens x $1/M + 12 out-tokens x $5/M)
    assert est.target_usd == pytest.approx(40 * (13 * 1.0 + 12 * 5.0) / 1_000_000)
    # reflection: 10 calls x (3000 in-tokens x $15/M + 500 out-tokens x $75/M)
    assert est.reflection_usd == pytest.approx(10 * (3000 * 15.0 + 500 * 75.0) / 1_000_000)
    assert est.total_usd == pytest.approx(est.target_usd + est.reflection_usd)


def test_estimate_optimize_cost_auto_budget_scaling(isolated_root):
    cfg, paths = scaffold_task(isolated_root)
    train = [_train_case(i, "x" * 30) for i in range(4)]
    light = optimize_mod.estimate_optimize_cost(cfg, train, "p")

    cfg_heavy, _ = _set_optimize_method(paths, isolated_root, "gepa", params={"auto": "heavy"})
    heavy = optimize_mod.estimate_optimize_cost(cfg_heavy, train, "p")

    assert light.rollout_factor < heavy.rollout_factor
    assert light.rollout_count < heavy.rollout_count


def test_estimate_optimize_cost_copro_factor_is_breadth_times_depth(isolated_root):
    _cfg, paths = scaffold_task(isolated_root)
    cfg, _ = _set_optimize_method(paths, isolated_root, "copro", params={"breadth": 5, "depth": 2})
    train = [_train_case(i, "x" * 30) for i in range(3)]
    est = optimize_mod.estimate_optimize_cost(cfg, train, "p")
    assert est.rollout_factor == 10  # breadth 5 x depth 2
    assert est.rollout_count == 30

    # no params -> the pinned dspy defaults (breadth 10 x depth 3)
    cfg_default, _ = _set_optimize_method(paths, isolated_root, "copro", params={})
    assert optimize_mod.estimate_optimize_cost(cfg_default, train, "p").rollout_factor == 30


def test_estimate_optimize_cost_reflection_price_unknown(isolated_root):
    # DEFAULT_GLOBAL_MODELS has no registry entry matching the reflection
    # provider string -> price unknown (None), counted as 0 in the total
    cfg, _paths = scaffold_task(isolated_root)
    train = [_train_case(i, "x" * 30) for i in range(4)]
    est = optimize_mod.estimate_optimize_cost(cfg, train, "p")
    assert est.reflection_usd is None
    assert est.total_usd == est.target_usd


def test_optimize_aborts_when_estimate_exceeds_cost_warn(isolated_root):
    # priced target + tiny cost_warn_usd -> the confirmation fires; declining
    # aborts BEFORE any dspy/promptfoo work, so no stubs are needed
    cfg, paths = scaffold_task(
        isolated_root,
        golden_rows=_label_type_golden_rows(),
        global_models=PRICED_GLOBAL_MODELS,
        optimize_target="haiku45",
        global_run={"cost_warn_usd": 0.000001},
    )
    build_mod.build(cfg, paths, yes=True)

    with pytest.raises(optimize_mod.OptimizeError, match="aborted by user"):
        optimize_mod.optimize(cfg, paths, confirm_fn=lambda msg: False)


def test_optimize_yes_skips_cost_confirmation(isolated_root, monkeypatch):
    cfg, paths = scaffold_task(
        isolated_root,
        golden_rows=_label_type_golden_rows(),
        global_models=PRICED_GLOBAL_MODELS,
        optimize_target="haiku45",
        global_run={"cost_warn_usd": 0.000001},
    )
    build_mod.build(cfg, paths, yes=True)

    fake_optimized = types.SimpleNamespace(
        signature=types.SimpleNamespace(instructions="cost-estimate test instructions")
    )
    monkeypatch.setattr(optimize_mod, "run_gepa", lambda *a, **k: fake_optimized)

    def fake_eval(config_path, output_path, **kwargs):
        cfg_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rows = [
            {
                "vars": {"case_id": f"case-{i:04d}", "expected": "契約照会", "category": "基本"},
                "provider": {"id": p["id"], "label": p["label"]},
                "response": {"output": "契約照会"},
                "gradingResult": {"pass": True, "score": 1},
                "success": True,
                "cost": 0.0,
            }
            for i, p in enumerate(cfg_yaml["providers"], start=1)
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    _stub_run_env(monkeypatch, fake_eval)

    outcome = optimize_mod.optimize(
        cfg, paths, yes=True, confirm_fn=lambda msg: pytest.fail("confirm_fn must not be called with --yes")
    )
    assert outcome.run_id  # the full pipeline ran non-interactively
