import json
import types

import pytest
import yaml

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import run as run_mod
from evalloop.paths import REPO_ROOT, TaskPaths
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
    assert "F1" in feedback


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

    path = optimize_mod.compare("before", "after", paths)
    content = path.read_text(encoding="utf-8")

    assert path.parent == paths.reports_dir
    assert "qwen7b" in content
    assert "+100.0%" in content


def test_compare_missing_run_raises(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    with pytest.raises(optimize_mod.OptimizeError):
        optimize_mod.compare("nope-a", "nope-b", paths)


# ---------------------------------------------------------------------------
# optimize() end-to-end orchestration, with the real GEPA call and the real
# promptfoo subprocess both stubbed out (neither is feasible without live
# API keys / a compatible npx promptfoo install)
# ---------------------------------------------------------------------------


def _label_type_golden_rows():
    labels = ["契約照会", "障害報告", "機能要望", "その他"]
    rows = []
    for i in range(8):
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

    outcome = optimize_mod.optimize(cfg, paths)

    assert outcome.task_path.exists()
    assert paths.optimized_dir in outcome.task_path.parents
    assert "最適化された新しい指示文です。" in outcome.task_path.read_text(encoding="utf-8")
    assert outcome.variant_path.exists()
    assert outcome.run_id
    assert (paths.runs_dir / outcome.run_id / "output.json").exists()
    # the isolated index.jsonl only holds this variant run (base runs are
    # variant=None), so compare is skipped -- deterministically, unlike when
    # this read the developer's real results/index.jsonl
    assert outcome.base_run_id is None
    assert outcome.compare_path is None


def _text_type_golden_rows():
    spans = [
        "This Agreement shall be governed by the laws of the State of New York.",
        "Either party may terminate upon thirty days written notice.",
        "該当条項なし",
    ]
    rows = []
    for i in range(6):
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

    outcome = optimize_mod.optimize(cfg, paths)

    assert outcome.task_path.exists()
    assert "optimized text instructions" in outcome.task_path.read_text(encoding="utf-8")
    # a rollout that echoes the gold answer must score 1.0 through the real
    # metric wiring (incl. the 該当条項なし sentinel case)
    assert captured["scores"] == [1.0] * 6
