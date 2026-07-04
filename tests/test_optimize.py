import json
import types

import pytest
import yaml

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import run as run_mod

REPO_ROOT = build_mod.REPO_ROOT


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
    real_template = (REPO_ROOT / "prompts" / "base" / "task.txt").read_text(encoding="utf-8")
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
# variant config re-rooting
# ---------------------------------------------------------------------------


def test_reroot_file_refs_adds_prefix_only_to_file_uris():
    obj = {"a": "file://../x.txt", "b": ["file://../y.js", "not-a-file-ref"], "c": 3}
    rerooted = optimize_mod._reroot_file_refs(obj, prefix="../")
    assert rerooted["a"] == "file://../../x.txt"
    assert rerooted["b"][0] == "file://../../y.js"
    assert rerooted["b"][1] == "not-a-file-ref"
    assert rerooted["c"] == 3


def test_build_variant_config_reroots_and_swaps_prompt(tmp_path):
    # promptfooconfig.yaml paths are relative to promptfoo/; the variant lives
    # one level deeper at promptfoo/variants/, so every file:// ref must gain
    # one extra "../"
    fake_task_path = optimize_mod.OPTIMIZED_DIR / "qwen7b" / "20260101-000000" / "task.txt"
    variant_config = optimize_mod.build_variant_config("qwen7b", fake_task_path)

    assert variant_config["prompts"] == [f"file://{optimize_mod.to_variant_relpath(fake_task_path)}"]
    assert variant_config["defaultTest"]["assert"][0]["value"].startswith("file://../../")
    assert variant_config["tests"].startswith("file://../../")
    assert "optimized" in variant_config["description"]


def test_build_variant_config_missing_base_config_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(build_mod, "PROMPTFOO_CONFIG_PATH", tmp_path / "nope.yaml")
    with pytest.raises(optimize_mod.OptimizeError):
        optimize_mod.build_variant_config("qwen7b", tmp_path / "task.txt")


# ---------------------------------------------------------------------------
# _find_latest_base_run
# ---------------------------------------------------------------------------


def test_find_latest_base_run_picks_most_recent_successful_base(tmp_path, monkeypatch):
    index_path = tmp_path / "index.jsonl"
    monkeypatch.setattr(run_mod, "INDEX_PATH", index_path)
    entries = [
        {"run_id": "old", "created_at": "2026-01-01T00:00:00Z", "task_name": "t", "variant": None, "promptfoo_exit_code": 0},
        {"run_id": "variant-run", "created_at": "2026-01-02T00:00:00Z", "task_name": "t", "variant": "x", "promptfoo_exit_code": 0},
        {"run_id": "failed", "created_at": "2026-01-03T00:00:00Z", "task_name": "t", "variant": None, "promptfoo_exit_code": 1},
        {"run_id": "newest-base", "created_at": "2026-01-04T00:00:00Z", "task_name": "t", "variant": None, "promptfoo_exit_code": 0},
    ]
    index_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    assert optimize_mod._find_latest_base_run("t") == "newest-base"


def test_find_latest_base_run_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "INDEX_PATH", tmp_path / "index.jsonl")
    assert optimize_mod._find_latest_base_run("t") is None


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


def test_compare_computes_deltas(tmp_path, monkeypatch):
    import evalloop.report as report_mod

    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(run_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(report_mod, "REPORTS_DIR", reports_dir)

    _write_output(runs_dir, "before", [_row("case-0001", "qwen7b", False, cost=0.0), _row("case-0002", "qwen7b", False, cost=0.0)])
    _write_output(runs_dir, "after", [_row("case-0001", "qwen7b", True, cost=0.0), _row("case-0002", "qwen7b", True, cost=0.0)])

    path = optimize_mod.compare("before", "after")
    content = path.read_text(encoding="utf-8")

    assert "qwen7b" in content
    assert "+100.0%" in content


def test_compare_missing_run_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "RUNS_DIR", tmp_path / "runs")
    with pytest.raises(optimize_mod.OptimizeError):
        optimize_mod.compare("nope-a", "nope-b")


# ---------------------------------------------------------------------------
# optimize() end-to-end orchestration, with the real GEPA call and the real
# promptfoo subprocess both stubbed out (neither is feasible without live
# API keys / a compatible npx promptfoo install)
# ---------------------------------------------------------------------------


def test_optimize_end_to_end_with_stubbed_gepa_and_promptfoo(monkeypatch):
    build_mod.build(config_path=REPO_ROOT / "config.yaml", yes=True)

    fake_optimized = types.SimpleNamespace(
        signature=types.SimpleNamespace(instructions="最適化された新しい指示文です。")
    )
    monkeypatch.setattr(optimize_mod, "run_gepa", lambda *a, **k: fake_optimized)

    def fake_eval(config_path, output_path, **kwargs):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rows = [
            {
                "vars": {"case_id": f"case-{i:04d}", "expected": "契約照会", "category": "基本"},
                "provider": {"id": p["id"], "label": p["label"]},
                "response": {"output": "契約照会"},
                "gradingResult": {"pass": True, "score": 1},
                "success": True,
                "cost": 0.0,
            }
            for i, p in enumerate(cfg["providers"], start=1)
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    outcome = optimize_mod.optimize(config_path=REPO_ROOT / "config.yaml")

    assert outcome.task_path.exists()
    assert "最適化された新しい指示文です。" in outcome.task_path.read_text(encoding="utf-8")
    assert outcome.variant_path.exists()
    assert outcome.run_id
    assert (run_mod.RUNS_DIR / outcome.run_id / "output.json").exists()
    # no prior base run existed in this project's real index.jsonl -> compare skipped
    assert outcome.base_run_id is None
    assert outcome.compare_path is None


def test_optimize_rejects_non_label_answer_type(monkeypatch, tmp_path):
    bad_config = tmp_path / "config.yaml"
    bad_config.write_text(
        (REPO_ROOT / "config.yaml").read_text(encoding="utf-8").replace("answer_type: label", "answer_type: text"),
        encoding="utf-8",
    )
    with pytest.raises(optimize_mod.OptimizeError):
        optimize_mod.optimize(config_path=bad_config)
