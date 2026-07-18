import json

import pytest

from evalloop import report as report_mod
from evalloop.paths import TaskPaths
from evalloop.schemas import CaseResult


def _cr(
    alias,
    passed,
    cost=0.001,
    latency_ms=100,
    cached=False,
    error=None,
    case_id="case-0001",
    repeat_index=0,
    token_usage=None,
    raw=None,
):
    return CaseResult(
        case_id=case_id,
        alias=alias,
        provider_id=alias,
        expected="契約照会",
        category="基本",
        output="契約照会",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="ok" if passed else "mismatch",
        cost=cost,
        latency_ms=latency_ms,
        cached=cached,
        token_usage=token_usage or {},
        error=error,
        repeat_index=repeat_index,
        raw=raw or {},
    )


def test_compute_alias_stats_separates_model_and_judge_tokens():
    """Judge consumption must never leak into the model_tokens column, even
    for rows where the provider omitted response.tokenUsage (issue #85)."""
    judged = {"gradingResult": {"tokensUsed": {"prompt": 100, "completion": 20}}}
    results = [
        # model tokens present AND judge tokens present -> counted separately
        _cr("haiku45", True, token_usage={"prompt": 50, "completion": 10}, raw=judged),
        # provider omitted response.tokenUsage (e.g. Ollama): model side is 0,
        # judge side is still counted exactly once
        _cr("haiku45", False, case_id="case-0002", raw=judged),
    ]
    s = report_mod.compute_alias_stats(results)[0]
    assert s.model_prompt_tokens == 50
    assert s.model_completion_tokens == 10
    assert s.avg_model_prompt_tokens == pytest.approx(25.0)
    assert s.avg_model_completion_tokens == pytest.approx(5.0)
    assert s.judge_prompt_tokens == 200
    assert s.judge_completion_tokens == 40


def test_compute_alias_stats_avg_tokens_missing_when_no_model_usage():
    """Providers without tokenUsage (Ollama etc.) should yield n/a averages."""
    results = [
        _cr("ollama", True, token_usage={}),
        _cr("ollama", False, case_id="case-0002", token_usage=None),
    ]
    s = report_mod.compute_alias_stats(results)[0]
    assert s.model_prompt_tokens == 0
    assert s.model_completion_tokens == 0
    assert s.avg_model_prompt_tokens is None
    assert s.avg_model_completion_tokens is None


def test_compute_alias_stats_avg_tokens_per_row():
    results = [
        _cr("haiku45", True, token_usage={"prompt": 100, "completion": 20}),
        _cr("haiku45", False, case_id="case-0002", token_usage={"prompt": 200, "completion": 40}),
    ]
    s = report_mod.compute_alias_stats(results)[0]
    assert s.avg_model_prompt_tokens == pytest.approx(150.0)
    assert s.avg_model_completion_tokens == pytest.approx(30.0)


def test_compute_alias_stats_pass_rate_and_cost():
    results = [
        _cr("haiku45", True, cost=0.001, latency_ms=100),
        _cr("haiku45", False, cost=0.002, latency_ms=200),
        _cr("qwen7b", True, cost=0.0, latency_ms=50),
    ]
    stats = report_mod.compute_alias_stats(results)
    by_alias = {s.alias: s for s in stats}

    assert by_alias["haiku45"].pass_rate == pytest.approx(0.5)
    assert by_alias["haiku45"].total_cost_usd == pytest.approx(0.003)
    assert by_alias["haiku45"].n == 2
    assert by_alias["qwen7b"].pass_rate == 1.0
    assert by_alias["qwen7b"].total_cost_usd == 0.0


def test_compute_alias_stats_cache_rate_and_errors():
    results = [
        _cr("haiku45", True, cached=True),
        _cr("haiku45", True, cached=False),
        _cr("haiku45", None, error="timeout"),
    ]
    stats = report_mod.compute_alias_stats(results)
    s = stats[0]
    assert s.cache_rate == pytest.approx(1 / 3)
    assert s.error_count == 1
    # pass_rate ignores the None (errored) row's passed field
    assert s.pass_rate == pytest.approx(1.0)


def test_wilson_interval_bounds_and_midpoint():
    low, high = report_mod.wilson_interval(0, 10)
    assert low == 0.0 and 0.0 < high < 0.35
    low, high = report_mod.wilson_interval(10, 10)
    assert 0.65 < low < 1.0 and high == 1.0
    low, high = report_mod.wilson_interval(40, 80)
    assert low < 0.5 < high
    # wider interval for smaller n at the same proportion
    low_small, high_small = report_mod.wilson_interval(5, 10)
    assert (high_small - low_small) > (high - low)


def test_wilson_interval_empty_sample_is_maximally_uncertain():
    assert report_mod.wilson_interval(0, 0) == (0.0, 1.0)


def test_compute_alias_stats_single_repeat_has_ci_but_no_repeat_stats():
    stats = report_mod.compute_alias_stats(
        [_cr("haiku45", True, case_id="case-0001"), _cr("haiku45", False, case_id="case-0002")]
    )
    s = stats[0]
    assert s.pass_ci_low is not None and s.pass_ci_high is not None
    assert s.pass_ci_low < s.pass_rate < s.pass_ci_high
    assert s.repeat_pass_rates == []
    assert s.repeat_stddev is None
    assert s.flip_rate is None
    assert s.flip_case_ids == []


def test_compute_alias_stats_repeat_axis_and_flips():
    results = [
        # case-0001: stable pass across both repeats
        _cr("haiku45", True, case_id="case-0001", repeat_index=0),
        _cr("haiku45", True, case_id="case-0001", repeat_index=1),
        # case-0002: flips fail -> pass
        _cr("haiku45", False, case_id="case-0002", repeat_index=0),
        _cr("haiku45", True, case_id="case-0002", repeat_index=1),
    ]
    s = report_mod.compute_alias_stats(results)[0]
    assert s.repeat_pass_rates == pytest.approx([0.5, 1.0])
    assert s.repeat_stddev == pytest.approx(0.3535, abs=1e-3)
    assert s.flip_case_ids == ["case-0002"]
    assert s.flip_rate == pytest.approx(0.5)


def test_render_markdown_repeat_section_only_when_repeats_exist():
    single = report_mod.compute_alias_stats([_cr("haiku45", True)])
    meta = {"task_name": "t", "answer_type": "label", "created_at": "now", "repeat": 1, "limit": None,
            "promptfoo_config_path": "x", "promptfoo_version": "0.1.0"}
    md = report_mod.render_markdown("run-1", meta, single, [])
    assert "pass_95ci" in md
    assert "Repeat stability" not in md

    repeated = report_mod.compute_alias_stats(
        [
            _cr("haiku45", True, case_id="case-0001", repeat_index=0),
            _cr("haiku45", False, case_id="case-0001", repeat_index=1),
        ]
    )
    md = report_mod.render_markdown("run-2", {**meta, "repeat": 2}, repeated, [])
    assert "Repeat stability" in md
    assert "case-0001" in md  # flipped case listed


def test_render_markdown_includes_warnings_and_table():
    stats = report_mod.compute_alias_stats(
        [_cr("haiku45", True, token_usage={"prompt": 100, "completion": 20})]
    )
    md = report_mod.render_markdown(
        "20260101-000000-abcd",
        {"task_name": "t", "answer_type": "label", "created_at": "now", "repeat": 1, "limit": None,
         "promptfoo_config_path": "promptfoo/t1/promptfooconfig.yaml", "promptfoo_version": "0.1.0"},
        stats,
        ["uncalibrated/low-agreement judge: run `evalloop calibrate`"],
    )
    assert "⚠ uncalibrated" in md
    assert "haiku45" in md
    assert "| alias |" in md
    assert "avg_prompt_tokens" in md
    assert "avg_output_tokens" in md
    assert "100.0" in md
    assert "20.0" in md


def test_render_markdown_shows_na_for_missing_token_usage():
    stats = report_mod.compute_alias_stats([_cr("ollama", True, token_usage={})])
    md = report_mod.render_markdown(
        "run-ollama",
        {"task_name": "t", "answer_type": "label", "created_at": "now", "repeat": 1, "limit": None,
         "promptfoo_config_path": "x", "promptfoo_version": "0.1.0"},
        stats,
        [],
    )
    row = [line for line in md.splitlines() if line.startswith("| ollama |")][0]
    assert row.count("n/a") >= 2


def test_render_markdown_variant_caption_includes_prompt_template_length(isolated_root, monkeypatch):
    monkeypatch.setattr(report_mod, "REPO_ROOT", isolated_root)
    prompt_path = isolated_root / "tasks" / "t1" / "optimized" / "qwen7b" / "task.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_text = "optimized prompt template {{input}}"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    stats = report_mod.compute_alias_stats([_cr("qwen7b", True)])
    rel_prompt = prompt_path.relative_to(isolated_root)
    md = report_mod.render_markdown(
        "run-variant",
        {
            "task_name": "t1",
            "answer_type": "label",
            "variant": "qwen7b_opt",
            "prompt_file": str(rel_prompt),
            "created_at": "now",
            "repeat": 1,
            "limit": None,
            "promptfoo_config_path": "x",
            "promptfoo_version": "0.1.0",
        },
        stats,
        [],
    )
    assert f"variant prompt template: {len(prompt_text)} characters" in md
    assert f"`{rel_prompt}`" in md


def test_report_end_to_end(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    run_id = "20260101-000000-abcd"
    run_dir = paths.runs_dir / run_id
    run_dir.mkdir(parents=True)

    output = {
        "results": {
            "results": [
                {
                    "vars": {"case_id": "case-0001", "expected": "契約照会", "category": "基本"},
                    "provider": {"id": "p", "label": "haiku45"},
                    "response": {"output": "契約照会", "cached": False},
                    "gradingResult": {"pass": True, "score": 1, "reason": "ok"},
                    "success": True,
                    "cost": 0.001,
                    "latencyMs": 120,
                }
            ]
        }
    }
    (run_dir / "output.json").write_text(json.dumps(output), encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps({"task_name": "t", "answer_type": "label", "created_at": "now", "repeat": 1, "limit": None,
                    "promptfoo_config_path": "x", "promptfoo_version": "0.1.0",
                    "judge": {"provider": "j", "calibration_status": "uncalibrated"}}),
        encoding="utf-8",
    )

    report_path = report_mod.report(run_id, paths)

    assert report_path.exists()
    assert report_path.parent == paths.reports_dir
    content = report_path.read_text(encoding="utf-8")
    assert "haiku45" in content
    # answer_type=label -> no judge used -> no calibration warning expected
    assert "uncalibrated" not in content


def test_report_missing_run_raises(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    with pytest.raises(report_mod.ReportError):
        report_mod.report("does-not-exist", paths)
