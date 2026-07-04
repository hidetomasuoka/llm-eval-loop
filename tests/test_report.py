import json

import pytest

from evalloop import report as report_mod
from evalloop.schemas import CaseResult


def _cr(alias, passed, cost=0.001, latency_ms=100, cached=False, error=None):
    return CaseResult(
        case_id="case-0001",
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
        token_usage={},
        error=error,
        repeat_index=0,
    )


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


def test_render_markdown_includes_warnings_and_table():
    stats = report_mod.compute_alias_stats([_cr("haiku45", True)])
    md = report_mod.render_markdown(
        "20260101-000000-abcd",
        {"task_name": "t", "answer_type": "label", "created_at": "now", "repeat": 1, "limit": None,
         "promptfoo_config_path": "promptfoo/promptfooconfig.yaml", "promptfoo_version": "0.1.0"},
        stats,
        ["uncalibrated/low-agreement judge: run `evalloop calibrate`"],
    )
    assert "⚠ uncalibrated" in md
    assert "haiku45" in md
    assert "| alias |" in md


def test_report_end_to_end(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(report_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(report_mod, "REPORTS_DIR", reports_dir)

    run_id = "20260101-000000-abcd"
    run_dir = runs_dir / run_id
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

    report_path = report_mod.report(run_id)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "haiku45" in content
    # answer_type=label -> no judge used -> no calibration warning expected
    assert "uncalibrated" not in content


def test_report_missing_run_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(report_mod, "REPORTS_DIR", tmp_path / "reports")
    with pytest.raises(report_mod.ReportError):
        report_mod.report("does-not-exist")
