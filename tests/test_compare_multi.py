"""Multi-run compare matrix (APO-13 / issue #72)."""

from __future__ import annotations

import json

import pytest

from evalloop import optimize as optimize_mod
from evalloop.paths import TaskPaths


def _write_output(runs_dir, run_id, rows, *, variant=None, meta_extra=None):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "output.json").write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")
    meta = {"run_id": run_id, "variant": variant, "task_name": "t1"}
    if meta_extra:
        meta.update(meta_extra)
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _row(case_id, alias, passed, cost=0.001, latency_ms=100.0):
    return {
        "vars": {"case_id": case_id, "expected": "契約照会"},
        "provider": {"id": "p", "label": alias},
        "response": {"output": "契約照会"},
        "gradingResult": {"pass": passed, "score": 1 if passed else 0},
        "success": passed,
        "cost": cost,
        "latencyMs": latency_ms,
    }


def test_compare_three_runs_writes_matrix_with_headers_and_disclaimer(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    _write_output(
        paths.runs_dir,
        "base",
        [_row("case-0001", "qwen7b", True, cost=0.01, latency_ms=50)],
        variant=None,
    )
    _write_output(
        paths.runs_dir,
        "gepa-run",
        [_row("case-0001", "qwen7b", True, cost=0.02, latency_ms=80)],
        variant="qwen7b_gepa_20260718-120000_abcd",
    )
    _write_output(
        paths.runs_dir,
        "copro-run",
        [_row("case-0001", "qwen7b", False, cost=0.015, latency_ms=70)],
        variant="qwen7b_copro_20260718-130000_efgh",
    )

    path = optimize_mod.compare(["base", "gepa-run", "copro-run"], paths)
    content = path.read_text(encoding="utf-8")

    assert path.name == "compare_base_gepa-run_copro-run.md"
    assert "# Compare: base vs gepa-run vs copro-run" in content
    assert "## Runs" in content
    assert "variant=`(base)`" in content
    assert "method=`gepa`" in content
    assert "method=`copro`" in content
    assert "pass_rate R1" in content
    assert "cost R2" in content
    assert "p50_ms R3" in content
    assert "search_cost R1" in content
    assert "duration_s R2" in content
    assert "qwen7b" in content
    assert "条件依存" in content
    assert "本タスク・本設定に限る" in content
    # base run has no optimize_log → `-`; variant without log → `n/a`
    assert "| qwen7b |" in content
    assert "| - | - |" in content


def test_compare_four_runs_uses_hashed_filename(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    run_ids = [f"20260718-12000{i}-abcd" for i in range(4)]
    for run_id in run_ids:
        _write_output(paths.runs_dir, run_id, [_row("case-0001", "m", True)])

    path = optimize_mod.compare(run_ids, paths)
    assert path.name.startswith("compare_4runs_")
    assert path.name.endswith(".md")
    assert path.name == optimize_mod._compare_report_filename(run_ids)
    assert len(path.name) < len("compare_" + "_".join(run_ids) + ".md")


def test_compare_requires_at_least_two_unique_runs(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    with pytest.raises(optimize_mod.OptimizeError, match="at least 2"):
        optimize_mod.compare(["only-one"], paths)
    _write_output(paths.runs_dir, "a", [_row("case-0001", "m", True)])
    with pytest.raises(optimize_mod.OptimizeError, match="unique"):
        optimize_mod.compare(["a", "a"], paths)


def test_compare_method_from_optimized_index(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    _write_output(paths.runs_dir, "a", [_row("case-0001", "m", True)], variant=None)
    _write_output(
        paths.runs_dir,
        "b",
        [_row("case-0001", "m", True)],
        variant="custom-variant-name",
    )
    _write_output(
        paths.runs_dir,
        "c",
        [_row("case-0001", "m", False)],
        variant="custom-variant-name-2",
    )
    paths.optimized_dir.mkdir(parents=True)
    paths.optimized_index.write_text(
        json.dumps({"variant_name": "custom-variant-name", "method": "miprov2"}) + "\n",
        encoding="utf-8",
    )

    content = optimize_mod.compare(["a", "b", "c"], paths).read_text(encoding="utf-8")
    assert "method=`miprov2`" in content


def test_compare_matrix_shows_search_cost_and_duration_from_optimize_log(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    _write_output(paths.runs_dir, "base", [_row("case-0001", "m", True)], variant=None)
    _write_output(
        paths.runs_dir,
        "opt-a",
        [_row("case-0001", "m", True)],
        variant="m_gepa_20260719-010000_abcd",
    )
    _write_output(
        paths.runs_dir,
        "opt-b",
        [_row("case-0001", "m", False)],
        variant="m_gepa_20260719-020000_efgh",
    )

    log_rel = "m/gepa-20260719-010000-abcd/optimize_log.json"
    log_path = paths.optimized_dir / log_rel
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        json.dumps({"search_cost_usd": 0.0123, "duration_seconds": 42.5}),
        encoding="utf-8",
    )
    paths.optimized_index.write_text(
        json.dumps(
            {
                "variant_name": "m_gepa_20260719-010000_abcd",
                "method": "gepa",
                "run_id": "opt-a",
                "optimize_log": log_rel,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    content = optimize_mod.compare(["base", "opt-a", "opt-b"], paths).read_text(encoding="utf-8")
    assert "search_cost R1" in content and "duration_s R1" in content
    # R1 base → `-`; R2 logged → values; R3 missing log → `n/a`
    assert "| - | - |" in content
    assert "$0.0123" in content
    assert "42.5" in content
    assert "n/a" in content
    assert "optimize_log.json" in content


def test_compare_matrix_resolves_optimize_log_via_variant_on_reeval(isolated_root):
    """Bugbot: re-eval keeps variant name but gets a new run_id."""
    paths = TaskPaths(root=isolated_root, task="t1")
    variant = "m_gepa_20260719-010000_abcd"
    _write_output(paths.runs_dir, "base", [_row("case-0001", "m", True)], variant=None)
    _write_output(paths.runs_dir, "orig-opt", [_row("case-0001", "m", True)], variant=variant)
    # Later re-eval of the same variant with a different run_id
    _write_output(paths.runs_dir, "reeval-opt", [_row("case-0001", "m", False)], variant=variant)

    log_rel = "m/gepa-20260719-010000-abcd/optimize_log.json"
    log_path = paths.optimized_dir / log_rel
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        json.dumps({"search_cost_usd": 0.05, "duration_seconds": 12.0}),
        encoding="utf-8",
    )
    # Index still points at the original optimize run_id
    paths.optimized_index.write_text(
        json.dumps(
            {
                "variant_name": variant,
                "method": "gepa",
                "run_id": "orig-opt",
                "optimize_log": log_rel,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    content = optimize_mod.compare(["base", "reeval-opt", "orig-opt"], paths).read_text(encoding="utf-8")
    assert content.count("$0.0500") >= 2  # both variant runs resolve the log
    assert "12.0" in content


def test_compare_matrix_missing_optimize_log_file_is_na(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    _write_output(paths.runs_dir, "a", [_row("case-0001", "m", True)], variant=None)
    _write_output(
        paths.runs_dir,
        "b",
        [_row("case-0001", "m", True)],
        variant="m_gepa_x",
    )
    _write_output(
        paths.runs_dir,
        "c",
        [_row("case-0001", "m", True)],
        variant="m_gepa_y",
    )
    paths.optimized_dir.mkdir(parents=True)
    paths.optimized_index.write_text(
        json.dumps(
            {
                "variant_name": "m_gepa_x",
                "method": "gepa",
                "run_id": "b",
                "optimize_log": "missing/optimize_log.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    content = optimize_mod.compare(["a", "b", "c"], paths).read_text(encoding="utf-8")
    # table still renders; missing log → n/a for explore columns on R2
    assert "search_cost R2" in content
    assert "n/a" in content


def test_compare_pair_includes_mcnemar_columns(isolated_root):
    """2-run compare gets paired b/c + mcnemar_p columns (improvement plan #2)."""
    paths = TaskPaths(root=isolated_root, task="t1")
    rows_a = [
        _row("case-0001", "m", False),  # improves in B (b)
        _row("case-0002", "m", True),  # regresses in B (c)
        _row("case-0003", "m", True),
    ]
    rows_b = [
        _row("case-0001", "m", True),
        _row("case-0002", "m", False),
        _row("case-0003", "m", True),
    ]
    _write_output(paths.runs_dir, "base", rows_a)
    _write_output(paths.runs_dir, "after", rows_b)

    content = optimize_mod.compare(["base", "after"], paths).read_text(encoding="utf-8")

    assert "| b/c | mcnemar_p |" in content
    # b=1, c=1 -> two-sided exact p = 1.0
    assert "| 1/1 | 1.000 |" in content
    assert "paired McNemar exact test" in content


def test_compare_pair_mcnemar_na_when_no_shared_cases(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    _write_output(paths.runs_dir, "base", [_row("case-0001", "m", True)])
    _write_output(paths.runs_dir, "after", [_row("case-0002", "m", True)])

    content = optimize_mod.compare(["base", "after"], paths).read_text(encoding="utf-8")

    assert "| n/a | n/a |" in content
