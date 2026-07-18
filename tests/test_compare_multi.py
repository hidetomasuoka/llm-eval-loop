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
    assert "qwen7b" in content
    assert "条件依存" in content
    assert "本タスク・本設定に限る" in content


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
