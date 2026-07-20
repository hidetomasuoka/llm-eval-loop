"""APO-11 generalization gate: train proxy vs holdout pass rate (issue #70)."""

import json
import types

import yaml

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop.paths import TaskPaths
from tests.conftest import scaffold_task
from tests.test_optimize import _label_type_golden_rows, _stub_run_env


def _write_output(runs_dir, run_id, rows):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "output.json").write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")


def _row(case_id, alias, passed, cost=0.0):
    return {
        "vars": {"case_id": case_id, "expected": "契約照会"},
        "provider": {"id": "p", "label": alias},
        "response": {"output": "契約照会"},
        "gradingResult": {"pass": passed, "score": 1 if passed else 0},
        "success": passed,
        "cost": cost,
    }


def _seed_base_run(paths: TaskPaths, run_id: str, alias: str, pass_flags: list[bool], *, task_name: str) -> None:
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row(f"case-{i + 1:04d}", alias, passed) for i, passed in enumerate(pass_flags)]
    _write_output(paths.runs_dir, run_id, rows)
    entry = {
        "run_id": run_id,
        "created_at": "2026-01-01T00:00:00Z",
        "task": paths.task,
        "task_name": task_name,
        "variant": None,
        "promptfoo_exit_code": 0,
    }
    paths.index.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def test_evaluate_generalization_gate_pass_and_fail(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    _write_output(
        paths.runs_dir,
        "base",
        [_row("case-0001", "qwen7b", True), _row("case-0002", "qwen7b", True)],
    )
    _write_output(
        paths.runs_dir,
        "better",
        [_row("case-0001", "qwen7b", True), _row("case-0002", "qwen7b", True)],
    )
    _write_output(
        paths.runs_dir,
        "worse",
        [_row("case-0001", "qwen7b", False), _row("case-0002", "qwen7b", False)],
    )

    improved = optimize_mod.evaluate_generalization_gate(
        train_score=0.95,
        optimized_run_id="better",
        base_run_id="base",
        target_alias="qwen7b",
        paths=paths,
    )
    assert improved.train_score == 0.95
    assert improved.holdout_score == 1.0
    assert improved.base_holdout_score == 1.0
    assert improved.holdout_delta == 0.0
    assert improved.generalization == "fail"

    regressed = optimize_mod.evaluate_generalization_gate(
        train_score=1.0,
        optimized_run_id="worse",
        base_run_id="base",
        target_alias="qwen7b",
        paths=paths,
    )
    assert regressed.holdout_score == 0.0
    assert regressed.holdout_delta == -1.0
    assert regressed.generalization == "fail"

    passing = optimize_mod.evaluate_generalization_gate(
        train_score=0.8,
        optimized_run_id="better",
        base_run_id="worse",
        target_alias="qwen7b",
        paths=paths,
    )
    assert passing.holdout_delta == 1.0
    assert passing.generalization == "pass"


def test_evaluate_generalization_gate_without_baseline(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    _write_output(paths.runs_dir, "opt", [_row("case-0001", "qwen7b", True)])

    record = optimize_mod.evaluate_generalization_gate(
        train_score=0.9,
        optimized_run_id="opt",
        base_run_id=None,
        target_alias="qwen7b",
        paths=paths,
    )
    assert record.holdout_score == 1.0
    assert record.base_holdout_score is None
    assert record.generalization is None


def test_optimize_logs_fail_on_overfitting_mock(isolated_root, monkeypatch, capsys):
    cfg, paths = scaffold_task(isolated_root, golden_rows=_label_type_golden_rows())
    build_mod.build(cfg, paths, yes=True)
    _seed_base_run(paths, "base-before", "qwen7b", [True, True, True, True], task_name=cfg.task.name)

    fake_program = types.SimpleNamespace(
        signature=types.SimpleNamespace(instructions="optimized with high train proxy"),
        train_score=1.0,
    )
    monkeypatch.setattr(optimize_mod, "run_gepa", lambda *a, **k: fake_program)

    def fake_eval(config_path, output_path, **kwargs):
        cfg_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rows = [
            {
                "vars": {"case_id": f"case-{i:04d}", "expected": "契約照会", "category": "基本"},
                "provider": {"id": p["id"], "label": p["label"]},
                "response": {"output": "その他"},
                "gradingResult": {"pass": False, "score": 0},
                "success": False,
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

    outcome = optimize_mod.optimize(cfg, paths, yes=True)
    log = json.loads((outcome.task_path.parent / "optimize_log.json").read_text(encoding="utf-8"))

    assert log["train_score"] == 1.0
    assert log["holdout_score"] == 0.0
    assert log["base_holdout_score"] == 1.0
    assert log["holdout_delta"] == -1.0
    assert log["generalization"] == "fail"

    captured = capsys.readouterr().out
    assert "不合格: 過学習の疑い" in captured
    assert outcome.base_run_id == "base-before"


def test_optimize_logs_pass_when_holdout_improves(isolated_root, monkeypatch):
    cfg, paths = scaffold_task(isolated_root, golden_rows=_label_type_golden_rows())
    build_mod.build(cfg, paths, yes=True)
    _seed_base_run(paths, "base-half", "qwen7b", [True, False, True, False], task_name=cfg.task.name)

    fake_program = types.SimpleNamespace(
        signature=types.SimpleNamespace(instructions="holdout improved"),
        train_score=0.85,
    )
    monkeypatch.setattr(optimize_mod, "run_gepa", lambda *a, **k: fake_program)

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

    outcome = optimize_mod.optimize(cfg, paths, yes=True)
    log = json.loads((outcome.task_path.parent / "optimize_log.json").read_text(encoding="utf-8"))

    assert log["train_score"] == 0.85
    assert log["holdout_score"] == 1.0
    assert log["base_holdout_score"] == 0.5
    assert log["holdout_delta"] == 0.5
    assert log["generalization"] == "pass"
