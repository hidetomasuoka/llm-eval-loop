"""Shipping gate (improvement plan #4): dev split + McNemar-gated promoted flag.

The optimize auto-run evaluates on dev when the task has one, and a variant is
promoted only when it beats the base dev run with McNemar significance. Tasks
without a dev split keep evaluating on test (with a warning) and never promote.
"""

import json
import types

import yaml

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import run as run_mod
from evalloop.paths import TaskPaths
from tests.conftest import scaffold_task
from tests.test_optimize import _label_type_golden_rows, _stub_run_env

LABELS = ["契約照会", "障害報告", "機能要望", "その他"]

DEV_CASE_IDS = [f"case-{i + 200:04d}" for i in range(8)]


def _golden_rows_with_dev():
    rows = _label_type_golden_rows()
    for i, case_id in enumerate(DEV_CASE_IDS):
        rows.append(
            {
                "id": case_id,
                "input": f"問い合わせ文サンプルdev{i}",
                "expected": LABELS[i % len(LABELS)],
                "split": "dev",
                "meta": {"category": "基本", "source": "self-made"},
            }
        )
    return rows


def _row(case_id, alias, passed):
    return {
        "vars": {"case_id": case_id, "expected": "契約照会"},
        "provider": {"id": "p", "label": alias},
        "response": {"output": "契約照会"},
        "gradingResult": {"pass": passed, "score": 1 if passed else 0},
        "success": passed,
        "cost": 0.0,
    }


def _write_output(runs_dir, run_id, rows):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "output.json").write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")


def _seed_base_run(paths: TaskPaths, run_id: str, alias: str, verdicts: dict[str, bool], *, task_name: str, split):
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    _write_output(paths.runs_dir, run_id, [_row(cid, alias, ok) for cid, ok in verdicts.items()])
    entry = {
        "run_id": run_id,
        "created_at": "2026-01-01T00:00:00Z",
        "task": paths.task,
        "task_name": task_name,
        "variant": None,
        "promptfoo_exit_code": 0,
    }
    if split is not None:
        entry["split"] = split
    with paths.index.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _passing_eval(case_ids):
    """fake run_promptfoo_eval: every provider passes every given case."""

    def fake_eval(config_path, output_path, **kwargs):
        cfg_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rows = [_row(cid, p["label"], True) for p in cfg_yaml["providers"] for cid in case_ids]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    return fake_eval


# --- build artifacts ---------------------------------------------------------


def test_build_writes_dev_tests_and_config_when_dev_cases_exist(isolated_root):
    cfg, paths = scaffold_task(isolated_root, golden_rows=_golden_rows_with_dev())
    build_mod.build(cfg, paths, yes=True)

    assert paths.tests_dev.exists()
    assert paths.promptfoo_config_dev.exists()
    dev_cfg = yaml.safe_load(paths.promptfoo_config_dev.read_text(encoding="utf-8"))
    assert "tests_dev.yaml" in dev_cfg["tests"]
    assert "tests_train" not in paths.promptfoo_config_dev.read_text(encoding="utf-8")
    dev_entries = yaml.safe_load(paths.tests_dev.read_text(encoding="utf-8"))
    assert {e["vars"]["case_id"] for e in dev_entries} == set(DEV_CASE_IDS)
    # the base config still points at test
    base_cfg = yaml.safe_load(paths.promptfoo_config.read_text(encoding="utf-8"))
    assert "tests_test.yaml" in base_cfg["tests"]


def test_build_removes_stale_dev_artifacts_when_dev_split_removed(isolated_root):
    cfg, paths = scaffold_task(isolated_root, golden_rows=_golden_rows_with_dev())
    build_mod.build(cfg, paths, yes=True)
    assert paths.tests_dev.exists()

    # rewrite golden without dev cases and rebuild
    rows = [json.dumps(r, ensure_ascii=False) for r in _label_type_golden_rows()]
    paths.golden.write_text("\n".join(rows) + "\n", encoding="utf-8")
    build_mod.build(cfg, paths, yes=True)

    assert not paths.tests_dev.exists()
    assert not paths.promptfoo_config_dev.exists()


# --- run --split -------------------------------------------------------------


def test_resolve_config_path_dev_requires_dev_config(isolated_root):
    import pytest

    paths = TaskPaths(root=isolated_root, task="t1")
    with pytest.raises(run_mod.RunError, match="dev"):
        run_mod.resolve_config_path(paths, None, split="dev")


def test_resolve_config_path_dev_variant_uses_dev_yaml_suffix(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    paths.variants_dir.mkdir(parents=True)
    (paths.variants_dir / "v1.dev.yaml").write_text("{}", encoding="utf-8")
    resolved = run_mod.resolve_config_path(paths, "v1", split="dev")
    assert resolved.name == "v1.dev.yaml"


def test_run_records_split_in_meta_and_index(isolated_root, monkeypatch):
    cfg, paths = scaffold_task(isolated_root, golden_rows=_golden_rows_with_dev())
    build_mod.build(cfg, paths, yes=True)
    _stub_run_env(monkeypatch, _passing_eval(DEV_CASE_IDS))

    outcome = run_mod.run(cfg, paths, split="dev")

    assert outcome.meta["split"] == "dev"
    assert "--split dev" in outcome.meta["evalloop_command"]
    index_lines = [json.loads(line) for line in paths.index.read_text(encoding="utf-8").splitlines()]
    assert index_lines[-1]["split"] == "dev"


# --- evaluate_shipping_gate (unit) --------------------------------------------


def test_shipping_gate_promotes_on_significant_dev_win(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    _write_output(paths.runs_dir, "base", [_row(cid, "qwen7b", False) for cid in DEV_CASE_IDS])
    _write_output(paths.runs_dir, "opt", [_row(cid, "qwen7b", True) for cid in DEV_CASE_IDS])

    record = optimize_mod.evaluate_shipping_gate(
        optimized_run_id="opt", base_run_id="base", target_alias="qwen7b", paths=paths, gate_split="dev"
    )
    # b=8, c=0 -> p = 2/256 < 0.05
    assert (record.b, record.c, record.n_paired) == (8, 0, 8)
    assert record.p_value is not None and record.p_value < optimize_mod.GATE_ALPHA
    assert record.delta == 1.0
    assert record.promoted is True


def test_shipping_gate_rejects_insignificant_dev_win(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    base_verdicts = [False, False, True, True, True, True, True, True]  # 6/8
    opt_verdicts = [True, False, True, True, True, True, True, True]  # 7/8: b=1, c=0
    _write_output(
        paths.runs_dir, "base", [_row(cid, "qwen7b", ok) for cid, ok in zip(DEV_CASE_IDS, base_verdicts, strict=True)]
    )
    _write_output(
        paths.runs_dir, "opt", [_row(cid, "qwen7b", ok) for cid, ok in zip(DEV_CASE_IDS, opt_verdicts, strict=True)]
    )

    record = optimize_mod.evaluate_shipping_gate(
        optimized_run_id="opt", base_run_id="base", target_alias="qwen7b", paths=paths, gate_split="dev"
    )
    assert record.delta is not None and record.delta > 0
    assert record.p_value == 1.0  # b=1, c=0: two-sided exact p = 2 * 1/2 = 1.0
    assert record.promoted is False


def test_shipping_gate_never_promotes_on_test_split(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    _write_output(paths.runs_dir, "base", [_row(cid, "qwen7b", False) for cid in DEV_CASE_IDS])
    _write_output(paths.runs_dir, "opt", [_row(cid, "qwen7b", True) for cid in DEV_CASE_IDS])

    record = optimize_mod.evaluate_shipping_gate(
        optimized_run_id="opt", base_run_id="base", target_alias="qwen7b", paths=paths, gate_split="test"
    )
    # significant win, but on the test split: informational only, never promoted
    assert record.p_value is not None and record.p_value < optimize_mod.GATE_ALPHA
    assert record.promoted is None


def test_shipping_gate_undecided_without_base_run(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    record = optimize_mod.evaluate_shipping_gate(
        optimized_run_id="opt", base_run_id=None, target_alias="qwen7b", paths=paths, gate_split="dev"
    )
    assert record.promoted is None
    assert record.p_value is None


# --- optimize() end-to-end (stubbed GEPA + promptfoo) --------------------------


def _stub_gepa(monkeypatch, instructions="optimized instructions"):
    fake_program = types.SimpleNamespace(signature=types.SimpleNamespace(instructions=instructions), train_score=0.9)
    monkeypatch.setattr(optimize_mod, "run_gepa", lambda *a, **k: fake_program)


def test_optimize_with_dev_split_runs_dev_and_promotes(isolated_root, monkeypatch, capsys):
    cfg, paths = scaffold_task(isolated_root, golden_rows=_golden_rows_with_dev())
    build_mod.build(cfg, paths, yes=True)
    # base dev run: target alias fails every dev case -> optimized all-pass is a significant win
    _seed_base_run(
        paths, "base-dev", "qwen7b", {cid: False for cid in DEV_CASE_IDS}, task_name=cfg.task.name, split="dev"
    )
    _stub_gepa(monkeypatch)
    _stub_run_env(monkeypatch, _passing_eval(DEV_CASE_IDS))

    outcome = optimize_mod.optimize(cfg, paths, yes=True)

    assert outcome.gate_split == "dev"
    assert outcome.promoted is True
    assert outcome.base_run_id == "base-dev"
    # the dev variant config was written and the auto-run evaluated dev
    assert (paths.variants_dir / f"{outcome.variant_name}.dev.yaml").exists()
    meta = json.loads((paths.runs_dir / outcome.run_id / "meta.json").read_text(encoding="utf-8"))
    assert meta["split"] == "dev"
    log = json.loads((outcome.task_path.parent / "optimize_log.json").read_text(encoding="utf-8"))
    assert log["promoted"] is True
    assert log["gate_split"] == "dev"
    assert log["gate_p_value"] < optimize_mod.GATE_ALPHA
    out = capsys.readouterr().out
    assert "promoted: yes" in out


def test_optimize_with_dev_split_but_no_base_dev_run_is_undecided(isolated_root, monkeypatch, capsys):
    cfg, paths = scaffold_task(isolated_root, golden_rows=_golden_rows_with_dev())
    build_mod.build(cfg, paths, yes=True)
    # only a TEST base run exists; the dev gate must not use it
    _seed_base_run(
        paths,
        "base-test",
        "qwen7b",
        {f"case-{i + 100:04d}": False for i in range(12)},
        task_name=cfg.task.name,
        split=None,  # legacy entry without a split key = test
    )
    _stub_gepa(monkeypatch)
    _stub_run_env(monkeypatch, _passing_eval(DEV_CASE_IDS))

    outcome = optimize_mod.optimize(cfg, paths, yes=True)

    assert outcome.gate_split == "dev"
    assert outcome.promoted is None
    assert outcome.base_run_id is None
    out = capsys.readouterr().out
    # instructs how to establish the baseline (single-token assert: rich may
    # word-wrap the sentence at any space)
    assert "--split" in out
    log = json.loads((outcome.task_path.parent / "optimize_log.json").read_text(encoding="utf-8"))
    assert log["promoted"] is None
    assert log["gate_split"] == "dev"


def test_optimize_without_dev_split_evaluates_test_and_warns(isolated_root, monkeypatch, capsys):
    cfg, paths = scaffold_task(isolated_root, golden_rows=_label_type_golden_rows())
    build_mod.build(cfg, paths, yes=True)
    test_ids = [f"case-{i + 100:04d}" for i in range(12)]
    _seed_base_run(paths, "base-test", "qwen7b", {cid: False for cid in test_ids}, task_name=cfg.task.name, split=None)
    _stub_gepa(monkeypatch)
    _stub_run_env(monkeypatch, _passing_eval(test_ids))

    outcome = optimize_mod.optimize(cfg, paths, yes=True)

    assert outcome.gate_split == "test"
    assert outcome.promoted is None  # significant win, but test never promotes
    meta = json.loads((paths.runs_dir / outcome.run_id / "meta.json").read_text(encoding="utf-8"))
    assert meta["split"] == "test"
    out = capsys.readouterr().out
    assert "no dev split" in out
