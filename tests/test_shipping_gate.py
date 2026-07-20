"""Shipping gate (improvement plan #4): dev split + McNemar-gated promoted flag.

The optimize auto-run evaluates on dev when the task has one, and a variant is
promoted only when it beats the base dev run with McNemar significance. Tasks
without a dev split keep evaluating on test (with a warning) and never promote.
"""

import json
import types

import pytest
import yaml

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import run as run_mod
from evalloop.paths import TaskPaths
from evalloop.schemas import SchemaError
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

    # a leftover optimized variant's dev config, as if `evalloop optimize`
    # had run before the dev split was removed
    paths.variants_dir.mkdir(parents=True, exist_ok=True)
    stale_variant_dev = paths.variants_dir / "qwen7b_gepa_20260101-000000.dev.yaml"
    stale_variant_dev.write_text("tests: file://../tests_dev.yaml\n", encoding="utf-8")

    # rewrite golden without dev cases and rebuild
    rows = [json.dumps(r, ensure_ascii=False) for r in _label_type_golden_rows()]
    paths.golden.write_text("\n".join(rows) + "\n", encoding="utf-8")
    build_mod.build(cfg, paths, yes=True)

    assert not paths.tests_dev.exists()
    assert not paths.promptfoo_config_dev.exists()
    # otherwise `evalloop run --variant ... --split dev` would resolve to this
    # file and hand promptfoo a tests: reference that no longer exists
    assert not stale_variant_dev.exists()


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


# --- optimize(): iron rule #1 re-check covers dev/test too --------------------


def test_optimize_rejects_stale_dev_test_overlap_independently_of_build(isolated_root):
    """build.py checks train/test, train/dev, and dev/test disjointness at
    build time, but optimize.py re-checks independently in case the build
    artifacts are stale (golden.jsonl edited without a rebuild). The old
    re-check only compared train against the *merged* test+dev holdout, so a
    dev/test overlap introduced after the last build was invisible to it --
    the union silently absorbed the duplicate id.
    """
    cfg, paths = scaffold_task(isolated_root, golden_rows=_golden_rows_with_dev())
    build_mod.build(cfg, paths, yes=True)

    # Simulate a stale tests_dev.yaml: a test case id now also appears in dev.
    test_entries = yaml.safe_load(paths.tests_test.read_text(encoding="utf-8"))
    dev_entries = yaml.safe_load(paths.tests_dev.read_text(encoding="utf-8"))
    dev_entries.append(test_entries[0])
    paths.tests_dev.write_text(yaml.safe_dump(dev_entries, allow_unicode=True), encoding="utf-8")

    with pytest.raises(SchemaError, match="dev/test"):
        optimize_mod.optimize(cfg, paths, yes=True)


def test_optimize_demo_leak_check_covers_golden_dev_cases(isolated_root):
    """The demo-leak backstop unions golden.jsonl's holdout cases with the
    build YAML holdout so a golden edit without a rebuild is still caught.
    It must include split=='dev' cases, not just 'test': the shipping gate
    decision is made on dev, so a demo leaking into a post-build dev case is
    exactly as dangerous as leaking into test.
    """
    cfg, paths = scaffold_task(isolated_root, golden_rows=_golden_rows_with_dev())
    build_mod.build(cfg, paths, yes=True)

    # A demo whose input matches a dev case that exists in golden.jsonl but
    # NOT yet in the (stale) tests_dev.yaml on disk.
    paths.demos.write_text(
        json.dumps({"id": "demo-1", "input": "問い合わせ文サンプルdev0", "output": "契約照会"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    prompt_with_demos = paths.task_dir / "prompts" / "task.txt"
    prompt_with_demos.write_text(prompt_with_demos.read_text(encoding="utf-8") + "\n{{demos}}\n", encoding="utf-8")

    with pytest.raises(optimize_mod.OptimizeError, match="leak"):
        optimize_mod.optimize(cfg, paths, yes=True)


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


def test_shipping_gate_rejects_significant_p_when_paired_direction_regresses(isolated_root):
    """delta and the paired transition can disagree because delta is the
    per-row-averaged pass rate over each run's OWN rows (any non-shared
    cases included), while b/c/p_value only look at the case intersection.
    Here the 6 cases common to both runs regress unanimously (c=6, b=0,
    p<0.05) but base also carries 10 base-only failing rows that drag its
    overall rate down, so the naive delta still comes out positive. The gate
    must not promote on p alone when the paired direction disagrees with it.
    """
    paths = TaskPaths(root=isolated_root, task="t1")
    common_ids = [f"case-{i:04d}" for i in range(1, 7)]
    base_only_ids = [f"case-{i:04d}" for i in range(101, 111)]
    opt_only_ids = [f"case-{i:04d}" for i in range(201, 205)]
    base_rows = [_row(cid, "qwen7b", True) for cid in common_ids] + [
        _row(cid, "qwen7b", False) for cid in base_only_ids
    ]
    opt_rows = [_row(cid, "qwen7b", False) for cid in common_ids] + [_row(cid, "qwen7b", True) for cid in opt_only_ids]
    _write_output(paths.runs_dir, "base", base_rows)
    _write_output(paths.runs_dir, "opt", opt_rows)

    record = optimize_mod.evaluate_shipping_gate(
        optimized_run_id="opt", base_run_id="base", target_alias="qwen7b", paths=paths, gate_split="dev"
    )
    assert record.delta is not None and record.delta > 0  # 4/10 > 6/16 on the naive per-row rate
    assert (record.b, record.c) == (0, 6)  # every shared case regressed
    assert record.p_value is not None and record.p_value < optimize_mod.GATE_ALPHA
    assert record.promoted is False


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
