"""MIPROv2 optimizer (APO-06): full mocked flow + the guarantees that matter --
the validation set is carved deterministically out of the TRAIN split only,
and the GEPA-style metric is adapted to MIPROv2's scalar contract.
"""

import json
import types

import pytest
import yaml

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import run as run_mod
from evalloop.optimizers.base import OptimizeError
from evalloop.optimizers import miprov2 as miprov2_mod
from evalloop.optimizers.miprov2 import MiproV2Optimizer, _scalar_metric, split_train_val
from evalloop.schemas import load_task
from tests.conftest import scaffold_task

# ---------------------------------------------------------------------------
# split_train_val
# ---------------------------------------------------------------------------


def test_split_train_val_ratio_and_determinism():
    items = [f"ex-{i}" for i in range(10)]
    train_a, val_a = split_train_val(items, val_ratio=0.2, seed=0)
    train_b, val_b = split_train_val(items, val_ratio=0.2, seed=0)
    assert (train_a, val_a) == (train_b, val_b)  # fixed seed -> reproducible
    assert len(val_a) == 2 and len(train_a) == 8
    assert sorted(train_a + val_a) == sorted(items)  # a partition: no loss, no overlap


def test_split_train_val_always_keeps_both_sides_nonempty():
    train, val = split_train_val(["a", "b"], val_ratio=0.9, seed=0)
    assert len(train) == 1 and len(val) == 1


def test_split_train_val_guards():
    with pytest.raises(OptimizeError, match="val_ratio"):
        split_train_val(["a", "b", "c"], val_ratio=1.5, seed=0)
    with pytest.raises(OptimizeError, match="at least 2"):
        split_train_val(["only-one"], val_ratio=0.2, seed=0)


def test_miprov2_train_score_uses_train_part_only(monkeypatch):
    """Bugbot: train_score must average train_part, not full trainset+val."""
    trainset = [types.SimpleNamespace(input=f"x{i}") for i in range(10)]
    train_part, _val_part = split_train_val(trainset, val_ratio=0.2, seed=0)
    assert len(train_part) == 8

    seen: dict = {}

    def fake_score(examples, metric, optimized_program):
        seen["n"] = len(examples)
        seen["ids"] = [ex.input for ex in examples]
        return 0.5

    monkeypatch.setattr(miprov2_mod, "compute_train_score", fake_score)
    monkeypatch.setattr(
        optimize_mod,
        "run_miprov2",
        lambda *a, **k: types.SimpleNamespace(
            signature=types.SimpleNamespace(instructions="opt")
        ),
    )

    cfg = types.SimpleNamespace(
        optimize=types.SimpleNamespace(
            auto="light",
            params={"val_ratio": 0.2, "seed": 0},
        )
    )
    result = MiproV2Optimizer().optimize(
        base_instructions="base",
        trainset=trainset,
        metric=lambda *a, **k: 1.0,
        task_lm=object(),
        reflection_lm=object(),
        cfg=cfg,
    )
    assert seen["n"] == len(train_part)
    assert seen["ids"] == [ex.input for ex in train_part]
    assert result.extra_log["train_score"] == 0.5
    assert result.extra_log["train_size"] == len(train_part)


# ---------------------------------------------------------------------------
# metric adaptation
# ---------------------------------------------------------------------------


def test_scalar_metric_unwraps_prediction_score():
    def gepa_style_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return types.SimpleNamespace(score=0.75, feedback="unused by miprov2")

    wrapped = _scalar_metric(gepa_style_metric)
    assert wrapped(gold=None, pred=None) == 0.75


# ---------------------------------------------------------------------------
# full flow (mocked dspy + promptfoo)
# ---------------------------------------------------------------------------


def _scaffold_miprov2_task(root, params=None):
    cfg, paths = scaffold_task(root)
    raw = yaml.safe_load(paths.task_config.read_text(encoding="utf-8"))
    raw["optimize"]["method"] = "miprov2"
    if params is not None:
        raw["optimize"]["params"] = params
    paths.task_config.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_task(paths.task, root=root)


def test_optimize_end_to_end_with_stubbed_miprov2_and_promptfoo(isolated_root, monkeypatch):
    cfg, paths = _scaffold_miprov2_task(isolated_root, params={"val_ratio": 0.25, "seed": 7})
    build_mod.build(cfg, paths, yes=True)

    captured = {}

    def fake_miprov2(
        student,
        trainset,
        valset,
        metric,
        prompt_model,
        task_model,
        auto,
        seed,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
    ):
        captured.update(
            trainset=trainset,
            valset=valset,
            metric=metric,
            auto=auto,
            seed=seed,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
        )
        return types.SimpleNamespace(signature=types.SimpleNamespace(instructions="miprov2 optimized instructions"))

    monkeypatch.setattr(optimize_mod, "run_miprov2", fake_miprov2)

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

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)
    monkeypatch.setattr(run_mod, "get_promptfoo_version", lambda: "0.0.0-test")
    monkeypatch.setattr(run_mod, "get_node_version", lambda: "v22.22.0")

    # the scaffold's 4-case train split is deliberately tiny so the val-split
    # assertions below stay exact; force=True demotes the APO-09 preflight
    # errors to warnings (and covers the --force path end to end)
    outcome = optimize_mod.optimize(cfg, paths, force=True)

    # iron rule #1: everything the optimizer saw must come from the TRAIN
    # split (scaffold trainset inputs are サンプル1..4; test rows are 101..104)
    train_inputs = {f"問い合わせ文サンプル{i}" for i in range(1, 5)}
    seen = [ex.input for ex in captured["trainset"]] + [ex.input for ex in captured["valset"]]
    assert seen and set(seen) <= train_inputs
    # 4 train cases at val_ratio 0.25 -> 1 validation / 3 training examples
    assert len(captured["valset"]) == 1 and len(captured["trainset"]) == 3
    assert captured["auto"] == "light" and captured["seed"] == 7
    # the metric handed to MIPROv2 must already be scalar-adapted
    gold = types.SimpleNamespace(expected="契約照会")
    pred = types.SimpleNamespace(output="契約照会")
    assert captured["metric"](gold, pred) == 1.0

    # APO-05 identity plumbing applies to this method too (slug suffix included)
    assert "_miprov2_" in outcome.variant_name
    assert outcome.task_path.parent.name.startswith("miprov2-")
    assert "val0.25" in outcome.variant_name and "seed7" in outcome.variant_name
    assert "miprov2 optimized instructions" in outcome.task_path.read_text(encoding="utf-8")
    assert (paths.runs_dir / outcome.run_id / "output.json").exists()

    log = json.loads((outcome.task_path.parent / "optimize_log.json").read_text(encoding="utf-8"))
    assert log["method"] == "miprov2"
    assert log["params"] == {"val_ratio": 0.25, "seed": 7, "auto": "light"}
    assert log["slug"] == "light-seed7-val0.25-n4"
    assert "miprov2 auto=light" in log["summary"]
    assert log["val_ratio"] == 0.25 and log["seed"] == 7  # extra_log (effective values) merged
    assert log["train_size"] == 3 and log["val_size"] == 1
    assert paths.optimized_index.exists()
    index_lines = [
        json.loads(line) for line in paths.optimized_index.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(index_lines) == 1
    entry = index_lines[0]
    assert entry["variant_name"] == outcome.variant_name
    assert entry["slug"] == "light-seed7-val0.25-n4"
    assert entry["method"] == "miprov2"
    assert entry["run_id"] == outcome.run_id
    assert entry["base_run_id"] is None
    assert entry["optimize_log"].endswith("/optimize_log.json")


def test_miprov2_passes_demo_params_to_dspy_wrapper(isolated_root, monkeypatch):
    cfg, paths = _scaffold_miprov2_task(
        isolated_root,
        params={
            "val_ratio": 0.25,
            "seed": 3,
            "max_bootstrapped_demos": 2,
            "max_labeled_demos": 4,
        },
    )
    build_mod.build(cfg, paths, yes=True)

    captured = {}

    def fake_miprov2(
        student,
        trainset,
        valset,
        metric,
        prompt_model,
        task_model,
        auto,
        seed,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
    ):
        captured.update(
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            seed=seed,
        )
        return types.SimpleNamespace(signature=types.SimpleNamespace(instructions="miprov2 demos optimized"))

    monkeypatch.setattr(optimize_mod, "run_miprov2", fake_miprov2)

    def fake_eval(config_path, output_path, **kwargs):
        cfg_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rows = [
            {
                "vars": {"case_id": "case-0001", "expected": "契約照会", "category": "基本"},
                "provider": {"id": cfg_yaml["providers"][0]["id"], "label": cfg_yaml["providers"][0]["label"]},
                "response": {"output": "契約照会"},
                "gradingResult": {"pass": True, "score": 1},
                "success": True,
                "cost": 0.0,
            }
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)
    monkeypatch.setattr(run_mod, "get_promptfoo_version", lambda: "0.0.0-test")
    monkeypatch.setattr(run_mod, "get_node_version", lambda: "v22.22.0")

    outcome = optimize_mod.optimize(cfg, paths, force=True)

    assert captured == {"max_bootstrapped_demos": 2, "max_labeled_demos": 4, "seed": 3}
    log = json.loads((outcome.task_path.parent / "optimize_log.json").read_text(encoding="utf-8"))
    assert log["max_bootstrapped_demos"] == 2
    assert log["max_labeled_demos"] == 4


def test_optimize_rejects_miprov2_eval_budget_one_before_model_calls(isolated_root):
    cfg, paths = _scaffold_miprov2_task(
        isolated_root,
        params={"eval_scheduler": "random", "eval_budget": 1, "seed": 3},
    )
    build_mod.build(cfg, paths, yes=True)

    with pytest.raises(OptimizeError, match="at least 2 cases after eval scheduling"):
        optimize_mod.optimize(cfg, paths, force=True)
