"""TAPO optimizer (APO-23): mocked end-to-end + metric selection + param plumbing."""

from __future__ import annotations

import json
import types

import pytest
import yaml

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import run as run_mod
from evalloop.optimizers.base import OptimizeError
from evalloop.optimizers.tapo import select_metrics_for_answer_type
from evalloop.schemas import load_task
from tests.conftest import scaffold_task


def _scaffold_tapo_task(root, params=None):
    cfg, paths = scaffold_task(root)
    raw = yaml.safe_load(paths.task_config.read_text(encoding="utf-8"))
    raw["optimize"]["method"] = "tapo"
    if params is not None:
        raw["optimize"]["params"] = params
    paths.task_config.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_task(paths.task, root=root)


def _stub_promptfoo(monkeypatch):
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


def test_select_metrics_for_answer_type():
    assert select_metrics_for_answer_type("label") == ["label_match", "single_line_brevity"]
    assert select_metrics_for_answer_type("json") == ["json_deep_equal", "valid_json"]
    assert select_metrics_for_answer_type("text") == ["token_f1", "length_ratio"]
    with pytest.raises(OptimizeError, match="unsupported"):
        select_metrics_for_answer_type("other")


def test_optimize_end_to_end_with_stubbed_tapo_and_promptfoo(isolated_root, monkeypatch):
    cfg, paths = _scaffold_tapo_task(isolated_root, params={"population_size": 3, "generations": 2, "seed": 9})
    build_mod.build(cfg, paths, yes=True)

    captured = {}

    def fake_tapo(
        student,
        trainset,
        metric,
        prompt_model,
        task_model,
        population_size,
        generations,
        seed,
        answer_type,
    ):
        captured.update(
            trainset=trainset,
            population_size=population_size,
            generations=generations,
            seed=seed,
            answer_type=answer_type,
        )
        prog = types.SimpleNamespace(
            signature=types.SimpleNamespace(instructions="tapo optimized instructions"),
            tapo_selected_metrics=["label_match", "single_line_brevity"],
            tapo_generation_scores=[
                {"generation": 0, "best_fitness": 0.5, "mean_fitness": 0.4},
                {"generation": 1, "best_fitness": 0.9, "mean_fitness": 0.7},
            ],
            tapo_best_fitness=0.9,
        )
        return prog

    monkeypatch.setattr(optimize_mod, "run_tapo", fake_tapo)
    _stub_promptfoo(monkeypatch)

    outcome = optimize_mod.optimize(cfg, paths, force=True)

    train_inputs = {f"問い合わせ文サンプル{i}" for i in range(1, 5)}
    assert {ex.input for ex in captured["trainset"]} == train_inputs
    assert captured["population_size"] == 3
    assert captured["generations"] == 2
    assert captured["seed"] == 9
    assert captured["answer_type"] == "label"

    assert "_tapo_" in outcome.variant_name
    assert outcome.task_path.parent.name.startswith("tapo-")
    assert "pop3" in outcome.variant_name and "gen2" in outcome.variant_name
    assert "tapo optimized instructions" in outcome.task_path.read_text(encoding="utf-8")

    log = json.loads((outcome.task_path.parent / "optimize_log.json").read_text(encoding="utf-8"))
    assert log["method"] == "tapo"
    assert log["population_size"] == 3
    assert log["generations"] == 2
    assert log["seed"] == 9
    assert log["selected_metrics"] == ["label_match", "single_line_brevity"]
    assert log["generation_scores"][1]["best_fitness"] == 0.9
    assert log["best_fitness"] == 0.9
    assert log["train_size"] == 4


def test_tapo_defaults(isolated_root, monkeypatch):
    cfg, paths = _scaffold_tapo_task(isolated_root, params={})
    build_mod.build(cfg, paths, yes=True)
    captured = {}

    def fake_tapo(*a, **k):
        captured.update(
            population_size=a[5],
            generations=a[6],
            seed=a[7],
        )
        return types.SimpleNamespace(
            signature=types.SimpleNamespace(instructions="tapo default"),
            tapo_selected_metrics=["label_match", "single_line_brevity"],
            tapo_generation_scores=[],
            tapo_best_fitness=0.0,
        )

    monkeypatch.setattr(optimize_mod, "run_tapo", fake_tapo)
    _stub_promptfoo(monkeypatch)
    optimize_mod.optimize(cfg, paths, force=True)
    assert captured == {"population_size": 4, "generations": 3, "seed": 0}
