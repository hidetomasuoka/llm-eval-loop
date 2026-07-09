"""COPRO optimizer (APO-07): full mocked flow plus the guarantees that matter --
the whole (and only the) TRAIN split reaches the optimizer, params resolve
with defaults from the pinned dspy signature, and the metric is scalar-adapted.
"""

import json
import types

import yaml

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import run as run_mod
from evalloop.schemas import load_task
from tests.conftest import scaffold_task


def _scaffold_copro_task(root, params=None):
    cfg, paths = scaffold_task(root)
    raw = yaml.safe_load(paths.task_config.read_text(encoding="utf-8"))
    raw["optimize"]["method"] = "copro"
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


def test_optimize_end_to_end_with_stubbed_copro_and_promptfoo(isolated_root, monkeypatch):
    cfg, paths = _scaffold_copro_task(isolated_root, params={"breadth": 5, "depth": 2})
    build_mod.build(cfg, paths, yes=True)

    captured = {}

    def fake_copro(student, trainset, metric, prompt_model, breadth, depth, init_temperature):
        captured.update(
            trainset=trainset, metric=metric, breadth=breadth, depth=depth, init_temperature=init_temperature
        )
        return types.SimpleNamespace(signature=types.SimpleNamespace(instructions="copro optimized instructions"))

    monkeypatch.setattr(optimize_mod, "run_copro", fake_copro)
    _stub_promptfoo(monkeypatch)

    # scaffold's 4-case train split keeps the membership assertions exact;
    # force=True demotes the APO-09 preflight errors to warnings
    outcome = optimize_mod.optimize(cfg, paths, force=True)

    # iron rule #1: COPRO has no valset -- the WHOLE train split (and nothing
    # else) is handed over (scaffold train inputs are サンプル1..4; test rows
    # are 101..104)
    train_inputs = {f"問い合わせ文サンプル{i}" for i in range(1, 5)}
    assert {ex.input for ex in captured["trainset"]} == train_inputs

    # params resolve with the requested overrides + the pinned-signature default
    assert captured["breadth"] == 5 and captured["depth"] == 2
    assert captured["init_temperature"] == 1.4

    # the metric handed to COPRO must already be scalar-adapted
    gold = types.SimpleNamespace(expected="契約照会")
    pred = types.SimpleNamespace(output="契約照会")
    assert captured["metric"](gold, pred) == 1.0

    # APO-05 identity plumbing applies to this method too (slug suffix included)
    assert "_copro_" in outcome.variant_name
    assert outcome.task_path.parent.name.startswith("copro-")
    assert "br5" in outcome.variant_name and "d2" in outcome.variant_name
    assert "copro optimized instructions" in outcome.task_path.read_text(encoding="utf-8")
    assert (paths.runs_dir / outcome.run_id / "output.json").exists()

    log = json.loads((outcome.task_path.parent / "optimize_log.json").read_text(encoding="utf-8"))
    assert log["method"] == "copro"
    assert log["params"] == {"breadth": 5, "depth": 2, "auto": "light"}
    assert log["slug"] == "light-br5-d2-n4"
    assert "copro auto=light" in log["summary"]
    # extra_log (effective values) merged
    assert log["breadth"] == 5 and log["depth"] == 2 and log["init_temperature"] == 1.4
    assert log["train_size"] == 4
    assert paths.optimized_index.exists()
    index_lines = [json.loads(l) for l in paths.optimized_index.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(index_lines) == 1
    entry = index_lines[0]
    assert entry["variant_name"] == outcome.variant_name
    assert entry["slug"] == "light-br5-d2-n4"
    assert entry["method"] == "copro"
    assert entry["run_id"] == outcome.run_id
    assert entry["base_run_id"] is None
    assert entry["optimize_log"].endswith("/optimize_log.json")


def test_copro_defaults_match_pinned_dspy_signature(isolated_root, monkeypatch):
    # no params at all -> the defaults from the pinned dspy COPRO signature
    cfg, paths = _scaffold_copro_task(isolated_root)
    build_mod.build(cfg, paths, yes=True)

    captured = {}

    def fake_copro(student, trainset, metric, prompt_model, breadth, depth, init_temperature):
        captured.update(breadth=breadth, depth=depth, init_temperature=init_temperature)
        return types.SimpleNamespace(signature=types.SimpleNamespace(instructions="x"))

    monkeypatch.setattr(optimize_mod, "run_copro", fake_copro)
    _stub_promptfoo(monkeypatch)

    optimize_mod.optimize(cfg, paths, force=True)

    assert captured == {"breadth": 10, "depth": 3, "init_temperature": 1.4}
