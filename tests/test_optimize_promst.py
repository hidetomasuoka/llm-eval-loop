"""PROMST optimizer: mocked end-to-end + real local-policy loop."""

from __future__ import annotations

import json
import types

import dspy
import pytest
import yaml

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import run as run_mod
from evalloop.optimizers.base import OptimizeError
from evalloop.optimizers.metrics import agent_score_and_feedback
from evalloop.optimizers.promst import run_promst
from evalloop.schemas import load_task
from tests.conftest import default_agent_golden_rows, scaffold_task

STRONG_INSTRUCTION = """\
障害報告: kb.search → ticket.create。回答: インシデントを起票しました。
契約照会: kb.search。回答: 契約FAQを案内しました。
機能要望: ツールなし。回答: 要望を記録しました。
その他: ツールなし。回答: 担当へ引き継ぎます。
"""


def _scaffold_promst_task(root, params=None):
    cfg, paths = scaffold_task(
        root,
        answer_type="agent",
        labels=[],
        golden_rows=default_agent_golden_rows(),
        prompt="エージェント指示\n{{input}}\n",
    )
    raw = yaml.safe_load(paths.task_config.read_text(encoding="utf-8"))
    raw["optimize"]["method"] = "promst"
    if params is not None:
        raw["optimize"]["params"] = params
    paths.task_config.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_task(paths.task, root=root)


def _stub_promptfoo(monkeypatch):
    def fake_eval(config_path, output_path, **kwargs):
        cfg_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rows = [
            {
                "vars": {"case_id": f"case-{i:04d}", "expected": {"tools": [], "answer": "x"}},
                "provider": {"id": p["id"], "label": p["label"]},
                "response": {"output": '{"tools": [], "answer": "x"}'},
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


def test_optimize_end_to_end_with_stubbed_promst_and_promptfoo(isolated_root, monkeypatch):
    cfg, paths = _scaffold_promst_task(isolated_root, params={"max_iterations": 3, "seed": 7})
    build_mod.build(cfg, paths, yes=True)

    captured = {}

    def fake_promst(
        student,
        trainset,
        metric,
        prompt_model,
        task_model,
        max_iterations,
        seed,
        answer_type,
    ):
        captured.update(
            trainset=trainset,
            max_iterations=max_iterations,
            seed=seed,
            answer_type=answer_type,
        )
        prog = types.SimpleNamespace(
            signature=types.SimpleNamespace(instructions="promst optimized instructions"),
            promst_iterations=[{"iteration": 0, "train_score": 0.4, "accepted": True}],
            promst_best_score=0.85,
        )
        return prog

    monkeypatch.setattr(optimize_mod, "run_promst", fake_promst)
    _stub_promptfoo(monkeypatch)

    outcome = optimize_mod.optimize(cfg, paths, force=True)

    assert captured["max_iterations"] == 3
    assert captured["seed"] == 7
    assert captured["answer_type"] == "agent"
    assert len(captured["trainset"]) == 4
    assert "_promst_" in outcome.variant_name
    assert "promst optimized instructions" in outcome.task_path.read_text(encoding="utf-8")

    log = json.loads((outcome.task_path.parent / "optimize_log.json").read_text(encoding="utf-8"))
    assert log["method"] == "promst"
    assert log["max_iterations"] == 3
    assert log["seed"] == 7
    assert log["best_score"] == 0.85
    assert log["train_size"] == 4


def test_run_promst_rejects_non_agent_answer_type():
    student = types.SimpleNamespace(signature=types.SimpleNamespace(instructions="x"))
    with pytest.raises(OptimizeError, match="answer_type=agent"):
        run_promst(student, [object()], lambda *a, **k: 0.0, lambda p: "y", None, 1, 0, "label")


def test_run_promst_local_policy_improves_with_reflection_stub():
    trainset = []
    for row in default_agent_golden_rows(n_train=4, n_test=0):
        trainset.append(
            dspy.Example(input=row["input"], expected=row["expected"], case_id=row["id"]).with_inputs("input")
        )

    def metric(gold, pred, trace=None):
        score, _fb = agent_score_and_feedback(getattr(pred, "output", ""), gold.expected)
        return score

    student = types.SimpleNamespace(
        signature=types.SimpleNamespace(
            instructions="よく分からないときは kb.search を使ってください。JSONのみ。"
        )
    )
    optimized = run_promst(
        student,
        trainset,
        metric,
        lambda prompt: STRONG_INSTRUCTION,
        task_model=None,
        max_iterations=2,
        seed=0,
        answer_type="agent",
    )
    assert optimized.signature.instructions.strip() == STRONG_INSTRUCTION.strip()
    assert optimized.promst_best_score == 1.0
    assert any(step.get("accepted") for step in optimized.promst_iterations)
