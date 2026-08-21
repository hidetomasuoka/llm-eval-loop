"""PROMST-inspired prompt optimizer for agent / multi-step trajectories.

Paper: PROMST (Prompt Optimization for Multi-Step Tasks). Official stacks
typically roll out a live agent per candidate. That is incompatible with this
harness's in-process proxy-metric contract (no promptfoo per candidate, no
hosted-model judge from Python).

This module is a **scratch** adaptation onto the PromptOptimizer contract:

1. Roll out each train case with the instruction-conditioned local policy
   (``optimizers.agent.rollout_from_instruction``) — not a hosted LLM.
2. Locate the first failing step (wrong / extra / missing tool, or answer).
3. Ask ``reflection_lm`` to rewrite the instruction targeting that step.
4. Keep the rewrite only if the TRAIN proxy score improves.

``run_promst()`` is the monkeypatch seam (same convention as ``run_gepa`` /
``run_tapo``). Final holdout evaluation stays promptfoo JSON deep-equality.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import dspy

from evalloop.optimizers.agent import (
    first_failing_step,
    parse_agent_trajectory,
    rollout_from_instruction,
    trajectory_json,
)
from evalloop.optimizers.base import OptimizeError, OptimizeResult
from evalloop.optimizers.metrics import compute_train_score
from evalloop.optimizers.miprov2 import _scalar_metric
from evalloop.optimizers.tapo import _lm_text
from evalloop.schemas import Config


class _AgentProgram:
    """dspy-shaped program whose ``__call__`` is the local trajectory policy."""

    def __init__(self, instructions: str):
        self.signature = dspy.Signature("input -> output", instructions=instructions)

    def __call__(self, **kwargs):
        user_input = kwargs.get("input") or ""
        traj = rollout_from_instruction(self.signature.instructions, user_input)
        return dspy.Prediction(output=trajectory_json(traj))


def _evaluate(instructions: str, trainset: list, metric: Callable) -> float:
    if not trainset:
        return 0.0
    program = _AgentProgram(instructions)
    total = 0.0
    for gold in trainset:
        pred = program(input=gold.input)
        total += float(metric(gold, pred))
    return total / len(trainset)


def _collect_failures(instructions: str, trainset: list, metric: Callable) -> list[dict]:
    program = _AgentProgram(instructions)
    failures: list[dict] = []
    for gold in trainset:
        pred = program(input=gold.input)
        score = float(metric(gold, pred))
        if score >= 1.0 - 1e-9:
            continue
        traj = parse_agent_trajectory(getattr(pred, "output", "") or "")
        expected = getattr(gold, "expected", None)
        failures.append(
            {
                "case_id": getattr(gold, "case_id", None) or getattr(gold, "id", None),
                "input": gold.input,
                "expected": expected,
                "predicted": traj,
                "score": round(score, 6),
                "first_fail": first_failing_step(traj, expected),
            }
        )
    return failures


def _reflect_instruction(prompt_model, instruction: str, failure: dict) -> str:
    fail = failure.get("first_fail") or {}
    prompt = (
        "You optimize an agent's instruction so its tool trajectory matches gold.\n"
        "Rewrite the instruction to fix the FIRST failing step. Keep any routing "
        "rules that already work. Return ONLY the improved instruction.\n\n"
        f"User input:\n{failure.get('input')}\n\n"
        f"Expected trajectory:\n{json.dumps(failure.get('expected'), ensure_ascii=False)}\n"
        f"Predicted trajectory:\n{json.dumps(failure.get('predicted'), ensure_ascii=False)}\n"
        f"First failing step: {json.dumps(fail, ensure_ascii=False)}\n\n"
        f"Current instruction:\n```\n{instruction}\n```\n"
    )
    mutated = _lm_text(prompt_model, prompt)
    return mutated if mutated else instruction


def run_promst(
    student,
    trainset,
    metric,
    prompt_model,
    task_model,
    max_iterations: int,
    seed: int,
    answer_type: str,
):
    """PROMST loop (monkeypatch target). Returns a program with signature.instructions."""
    del task_model, seed  # local policy does not call the task LM; seed reserved
    if max_iterations < 1:
        raise OptimizeError("promst max_iterations must be >= 1")
    if not trainset:
        raise OptimizeError("promst requires a non-empty trainset")
    if answer_type != "agent":
        raise OptimizeError("promst requires task.answer_type=agent")

    scalar = metric
    base = student.signature.instructions
    best_instr = base
    best_score = _evaluate(best_instr, trainset, scalar)
    iterations: list[dict] = []

    for step in range(max_iterations):
        failures = _collect_failures(best_instr, trainset, scalar)
        iteration = {
            "iteration": step,
            "train_score": round(best_score, 6),
            "n_failures": len(failures),
        }
        if not failures:
            iteration["accepted"] = False
            iteration["note"] = "no remaining train failures"
            iterations.append(iteration)
            break
        candidate = _reflect_instruction(prompt_model, best_instr, failures[0])
        cand_score = _evaluate(candidate, trainset, scalar)
        accepted = cand_score > best_score + 1e-12
        iteration.update(
            {
                "candidate_score": round(cand_score, 6),
                "accepted": accepted,
                "first_fail": failures[0]["first_fail"],
            }
        )
        if accepted:
            best_instr = candidate
            best_score = cand_score
        iterations.append(iteration)

    optimized = _AgentProgram(best_instr)
    optimized.promst_iterations = iterations
    optimized.promst_best_score = best_score
    return optimized


class PromstOptimizer:
    """PromptOptimizer implementation: PROMST-inspired trajectory repair."""

    name = "promst"

    def optimize(
        self,
        *,
        base_instructions: str,
        trainset: list,
        metric: Callable,
        task_lm,
        reflection_lm,
        cfg: Config,
    ) -> OptimizeResult:
        params = cfg.optimize.params
        max_iterations = int(params.get("max_iterations", 4))
        seed = int(params.get("seed", 0))
        answer_type = cfg.task.answer_type

        signature = dspy.Signature("input -> output", instructions=base_instructions)
        student = dspy.Predict(signature)

        from evalloop import optimize as optimize_mod

        optimized_program = optimize_mod.run_promst(
            student,
            trainset,
            _scalar_metric(metric),
            reflection_lm,
            task_lm,
            max_iterations,
            seed,
            answer_type,
        )
        extra_log = {
            "max_iterations": max_iterations,
            "seed": seed,
            "train_size": len(trainset),
            "iterations": getattr(optimized_program, "promst_iterations", []),
        }
        best = getattr(optimized_program, "promst_best_score", None)
        if best is not None:
            extra_log["best_score"] = best
        train_score = compute_train_score(trainset, metric, optimized_program)
        if train_score is not None:
            extra_log["train_score"] = train_score
        return OptimizeResult(
            optimized_instructions=optimized_program.signature.instructions,
            method=self.name,
            extra_log=extra_log,
        )
