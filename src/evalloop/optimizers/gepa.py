"""GEPA prompt optimizer: dspy.teleprompt.GEPA behind the PromptOptimizer contract.

Confirmed against the installed dspy==3.2.1 API (dspy.ai docs + `inspect.signature`):
    from dspy.teleprompt import GEPA
    GEPA(metric, *, auto=None, reflection_lm=None, seed=0, ...)
    GEPA.compile(student, *, trainset, teacher=None, valset=None)
    metric(gold, pred, trace, pred_name, pred_trace) -> dspy.Prediction(score=, feedback=)
"""

from __future__ import annotations

from collections.abc import Callable

import dspy

from evalloop.optimizers.base import OptimizeResult
from evalloop.schemas import Config


def run_gepa(student, trainset, metric, reflection_lm, auto: str, seed: int = 0):
    """Thin, monkeypatchable wrapper around the real dspy.teleprompt.GEPA call
    so orchestration logic (file writing, variant config, run/report/compare)
    can be unit-tested without spending real API calls.
    """
    from dspy.teleprompt import GEPA

    optimizer = GEPA(metric=metric, reflection_lm=reflection_lm, auto=auto, seed=seed)
    return optimizer.compile(student=student, trainset=trainset)


class GepaOptimizer:
    """PromptOptimizer implementation backed by dspy's GEPA."""

    name = "gepa"

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
        dspy.configure(lm=task_lm)
        signature = dspy.Signature("input -> output", instructions=base_instructions)
        student = dspy.Predict(signature)

        # Call run_gepa through evalloop.optimize -- its historical home -- so
        # existing tests that monkeypatch `optimize_mod.run_gepa` keep
        # intercepting the call that is actually executed. Imported lazily to
        # avoid a circular import (evalloop.optimize imports this module).
        from evalloop import optimize as optimize_mod

        optimized_program = optimize_mod.run_gepa(student, trainset, metric, reflection_lm, cfg.optimize.auto)
        return OptimizeResult(
            optimized_instructions=optimized_program.signature.instructions,
            method=self.name,
            extra_log={},
        )
