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
from evalloop.optimizers.metrics import compute_train_score
from evalloop.optimizers.miprov2 import split_train_val
from evalloop.schemas import Config


def run_gepa(student, trainset, metric, reflection_lm, auto: str, seed: int = 0, valset=None):
    """Thin, monkeypatchable wrapper around the real dspy.teleprompt.GEPA call
    so orchestration logic (file writing, variant config, run/report/compare)
    can be unit-tested without spending real API calls.
    """
    from dspy.teleprompt import GEPA

    optimizer = GEPA(metric=metric, reflection_lm=reflection_lm, auto=auto, seed=seed)
    return optimizer.compile(student=student, trainset=trainset, valset=valset)


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

        # Improvement plan #6: without a valset, GEPA selects candidates by
        # their score on the very cases they were trained on -- structural
        # overfitting ("best on train, worse on holdout" was the observed
        # outcome of the first cuad100 runs). Carve val out of TRAIN with the
        # same fixed-seed split miprov2 uses; test cases are never touched.
        seed = int(cfg.optimize.params.get("seed", 0) or 0)
        if len(trainset) >= 2:
            val_ratio = float(cfg.optimize.params.get("val_ratio", 0.2))
            train_part, val_part = split_train_val(trainset, val_ratio, seed)
        else:
            print("[optimize] WARN: trainset has <2 cases; GEPA runs without a valset (candidate selection on train)")
            val_ratio = None
            train_part, val_part = list(trainset), None

        # Call run_gepa through evalloop.optimize -- its historical home -- so
        # existing tests that monkeypatch `optimize_mod.run_gepa` keep
        # intercepting the call that is actually executed. Imported lazily to
        # avoid a circular import (evalloop.optimize imports this module).
        from evalloop import optimize as optimize_mod

        optimized_program = optimize_mod.run_gepa(
            student,
            train_part,
            metric,
            reflection_lm,
            cfg.optimize.auto,
            seed=seed,
            valset=val_part,
        )
        extra_log: dict = {
            "train_size": len(train_part),
            "val_size": len(val_part) if val_part is not None else 0,
        }
        if val_ratio is not None:
            extra_log["val_ratio"] = val_ratio
        # train_score covers train_part only (val_part is selection data, not
        # training data -- same accounting as miprov2)
        train_score = compute_train_score(train_part, metric, optimized_program)
        if train_score is not None:
            extra_log["train_score"] = train_score
        return OptimizeResult(
            optimized_instructions=optimized_program.signature.instructions,
            method=self.name,
            extra_log=extra_log,
        )
