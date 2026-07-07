"""COPRO prompt optimizer: dspy.teleprompt.COPRO behind the PromptOptimizer
contract. Coordinate-ascent style instruction refinement; instruction-only by
nature (COPRO does not bootstrap demos).

Confirmed against the installed dspy==3.2.1 API (`inspect.signature`):
    COPRO(prompt_model=None, metric=None, breadth=10, depth=3,
          init_temperature=1.4, track_stats=False, **kwargs)
    COPRO.compile(student, *, trainset, eval_kwargs)
    dspy.evaluate.Evaluate(*, devset, metric, ..., display_progress, display_table, ...)

NOTE: unlike GEPA/MIPROv2, this dspy version's COPRO takes NO seed parameter
(the issue's params example mentioned one; the actual pinned signature wins).
Supported params: breadth, depth, init_temperature.

COPRO evaluates candidates on the given trainset internally (no explicit
valset), so the whole TRAIN split is passed through -- the test split never
reaches the optimizer; iron rule #1 is re-asserted upstream in optimize().
The metric goes through the same scalar adaptation as MIPROv2 (COPRO averages
plain numbers via dspy.Evaluate; GEPA-style Prediction(score, feedback) would
not sum).
"""

from __future__ import annotations

from collections.abc import Callable

import dspy

from evalloop.optimizers.base import OptimizeResult
from evalloop.optimizers.miprov2 import _scalar_metric
from evalloop.schemas import Config


def run_copro(student, trainset, metric, prompt_model, breadth: int, depth: int, init_temperature: float):
    """Thin, monkeypatchable wrapper around the real dspy.teleprompt.COPRO
    call, mirroring run_gepa()/run_miprov2() so orchestration can be
    unit-tested without real API calls.
    """
    from dspy.teleprompt import COPRO

    optimizer = COPRO(
        prompt_model=prompt_model,  # instruction proposal -- the reflection role
        metric=metric,
        breadth=breadth,
        depth=depth,
        init_temperature=init_temperature,
    )
    return optimizer.compile(
        student,
        trainset=trainset,
        # keep candidate evaluation quiet in CI / non-tty runs
        eval_kwargs={"display_progress": False, "display_table": False},
    )


class CoproOptimizer:
    """PromptOptimizer implementation backed by dspy's COPRO (instruction-only)."""

    name = "copro"

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
        breadth = int(params.get("breadth", 10))
        depth = int(params.get("depth", 3))
        init_temperature = float(params.get("init_temperature", 1.4))

        dspy.configure(lm=task_lm)
        signature = dspy.Signature("input -> output", instructions=base_instructions)
        student = dspy.Predict(signature)

        # Call run_copro through evalloop.optimize -- the same monkeypatch
        # convention as run_gepa/run_miprov2. Imported lazily to avoid a
        # circular import.
        from evalloop import optimize as optimize_mod

        optimized_program = optimize_mod.run_copro(
            student,
            trainset,
            _scalar_metric(metric),
            reflection_lm,
            breadth,
            depth,
            init_temperature,
        )
        return OptimizeResult(
            optimized_instructions=optimized_program.signature.instructions,
            method=self.name,
            extra_log={
                # effective values actually used, for optimize_log.json
                "breadth": breadth,
                "depth": depth,
                "init_temperature": init_temperature,
                "train_size": len(trainset),
            },
        )
