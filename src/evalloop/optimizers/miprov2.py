"""MIPROv2 prompt optimizer: dspy.teleprompt.MIPROv2 behind the PromptOptimizer
contract. The default remains instruction-only for backward compatibility,
but params.max_bootstrapped_demos / max_labeled_demos can enable few-shot demo search.

Confirmed against the installed dspy==3.2.1 API (`inspect.signature`):
    MIPROv2(metric, prompt_model=None, task_model=None, ...,
            max_bootstrapped_demos=4, max_labeled_demos=4,
            auto='light'|'medium'|'heavy', seed=9, ...)
    MIPROv2.compile(student, *, trainset, valset=None, seed=None,
                    requires_permission_to_run=None, ...)

Differences from GEPA that this module absorbs:
    - the metric: GEPA consumes dspy.Prediction(score=, feedback=); MIPROv2
      averages plain numbers, so the orchestrator's metric is adapted via
      _scalar_metric (feedback text is simply unused by MIPROv2)
    - the validation set: GEPA splits internally; MIPROv2 wants an explicit
      valset, carved 8:2 out of the TRAIN split with a fixed seed
      (params.val_ratio / params.seed). Optional demo counts are passed through
      from params.max_bootstrapped_demos / max_labeled_demos. The test split is never touched --
      iron rule #1 is re-asserted upstream in optimize().
"""

from __future__ import annotations

import random
from collections.abc import Callable

import dspy

from evalloop.demos import demos_from_dspy_program
from evalloop.optimizers.base import OptimizeError, OptimizeResult
from evalloop.optimizers.metrics import compute_train_score
from evalloop.schemas import Config

# Popped by optimize() before writing optimize_log.json (not a log field).
OPTIMIZED_DEMOS_LOG_KEY = "_optimized_demos"


def _scalar_metric(metric: Callable) -> Callable:
    """Adapt the orchestrator's GEPA-style metric (returns
    dspy.Prediction(score=, feedback=)) to MIPROv2's plain-number contract."""

    def wrapped(gold, pred, trace=None):
        result = metric(gold, pred, trace)
        return getattr(result, "score", result)

    return wrapped


def split_train_val(trainset: list, val_ratio: float, seed: int) -> tuple[list, list]:
    """Deterministically carve a validation set out of the train split.
    Never sees the test split: optimize() only ever passes train cases here.
    """
    if not 0.0 < val_ratio < 1.0:
        raise OptimizeError(f"optimize.params.val_ratio must be between 0 and 1 (exclusive), got {val_ratio!r}")
    if len(trainset) < 2:
        raise OptimizeError(
            f"miprov2 needs at least 2 train cases to carve out a validation set, got {len(trainset)}"
        )
    indices = list(range(len(trainset)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, round(len(trainset) * val_ratio))
    if val_count >= len(trainset):
        val_count = len(trainset) - 1  # keep at least one training example
    val_idx = set(indices[:val_count])
    train_part = [ex for i, ex in enumerate(trainset) if i not in val_idx]
    val_part = [ex for i, ex in enumerate(trainset) if i in val_idx]
    return train_part, val_part


def run_miprov2(
    student,
    trainset,
    valset,
    metric,
    prompt_model,
    task_model,
    auto: str,
    seed: int,
    max_bootstrapped_demos: int = 0,
    max_labeled_demos: int = 0,
):
    """Thin, monkeypatchable wrapper around the real dspy.teleprompt.MIPROv2
    call, mirroring run_gepa() so orchestration can be unit-tested without
    real API calls.
    """
    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(
        metric=metric,
        prompt_model=prompt_model,  # instruction proposal -- the reflection role
        task_model=task_model,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        auto=auto,
        seed=seed,
    )
    return optimizer.compile(
        student,
        trainset=trainset,
        valset=valset,
        # never prompt interactively (CI / non-tty runs)
        requires_permission_to_run=False,
    )


class MiproV2Optimizer:
    """PromptOptimizer implementation backed by dspy's MIPROv2."""

    name = "miprov2"

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
        val_ratio = float(params.get("val_ratio", 0.2))
        seed = int(params.get("seed", 0))
        max_bootstrapped_demos = int(params.get("max_bootstrapped_demos", 0))
        max_labeled_demos = int(params.get("max_labeled_demos", 0))
        if max_bootstrapped_demos < 0 or max_labeled_demos < 0:
            raise OptimizeError("miprov2 demo counts must be non-negative")

        dspy.configure(lm=task_lm)
        signature = dspy.Signature("input -> output", instructions=base_instructions)
        student = dspy.Predict(signature)

        train_part, val_part = split_train_val(trainset, val_ratio, seed)

        # Call run_miprov2 through evalloop.optimize -- the same monkeypatch
        # convention as run_gepa -- so tests patch one well-known location.
        # Imported lazily to avoid a circular import.
        from evalloop import optimize as optimize_mod

        optimized_program = optimize_mod.run_miprov2(
            student,
            train_part,
            val_part,
            _scalar_metric(metric),
            reflection_lm,
            task_lm,
            cfg.optimize.auto,
            seed,
            max_bootstrapped_demos,
            max_labeled_demos,
        )
        extra_log = {
            # effective values actually used, for optimize_log.json
            "val_ratio": val_ratio,
            "seed": seed,
            "train_size": len(train_part),
            "val_size": len(val_part),
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
        }
        # train_score must match train_size (train_part only; exclude val_part).
        train_score = compute_train_score(train_part, metric, optimized_program)
        if train_score is not None:
            extra_log["train_score"] = train_score

        # APO-17: when demo search is enabled, hand extracted demos to optimize()
        # for demos.jsonl + {{demos}} re-injection into the variant prompt.
        if max_bootstrapped_demos > 0 or max_labeled_demos > 0:
            train_input_to_id = {
                str(ex.input): str(getattr(ex, "case_id", None) or ex.input) for ex in trainset
            }
            try:
                extracted = demos_from_dspy_program(
                    optimized_program, train_input_to_id=train_input_to_id
                )
            except Exception as e:
                raise OptimizeError(str(e)) from e
            extra_log[OPTIMIZED_DEMOS_LOG_KEY] = [
                {
                    "input": demo.input,
                    "output": demo.output,
                    "id": demo.id,
                    "origin": origin,
                }
                for demo, origin in extracted
            ]
            extra_log["demo_count"] = len(extracted)

        return OptimizeResult(
            optimized_instructions=optimized_program.signature.instructions,
            method=self.name,
            extra_log=extra_log,
        )
