"""Shared contract for prompt-optimization methods (APO).

Every method module in this package (gepa.py today; [APO-06]/[APO-07] later)
implements the PromptOptimizer protocol and returns an OptimizeResult, so
evalloop.optimize can orchestrate optimizer selection -> execution -> variant
generation -> run/report/compare without knowing method internals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from evalloop.schemas import Config


class OptimizeError(RuntimeError):
    pass


@dataclass
class OptimizeResult:
    optimized_instructions: str
    method: str
    extra_log: dict  # method-specific log (iteration counts etc.), merged into optimize_log.json


class PromptOptimizer(Protocol):
    name: str

    def optimize(
        self,
        *,
        base_instructions: str,
        trainset: list,
        metric: Callable,
        task_lm,
        reflection_lm,
        cfg: Config,
    ) -> OptimizeResult: ...
