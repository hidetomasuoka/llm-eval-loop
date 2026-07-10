"""Training-case subset schedulers for prompt optimization.

APO methods often spend most of their budget evaluating candidate prompts over
training examples.  The default scheduler keeps the historical behavior
(full-train evaluation), while budgeted schedulers provide a deterministic
place to experiment with prompt-aware / few-shot evaluation schedules without
letting optimizers see the test split.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from evalloop.optimizers.base import OptimizeError
from evalloop.schemas import GoldenCase


def _budget_or_all(case_count: int, budget: int | None) -> int:
    if budget is None:
        return case_count
    if budget <= 0:
        raise OptimizeError(f"optimize.params.eval_budget must be positive when set, got {budget!r}")
    return min(case_count, budget)


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = " ".join(text.split())
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def select_eval_subset(
    cases: Sequence[GoldenCase],
    *,
    strategy: str = "full",
    budget: int | None = None,
    seed: int = 0,
) -> list[GoldenCase]:
    """Return the train cases an optimizer should evaluate candidates on.

    Strategies:
    - ``full``: historical behavior; ignore ``budget`` and return all cases.
    - ``random``: deterministic random subset, preserving original order.
    - ``coverage``: greedy diversity over character n-grams of ``input``;
      useful as a lightweight local proxy before a full POES implementation.
    """
    items = list(cases)
    if not items:
        return []

    if strategy == "full":
        return items

    k = _budget_or_all(len(items), budget)
    if k == len(items):
        return items

    if strategy == "random":
        selected_idx = set(random.Random(seed).sample(range(len(items)), k))
        return [case for i, case in enumerate(items) if i in selected_idx]

    if strategy == "coverage":
        features = [_char_ngrams(case.input) for case in items]
        # Stable first pick: the longest input tends to carry the most surface
        # features; ties fall back to the original order.
        first = max(range(len(items)), key=lambda i: (len(features[i]), -i))
        selected = [first]
        remaining = set(range(len(items))) - {first}
        while len(selected) < k:
            best = max(
                remaining,
                key=lambda i: (
                    min(1.0 - _jaccard(features[i], features[j]) for j in selected),
                    len(features[i]),
                    -i,
                ),
            )
            selected.append(best)
            remaining.remove(best)
        selected_set = set(selected)
        return [case for i, case in enumerate(items) if i in selected_set]

    raise OptimizeError(
        "optimize.params.eval_scheduler must be one of ['full', 'random', 'coverage'], "
        f"got {strategy!r}"
    )
