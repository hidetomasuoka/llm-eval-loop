import pytest

from evalloop.optimizers.base import OptimizeError
from evalloop.optimizers.schedulers import select_eval_subset
from evalloop.schemas import GoldenCase


def _cases(n=6):
    return [
        GoldenCase(
            id=f"c{i}",
            input=f"text {i} {'x' * i}",
            expected="ok",
            split="train",
            category="基本",
            difficulty=None,
            source="self-made",
        )
        for i in range(n)
    ]


def test_full_scheduler_returns_all_cases_even_with_budget():
    cases = _cases(4)
    assert select_eval_subset(cases, strategy="full", budget=1) == cases


def test_random_scheduler_is_deterministic_and_preserves_original_order():
    cases = _cases(8)
    a = select_eval_subset(cases, strategy="random", budget=3, seed=7)
    b = select_eval_subset(cases, strategy="random", budget=3, seed=7)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.id for c in a] == sorted([c.id for c in a], key=lambda cid: int(cid[1:]))
    assert len(a) == 3


def test_coverage_scheduler_selects_budgeted_diverse_subset():
    cases = [
        GoldenCase(
            id="short",
            input="aaa",
            expected="ok",
            split="train",
            category="基本",
            difficulty=None,
            source="self-made",
        ),
        GoldenCase(
            id="similar",
            input="aaaa",
            expected="ok",
            split="train",
            category="基本",
            difficulty=None,
            source="self-made",
        ),
        GoldenCase(
            id="different",
            input="zzzzzz",
            expected="ok",
            split="train",
            category="基本",
            difficulty=None,
            source="self-made",
        ),
    ]
    selected = select_eval_subset(cases, strategy="coverage", budget=2)
    assert [c.id for c in selected] == ["similar", "different"]


def test_budget_larger_than_cases_returns_all_for_budgeted_schedulers():
    cases = _cases(3)
    assert select_eval_subset(cases, strategy="random", budget=10, seed=0) == cases


def test_scheduler_rejects_bad_budget_and_strategy():
    cases = _cases(3)
    with pytest.raises(OptimizeError, match="eval_budget"):
        select_eval_subset(cases, strategy="random", budget=0)
    with pytest.raises(OptimizeError, match="eval_scheduler"):
        select_eval_subset(cases, strategy="nope", budget=1)
