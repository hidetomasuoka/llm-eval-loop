import pytest

from evalloop import stats as stats_mod
from evalloop.schemas import CaseResult


def _cr(alias, passed, case_id="case-0001", repeat_index=0):
    return CaseResult(
        case_id=case_id,
        alias=alias,
        provider_id=alias,
        expected="契約照会",
        category="基本",
        output="契約照会",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="ok" if passed else "mismatch",
        cost=0.001,
        latency_ms=100,
        cached=False,
        token_usage={},
        error=None,
        repeat_index=repeat_index,
        raw={},
    )


# --- mcnemar_exact_p -------------------------------------------------------


def test_mcnemar_exact_p_known_value_from_improvement_plan():
    # docs/improvement-plan-gepa-cuad100.md: 遷移表 b=12, c=4 なら p≈0.077
    p = stats_mod.mcnemar_exact_p(12, 4)
    assert p == pytest.approx(0.0768, abs=0.0005)


def test_mcnemar_exact_p_symmetric():
    assert stats_mod.mcnemar_exact_p(12, 4) == stats_mod.mcnemar_exact_p(4, 12)


def test_mcnemar_exact_p_balanced_table_is_not_significant():
    # b == c is the least extreme outcome; two-sided p caps at 1.0
    assert stats_mod.mcnemar_exact_p(5, 5) == 1.0


def test_mcnemar_exact_p_one_sided_extreme():
    # 8 flips all in one direction: p = 2 * (1/2)^8 = 0.0078125
    assert stats_mod.mcnemar_exact_p(8, 0) == pytest.approx(2 / 256)


def test_mcnemar_exact_p_no_discordant_pairs_is_undefined():
    assert stats_mod.mcnemar_exact_p(0, 0) is None


def test_mcnemar_exact_p_rejects_negative_counts():
    with pytest.raises(ValueError):
        stats_mod.mcnemar_exact_p(-1, 3)


# --- paired_transition -----------------------------------------------------


def test_paired_transition_counts_b_and_c():
    results_a = [
        _cr("m", False, case_id="case-0001"),  # -> pass in B: b
        _cr("m", True, case_id="case-0002"),  # -> fail in B: c
        _cr("m", True, case_id="case-0003"),  # pass in both: concordant
        _cr("m", False, case_id="case-0004"),  # fail in both: concordant
    ]
    results_b = [
        _cr("m", True, case_id="case-0001"),
        _cr("m", False, case_id="case-0002"),
        _cr("m", True, case_id="case-0003"),
        _cr("m", False, case_id="case-0004"),
    ]
    t = stats_mod.paired_transition(results_a, results_b, "m")
    assert (t.n_paired, t.b, t.c) == (4, 1, 1)
    assert t.p_value == 1.0


def test_paired_transition_excludes_unpaired_cases_and_other_aliases():
    results_a = [
        _cr("m", True, case_id="case-0001"),
        _cr("m", True, case_id="case-only-in-a"),
        _cr("other", False, case_id="case-0001"),  # different alias: ignored
    ]
    results_b = [
        _cr("m", False, case_id="case-0001"),
        _cr("m", True, case_id="case-only-in-b"),
    ]
    t = stats_mod.paired_transition(results_a, results_b, "m")
    assert (t.n_paired, t.b, t.c) == (1, 0, 1)


def test_paired_transition_majority_vote_over_repeats_tie_is_fail():
    # case-0001 in A: pass/fail tie across 2 repeats -> conservative fail
    results_a = [
        _cr("m", True, case_id="case-0001", repeat_index=0),
        _cr("m", False, case_id="case-0001", repeat_index=1),
    ]
    # in B: 2/3 pass -> majority pass
    results_b = [
        _cr("m", True, case_id="case-0001", repeat_index=0),
        _cr("m", True, case_id="case-0001", repeat_index=1),
        _cr("m", False, case_id="case-0001", repeat_index=2),
    ]
    t = stats_mod.paired_transition(results_a, results_b, "m")
    assert (t.n_paired, t.b, t.c) == (1, 1, 0)


def test_paired_transition_skips_ungraded_results():
    results_a = [_cr("m", None, case_id="case-0001")]
    results_b = [_cr("m", True, case_id="case-0001")]
    t = stats_mod.paired_transition(results_a, results_b, "m")
    assert t.n_paired == 0
    assert t.p_value is None
