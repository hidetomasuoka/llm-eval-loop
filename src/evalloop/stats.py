"""Paired significance statistics for run comparison.

compare's original significance check ("beyond_95ci": non-overlapping Wilson
intervals) treats the two runs as independent samples, but a compare between
runs of the same task evaluates the SAME case set -- a paired design. The
McNemar exact test uses the per-case pass/fail transition table (b = cases
that flipped fail->pass, c = pass->fail) and therefore detects smaller real
deltas on the same n. Exact binomial via math.comb; no scipy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from evalloop.schemas import CaseResult


def mcnemar_exact_p(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value from discordant-pair counts.

    b = cases failing in A but passing in B, c = the reverse. Under H0 each
    discordant pair is a fair coin flip, so p = P(min tail) doubled, capped
    at 1.0. Returns None when there are no discordant pairs (the test is
    undefined -- the runs agree on every case).
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b} c={c}")
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass
class PairedTransition:
    """Per-case pass/fail transition between two runs for one alias."""

    n_paired: int  # cases with a verdict in BOTH runs
    b: int  # fail in A -> pass in B (improved)
    c: int  # pass in A -> fail in B (regressed)

    @property
    def p_value(self) -> float | None:
        return mcnemar_exact_p(self.b, self.c)


def _case_verdicts(results: list[CaseResult], alias: str) -> dict[str, bool]:
    """One boolean verdict per case_id for the alias.

    repeat>1 runs get a majority vote per case; a tie counts as fail (the
    conservative reading -- an unstable case is not a dependable pass).
    Ungraded results (passed is None) and rows without case_id are skipped.
    """
    votes: dict[str, list[bool]] = {}
    for r in results:
        if r.alias != alias or r.case_id is None or r.passed is None:
            continue
        votes.setdefault(r.case_id, []).append(bool(r.passed))
    return {case_id: sum(flags) * 2 > len(flags) for case_id, flags in votes.items()}


def paired_transition(results_a: list[CaseResult], results_b: list[CaseResult], alias: str) -> PairedTransition:
    """Build the McNemar transition table for one alias across two runs.

    Cases graded in only one of the runs (e.g. a --limit run) are excluded;
    only the intersection is a paired observation.
    """
    verdicts_a = _case_verdicts(results_a, alias)
    verdicts_b = _case_verdicts(results_b, alias)
    common = verdicts_a.keys() & verdicts_b.keys()
    b = sum(1 for cid in common if not verdicts_a[cid] and verdicts_b[cid])
    c = sum(1 for cid in common if verdicts_a[cid] and not verdicts_b[cid])
    return PairedTransition(n_paired=len(common), b=b, c=c)
