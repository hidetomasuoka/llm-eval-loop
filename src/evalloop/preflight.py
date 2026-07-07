"""Preflight checks for `evalloop optimize` (APO-09).

APO stands on top of the evaluation set (APO_GUIDE.md ops rule #1): with too
little or too skewed train data, the optimizer's candidate scores are noise
and the "improvement" it reports is overfitting. These checks run before any
LM is constructed or a single rollout is spent.

Threshold rationale (heuristics, deliberately simple -- the right size is
評価セット依存: it varies with task difficulty and label cardinality):

- MIN_TRAIN_CASES = 10 (error): below ~10 cases, one flipped case moves the
  train score by >=10pt, so candidate instructions are indistinguishable from
  noise; MIPROv2's train/val split would also leave a validation side of 1-2
  cases. Demotable to a warning with --force for deliberate tiny-data
  experiments.
- RECOMMENDED_TRAIN_CASES = 30 (warning): between 10 and ~30 cases the
  optimizer can still fit quirks of individual cases (過学習リスク) -- fine
  for exploration, but treat train-score gains as provisional until the
  holdout confirms them.
- MIN_CASES_PER_LABEL = 2 (error, label tasks): a label absent from train can
  never be learned, and a label with a single example gives the optimizer one
  data point to "generalize" from. Demotable with --force.
"""

from __future__ import annotations

from collections import Counter

from evalloop.optimizers.base import OptimizeError
from evalloop.schemas import Config, GoldenCase

MIN_TRAIN_CASES = 10
RECOMMENDED_TRAIN_CASES = 30
MIN_CASES_PER_LABEL = 2


def preflight_optimize(
    cfg: Config, train_cases: list[GoldenCase], test_case_count: int, force: bool = False
) -> list[str]:
    """Validate the train/holdout data before spending optimizer rollouts.

    Returns display-ready warning strings; raises OptimizeError on failures
    (or demotes them to warnings when force=True).
    """
    errors: list[str] = []
    warnings: list[str] = []

    n_train = len(train_cases)
    if n_train < MIN_TRAIN_CASES:
        errors.append(
            f"train split has only {n_train} case(s) (< {MIN_TRAIN_CASES}): candidate scores on so few "
            "cases are dominated by single-case flips, so the optimizer cannot tell instructions apart"
        )

    if cfg.task.answer_type == "label":
        train_label_counts = Counter(c.expected for c in train_cases if isinstance(c.expected, str))
        missing = [label for label in cfg.task.labels if train_label_counts.get(label, 0) == 0]
        sparse = [
            label for label in cfg.task.labels if 0 < train_label_counts.get(label, 0) < MIN_CASES_PER_LABEL
        ]
        if missing:
            errors.append(
                f"label(s) {missing} appear in task.yaml labels but have no train case at all -- "
                "the optimizer can never learn them"
            )
        if sparse:
            errors.append(
                f"label(s) {sparse} have fewer than {MIN_CASES_PER_LABEL} train case(s) -- one example "
                "is a single data point to 'generalize' from"
            )

    if test_case_count == 0:
        warnings.append(
            "holdout (test) split is empty: train-score gains cannot be checked for generalization "
            "(APO_GUIDE.md ops rule #4) -- add split=='test' cases before trusting any improvement"
        )
    if MIN_TRAIN_CASES <= n_train < RECOMMENDED_TRAIN_CASES:
        warnings.append(
            f"train split has {n_train} case(s) (< {RECOMMENDED_TRAIN_CASES}): overfitting risk -- "
            "treat train-score gains as provisional until the holdout confirms them"
        )

    if errors:
        if force:
            warnings = [f"(--force) demoted from error: {e}" for e in errors] + warnings
        else:
            details = "\n".join(f"  - {e}" for e in errors)
            raise OptimizeError(
                f"optimize preflight failed for task {cfg.task.name!r}:\n{details}\n"
                "Fix the train split (preferred), or re-run with --force to demote these to warnings "
                "for a deliberate small-data experiment."
            )
    return warnings
