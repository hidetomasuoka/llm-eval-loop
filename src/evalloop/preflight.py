"""Pre-flight checks for `evalloop optimize` (APO-09, issue #68).

Runs before any LM call to catch evaluation-design problems early:

- **error** (raises OptimizeError, aborts unless --force): train < 10 cases;
  label tasks with a task.yaml label never seen in train, or seen only once
- **warn** (prints, continues): test/holdout split empty (can't verify
  generalization); train < 30 (overfitting risk); answer_type=json priority
  reminder (Structured Outputs / schema before prompt optimization; APO-18)

The thresholds are constants with docstrings explaining their root cause
(eval-set dependence, not theory). They are deliberately conservative: a
small train set can still optimize successfully, but the risk of a
non-representative sample is high enough to surface to the user.

This module never calls a model provider and never reads the test split's
content (only its count) -- it is a pure data sanity check, safe to run
before `assert_split_disjoint`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from evalloop.demos import DEMOS_PLACEHOLDER
from evalloop.optimizers.base import OptimizeError
from evalloop.optimizers.metrics import normalize_label
from evalloop.schemas import GoldenCase

if TYPE_CHECKING:
    from evalloop.schemas import Config

# --- thresholds (constants; see module docstring for rationale) -------------

# A train split under this many cases cannot reliably represent the label
# space or the input distribution. 10 is a floor, not a recommendation -- a
# real eval set should be much larger. Lowering this is allowed via --force.
MIN_TRAIN_CASES = 10

# Below this train size, overfitting to the train set is a real risk even with
# a held-out test split. Warn (don't block) so the user can make an informed
# call about whether to proceed.
SMALL_TRAIN_WARN = 30

# A label seen only once in train gives the optimizer a single positive (or
# negative) example -- not enough to learn a decision boundary. This is an
# error for label tasks because the optimizer will likely overfit to that one
# case. Override via --force only if you know what you're doing.
MIN_LABEL_OCCURRENCES = 2


@dataclass
class PreflightResult:
    """Outcome of running preflight checks.

    `errors` aborts optimization unless `force=True` was passed (which
    demotes them to warnings and appends them to `warnings`). `warnings`
    are always shown but never abort.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings don't block)."""
        return not self.errors


def run_preflight(
    cfg: "Config",
    train_cases: list[GoldenCase],
    test_count: int,
    *,
    force: bool = False,
) -> PreflightResult:
    """Run all preflight checks and return the aggregated result.

    Args:
        cfg: the resolved task Config (used for task.answer_type and task.labels).
        train_cases: the split=='train' GoldenCase list (already filtered).
        test_count: number of split=='test' cases (holdout). Only the count is
            needed, never the content, so this stays a pure data check.
        force: when True, errors are demoted to warnings (printed, not raised).

    Returns:
        PreflightResult with errors/warnings populated. Caller is responsible
        for raising OptimizeError when result.ok is False and force is False.
    """
    result = PreflightResult()

    # --- error: train too small ------------------------------------------------
    if len(train_cases) < MIN_TRAIN_CASES:
        msg = (
            f"train split has only {len(train_cases)} case(s); need at least {MIN_TRAIN_CASES} "
            "for a representative sample (lower the floor via --force only if you accept the risk)"
        )
        result.errors.append(msg)

    # --- error (label tasks only): label coverage ------------------------------
    # task.labels is non-empty iff answer_type == "label" (enforced by TaskConfig.__post_init__)
    if cfg.task.labels:
        # count AND compare in the training metric's normalized space (strip
        # quotes/trailing punctuation, full-width -> half-width): normalizing
        # only the train side would misreport a task.yaml spelling variant as
        # unseen/singleton. Error messages keep the task.yaml spelling so the
        # user can find the label they actually wrote.
        norm_by_label = {label: normalize_label(label) for label in cfg.task.labels}
        seen_counts: dict[str, int] = {}
        for c in train_cases:
            norm = normalize_label(str(c.expected))
            seen_counts[norm] = seen_counts.get(norm, 0) + 1

        # a task.yaml label that never appears in train
        unseen = sorted(label for label, norm in norm_by_label.items() if norm not in seen_counts)
        for label in unseen:
            result.errors.append(
                f"task.yaml label {label!r} never appears in the train split; the optimizer "
                "cannot learn to predict it"
            )

        # a label seen only once
        singletons = sorted(
            label
            for label, norm in norm_by_label.items()
            if 0 < seen_counts.get(norm, 0) < MIN_LABEL_OCCURRENCES
        )
        for label in singletons:
            result.errors.append(
                f"label {label!r} appears only {seen_counts[norm_by_label[label]]} time(s) in train; "
                f"need at least {MIN_LABEL_OCCURRENCES} for the optimizer to learn a boundary "
                "(--force to override)"
            )

    # --- error: miprov2 demo search requires {{demos}} placeholder (APO-17) -----
    if cfg.optimize.method == "miprov2":
        boot = int(cfg.optimize.params.get("max_bootstrapped_demos", 0) or 0)
        labeled = int(cfg.optimize.params.get("max_labeled_demos", 0) or 0)
        if boot > 0 or labeled > 0:
            # prompt_file is absolute after schemas.load_task()
            prompt_path = Path(cfg.task.prompt_file)
            try:
                template = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""
            except OSError:
                template = ""
            if DEMOS_PLACEHOLDER not in template:
                result.errors.append(
                    f"miprov2 demo search is enabled (max_bootstrapped_demos={boot}, "
                    f"max_labeled_demos={labeled}) but {cfg.task.prompt_file} has no "
                    f"{DEMOS_PLACEHOLDER} placeholder; add it or set both demo counts to 0"
                )

    # --- warning: no holdout ----------------------------------------------------
    if test_count == 0:
        result.warnings.append(
            "test split (holdout) is empty; generalization cannot be verified on unseen data"
        )

    # --- warning: small train --------------------------------------------------
    if len(train_cases) < SMALL_TRAIN_WARN:
        result.warnings.append(
            f"train split has only {len(train_cases)} case(s); overfitting risk is high "
            f"(below the {SMALL_TRAIN_WARN}-case warning threshold)"
        )

    # --- warning (json tasks): prompt optimization is not the first lever ------
    # APO-18 / docs/APO_GUIDE.md: structural JSON failures should be fixed with
    # Structured Outputs / schema redesign before spending optimize budget on
    # the prompt. Content-accuracy issues remain a valid optimize use case, so
    # this is warn-only and never aborts.
    if cfg.task.answer_type == "json":
        result.warnings.append(
            "answer_type=json: if the main failure mode is JSON syntax/schema errors, "
            "prefer (1) the provider's Structured Outputs / tool calling and "
            "(2) schema redesign before prompt optimization "
            "(see docs/APO_GUIDE.md). Prompt optimization is for content-accuracy gains."
        )

    # --- force: demote errors to warnings --------------------------------------
    if force and result.errors:
        result.warnings.extend(f"[forced] {e}" for e in result.errors)
        result.errors = []

    return result


def format_preflight(result: PreflightResult) -> list[str]:
    """Format a PreflightResult into rich-console-styled lines for display."""
    lines: list[str] = []
    for e in result.errors:
        lines.append(f"[bold red]preflight ERROR:[/bold red] {e}")
    for w in result.warnings:
        lines.append(f"[yellow]preflight WARN:[/yellow] {w}")
    return lines


def check_or_raise(result: PreflightResult, *, force: bool = False) -> None:
    """Raise OptimizeError if the result has errors and force is False.

    This is the single chokepoint that decides abort-vs-continue. Callers
    that already demoted errors via run_preflight(force=True) get an empty
    errors list and pass through cleanly.
    """
    if result.errors and not force:
        joined = "\n".join(f"  - {e}" for e in result.errors)
        raise OptimizeError(
            f"preflight failed with {len(result.errors)} error(s):\n{joined}\n"
            "fix the evaluation set, or re-run with --force to demote errors to warnings"
        )