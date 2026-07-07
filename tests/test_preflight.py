"""Optimize preflight checks (APO-09): each error/warning condition in
isolation, --force demotion, and the acceptance criterion that the real
tracked sample-inquiry task passes.
"""

import pytest

from evalloop.optimizers.base import OptimizeError
from evalloop.paths import REPO_ROOT
from evalloop.preflight import (
    MIN_TRAIN_CASES,
    RECOMMENDED_TRAIN_CASES,
    preflight_optimize,
)
from evalloop.schemas import GoldenCase, load_golden_jsonl, load_task
from tests.conftest import DEFAULT_LABELS, scaffold_task


def _cases(label_counts: dict[str, int]) -> list[GoldenCase]:
    """Train cases with an exact per-label distribution."""
    cases = []
    for label, count in label_counts.items():
        for i in range(count):
            n = len(cases) + 1
            cases.append(
                GoldenCase(
                    id=f"case-{n:04d}",
                    input=f"問い合わせ文サンプル{n}",
                    expected=label,
                    split="train",
                    category="基本",
                    difficulty=None,
                    source="self-made",
                )
            )
    return cases


def _balanced(per_label: int) -> list[GoldenCase]:
    return _cases({label: per_label for label in DEFAULT_LABELS})


@pytest.fixture
def label_cfg(isolated_root):
    cfg, _paths = scaffold_task(isolated_root)
    return cfg


# ---------------------------------------------------------------------------
# errors (and their --force demotion)
# ---------------------------------------------------------------------------


def test_train_below_minimum_raises(label_cfg):
    train = _balanced(2)  # 8 cases: every label has >=2, so ONLY the size check fires
    assert len(train) < MIN_TRAIN_CASES
    with pytest.raises(OptimizeError, match=str(MIN_TRAIN_CASES)):
        preflight_optimize(label_cfg, train, test_case_count=4)


def test_force_demotes_train_minimum_to_warning(label_cfg):
    warnings = preflight_optimize(label_cfg, _balanced(2), test_case_count=4, force=True)
    assert any(w.startswith("(--force) demoted from error:") for w in warnings)


def test_label_missing_from_train_raises(label_cfg):
    train = _cases({"契約照会": 4, "障害報告": 4, "機能要望": 4})  # 12 cases, その他 absent
    with pytest.raises(OptimizeError, match="その他"):
        preflight_optimize(label_cfg, train, test_case_count=4)


def test_label_with_single_train_case_raises(label_cfg):
    train = _cases({"契約照会": 5, "障害報告": 4, "機能要望": 2, "その他": 1})  # 12 cases
    with pytest.raises(OptimizeError, match="その他"):
        preflight_optimize(label_cfg, train, test_case_count=4)


def test_force_demotes_label_errors_to_warnings(label_cfg):
    train = _cases({"契約照会": 6, "障害報告": 5, "機能要望": 1})  # missing AND sparse labels
    warnings = preflight_optimize(label_cfg, train, test_case_count=4, force=True)
    demoted = [w for w in warnings if w.startswith("(--force) demoted from error:")]
    assert any("その他" in w for w in demoted)  # missing
    assert any("機能要望" in w for w in demoted)  # single case


def test_text_task_skips_label_checks(isolated_root):
    cfg, _paths = scaffold_task(isolated_root, answer_type="text")
    train = _cases({"span A": 6, "span B": 6})  # arbitrary expected strings
    warnings = preflight_optimize(cfg, train, test_case_count=4)
    assert all("label" not in w for w in warnings)  # only the <30 size warning may fire


# ---------------------------------------------------------------------------
# warnings (never block)
# ---------------------------------------------------------------------------


def test_empty_holdout_warns(label_cfg):
    warnings = preflight_optimize(label_cfg, _balanced(3), test_case_count=0)
    assert any("holdout" in w for w in warnings)


def test_train_below_recommended_warns(label_cfg):
    train = _balanced(3)  # 12 cases: above the minimum, below the recommendation
    assert MIN_TRAIN_CASES <= len(train) < RECOMMENDED_TRAIN_CASES
    warnings = preflight_optimize(label_cfg, train, test_case_count=4)
    assert any(str(RECOMMENDED_TRAIN_CASES) in w for w in warnings)


def test_ample_balanced_data_passes_clean(label_cfg):
    warnings = preflight_optimize(label_cfg, _balanced(8), test_case_count=8)  # 32 train
    assert warnings == []


# ---------------------------------------------------------------------------
# acceptance: the real tracked sample-inquiry task passes preflight
# ---------------------------------------------------------------------------


def test_sample_inquiry_task_passes_preflight():
    cfg, paths = load_task("sample-inquiry", root=REPO_ROOT)
    cases = load_golden_jsonl(paths.golden)
    train = [c for c in cases if c.split == "train"]
    test_count = sum(1 for c in cases if c.split == "test")

    warnings = preflight_optimize(cfg, train, test_count)  # must not raise

    # 12 train cases: enough to optimize, still flagged as overfitting-prone
    assert any(str(RECOMMENDED_TRAIN_CASES) in w for w in warnings)
    assert not any(w.startswith("(--force)") for w in warnings)
