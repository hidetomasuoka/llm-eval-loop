"""Unit tests for the optimize preflight checks (APO-09, issue #68).

Covers each error/warning condition, the --force demotion, and that the
sample-inquiry-style task (the bundled default) passes cleanly. Preflight
is a pure data check -- no LM calls, no promptfoo -- so these tests run in
isolation without any monkeypatching of dspy/promptfoo.
"""

from __future__ import annotations

import pytest

from evalloop.optimizers.base import OptimizeError
from evalloop.preflight import (
    PreflightResult,
    check_or_raise,
    format_preflight,
    run_preflight,
)
from evalloop.schemas import GoldenCase, load_golden_jsonl, load_task
from tests.conftest import DEFAULT_LABELS, default_golden_rows, scaffold_task

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cases(rows: list[dict], split: str) -> list[GoldenCase]:
    """Filter golden rows to one split and build GoldenCase objects."""

    # load_golden_jsonl takes a file path; build a minimal in-memory list
    # by reusing its parsing logic through a temp file would be heavier than
    # just constructing the objects directly.
    out = []
    for r in rows:
        if r["split"] != split:
            continue
        out.append(
            GoldenCase(
                id=r["id"],
                input=r["input"],
                expected=r["expected"],
                split=r["split"],
                category=r["meta"]["category"],
                difficulty=r["meta"].get("difficulty"),
                source=r["meta"]["source"],
                raw_meta=r["meta"],
            )
        )
    return out


def _cfg_label(root, labels=None, n_train=4, n_test=4):
    """Build a label-task Config + train/test GoldenCase lists."""
    rows = default_golden_rows(labels=labels, n_train=n_train, n_test=n_test)
    cfg, _paths = scaffold_task(root, answer_type="label", labels=labels, golden_rows=rows)
    train = _cases(rows, "train")
    test = _cases(rows, "test")
    return cfg, train, len(test)


# ---------------------------------------------------------------------------
# error: train too small
# ---------------------------------------------------------------------------


def test_train_below_min_errors(isolated_root):
    cfg, train, test_count = _cfg_label(isolated_root, n_train=4, n_test=4)
    result = run_preflight(cfg, train, test_count)
    assert not result.ok
    assert any("at least 10" in e for e in result.errors)


def test_train_below_min_force_demotes_to_warning(isolated_root):
    cfg, train, test_count = _cfg_label(isolated_root, n_train=4, n_test=4)
    result = run_preflight(cfg, train, test_count, force=True)
    assert result.ok  # errors demoted
    assert not result.errors
    assert any("[forced]" in w and "at least 10" in w for w in result.warnings)


def test_train_at_min_passes(isolated_root):
    # 10 train cases, each label appears >= 2 times across 2 labels -> need >= 12
    # to avoid the singleton-label error, so use 2 labels with 5 each
    rows = default_golden_rows(labels=["A", "B"], n_train=10, n_test=4)
    cfg, _p = scaffold_task(isolated_root, answer_type="label", labels=["A", "B"], golden_rows=rows)
    train = _cases(rows, "train")
    result = run_preflight(cfg, train, 4)
    # 10 cases is exactly the floor -> no train-size error
    assert not any("at least 10" in e for e in result.errors)


# ---------------------------------------------------------------------------
# error (label tasks): label coverage
# ---------------------------------------------------------------------------


def test_unseen_label_errors(isolated_root):
    # task.yaml declares 4 labels but train only has the first 2
    rows = default_golden_rows(labels=["A", "B", "C", "D"], n_train=4, n_test=4)
    cfg, _p = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=["A", "B", "C", "D"],
        golden_rows=rows,
    )
    train = _cases(
        [
            {"id": "c1", "input": "x", "expected": "A", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}},
            {"id": "c2", "input": "x", "expected": "A", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}},
            {"id": "c3", "input": "x", "expected": "B", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}},
            {"id": "c4", "input": "x", "expected": "B", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}},
            {"id": "c5", "input": "x", "expected": "B", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}},
            {"id": "c6", "input": "x", "expected": "B", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}},
            {"id": "c7", "input": "x", "expected": "B", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}},
            {"id": "c8", "input": "x", "expected": "B", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}},
            {"id": "c9", "input": "x", "expected": "B", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}},
            {"id": "c10", "input": "x", "expected": "B", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}},
        ],
        "train",
    )
    result = run_preflight(cfg, train, 4)
    assert not result.ok
    assert any("'C'" in e and "never appears" in e for e in result.errors)
    assert any("'D'" in e and "never appears" in e for e in result.errors)


def test_singleton_label_errors(isolated_root):
    rows = default_golden_rows(labels=["A", "B"], n_train=10, n_test=4)
    cfg, _p = scaffold_task(isolated_root, answer_type="label", labels=["A", "B"], golden_rows=rows)
    # 10 cases: A once, B nine times -> A is a singleton
    train = _cases(
        [
            {"id": f"c{i}", "input": "x", "expected": "B" if i > 0 else "A", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}}
            for i in range(10)
        ],
        "train",
    )
    result = run_preflight(cfg, train, 4)
    assert not result.ok
    assert any("'A'" in e and "only 1 time" in e for e in result.errors)


def test_singleton_label_force_demotes(isolated_root):
    rows = default_golden_rows(labels=["A", "B"], n_train=10, n_test=4)
    cfg, _p = scaffold_task(isolated_root, answer_type="label", labels=["A", "B"], golden_rows=rows)
    train = _cases(
        [
            {"id": f"c{i}", "input": "x", "expected": "B" if i > 0 else "A", "split": "train",
             "meta": {"category": "基本", "source": "self-made"}}
            for i in range(10)
        ],
        "train",
    )
    result = run_preflight(cfg, train, 4, force=True)
    assert result.ok
    assert any("[forced]" in w and "only 1 time" in w for w in result.warnings)


def test_label_coverage_normalizes_task_yaml_spelling_variants(isolated_root):
    # task.yaml labels written with quotes / trailing punctuation / full-width
    # chars must match train expected values in the training metric's
    # normalized space -- comparing the raw task.yaml strings against the
    # normalized counts misreported them as unseen (PR #96 review finding)
    task_labels = ["「契約照会」", "障害報告。", "ＡＢ"]
    plain = ["契約照会", "障害報告", "AB"]  # what golden.jsonl actually contains
    rows = [
        {"id": f"c{i}", "input": "x", "expected": plain[i % 3], "split": "train",
         "meta": {"category": "基本", "source": "self-made"}}
        for i in range(12)
    ]
    cfg, _p = scaffold_task(isolated_root, answer_type="label", labels=task_labels, golden_rows=rows)
    train = _cases(rows, "train")
    result = run_preflight(cfg, train, 4)
    assert result.ok, f"spelling variants misjudged as uncovered: {result.errors}"


def test_singleton_reported_with_task_yaml_spelling(isolated_root):
    # the singleton check must also compare in normalized space, and the error
    # must quote the task.yaml spelling so the user can find the label they wrote
    task_labels = ["A", "Ｂ。"]  # full-width + trailing punctuation in task.yaml
    rows = [
        {"id": f"c{i}", "input": "x", "expected": "B" if i == 0 else "A", "split": "train",
         "meta": {"category": "基本", "source": "self-made"}}
        for i in range(10)
    ]
    cfg, _p = scaffold_task(isolated_root, answer_type="label", labels=task_labels, golden_rows=rows)
    train = _cases(rows, "train")
    result = run_preflight(cfg, train, 4)
    assert not result.ok
    assert any("'Ｂ。'" in e and "only 1 time" in e for e in result.errors)
    assert not any("never appears" in e for e in result.errors)


# ---------------------------------------------------------------------------
# warning: no holdout
# ---------------------------------------------------------------------------


def test_empty_holdout_warns(isolated_root):
    cfg, train, _test_count = _cfg_label(isolated_root, n_train=12, n_test=0)
    result = run_preflight(cfg, train, 0)
    # 12 train cases with 4 labels -> each label appears 3 times, no label errors,
    # but no holdout -> warning
    assert result.ok  # warnings don't block
    assert any("holdout" in w and "empty" in w for w in result.warnings)


def test_empty_holdout_no_warning_when_test_present(isolated_root):
    cfg, train, test_count = _cfg_label(isolated_root, n_train=12, n_test=4)
    result = run_preflight(cfg, train, test_count)
    assert not any("holdout" in w and "empty" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# warning: small train
# ---------------------------------------------------------------------------


def test_small_train_warns(isolated_root):
    # 10 train cases (>= MIN_TRAIN_CASES, < SMALL_TRAIN_WARN=30) -> warning
    rows = default_golden_rows(labels=["A", "B"], n_train=10, n_test=4)
    cfg, _p = scaffold_task(isolated_root, answer_type="label", labels=["A", "B"], golden_rows=rows)
    rows = default_golden_rows(labels=["A", "B"], n_train=10, n_test=4)
    train = _cases(rows, "train")
    result = run_preflight(cfg, train, 4)
    assert result.ok
    assert any("overfitting risk" in w for w in result.warnings)


def test_large_train_no_size_warning(isolated_root):
    # 30 train cases, 2 labels -> each label appears 15 times, no warnings
    rows = default_golden_rows(labels=["A", "B"], n_train=30, n_test=4)
    cfg, _p = scaffold_task(isolated_root, answer_type="label", labels=["A", "B"], golden_rows=rows)
    rows = default_golden_rows(labels=["A", "B"], n_train=30, n_test=4)
    train = _cases(rows, "train")
    result = run_preflight(cfg, train, 4)
    assert not any("overfitting risk" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# check_or_raise
# ---------------------------------------------------------------------------


def test_check_or_raise_aborts_on_error(isolated_root):
    result = PreflightResult(errors=["bad"], warnings=[])
    with pytest.raises(OptimizeError, match="preflight failed"):
        check_or_raise(result, force=False)


def test_check_or_raise_passes_when_force(isolated_root):
    # when force=True, run_preflight already demoted errors to warnings,
    # so check_or_raise sees an empty errors list and passes
    result = PreflightResult(errors=[], warnings=["[forced] bad"])
    check_or_raise(result, force=True)  # should not raise


def test_check_or_raise_passes_on_warnings_only(isolated_root):
    result = PreflightResult(errors=[], warnings=["meh"])
    check_or_raise(result, force=False)  # should not raise


# ---------------------------------------------------------------------------
# format_preflight
# ---------------------------------------------------------------------------


def test_format_preflight_styling():
    result = PreflightResult(errors=["e1"], warnings=["w1"])
    lines = format_preflight(result)
    assert any("preflight ERROR" in line and "e1" in line for line in lines)
    assert any("preflight WARN" in line and "w1" in line for line in lines)


# ---------------------------------------------------------------------------
# text/json tasks: only train-size and holdout checks apply (no label coverage)
# ---------------------------------------------------------------------------


def test_text_task_no_label_coverage_check(isolated_root):
    # text task: labels=[] so the label-coverage branch is skipped entirely.
    # 10 train cases -> passes the min-size check, warns on small train.
    rows = default_golden_rows(labels=None, n_train=10, n_test=4)
    cfg, _p = scaffold_task(isolated_root, answer_type="text", labels=[], golden_rows=rows)
    rows = default_golden_rows(labels=None, n_train=10, n_test=4)
    train = _cases(rows, "train")
    result = run_preflight(cfg, train, 4)
    assert result.ok
    assert not any("never appears" in e for e in result.errors)
    assert not any("only" in e and "time" in e for e in result.errors)


# ---------------------------------------------------------------------------
# integration: sample-inquiry-style task passes cleanly
# ---------------------------------------------------------------------------


def test_sample_inquiry_style_task_passes(isolated_root):
    # the bundled sample-inquiry task: 4 labels, 10 train (default_golden_rows),
    # 10 test. With 4 labels and 10 train, each label appears 2-3 times -> passes.
    # Use 12 train to guarantee each of the 4 labels appears >= 2 times.
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=10)
    cfg, _p = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
    )
    train = _cases(rows, "train")
    result = run_preflight(cfg, train, 10)
    assert result.ok, f"expected pass, got errors={result.errors} warnings={result.warnings}"
    # 12 < 30 so the small-train warning should fire
    assert any("overfitting risk" in w for w in result.warnings)


def test_real_sample_inquiry_task_passes_preflight():
    # acceptance (issue #68): the REAL tracked dataset -- extended to 24 cases
    # (train 12 = 3 per label) in PR #97 -- must clear preflight without --force
    cfg, paths = load_task("sample-inquiry")
    cases = load_golden_jsonl(paths.golden)
    train = [c for c in cases if c.split == "train"]
    test_count = sum(1 for c in cases if c.split == "test")

    result = run_preflight(cfg, train, test_count)

    assert result.ok, f"errors={result.errors}"
    # 12 train < 30 -> the overfitting warning still fires
    assert any("overfitting risk" in w for w in result.warnings)