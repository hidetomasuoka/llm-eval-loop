"""Demo-order shuffle sensitivity (APO-19 / issue #78)."""

from __future__ import annotations

import json

import pytest
import yaml

from evalloop import build as build_mod
from evalloop.demos import DemoCase, format_demos, shuffle_demos
from evalloop.sensitivity import (
    SensitivityError,
    build_demoshuffle_variants,
    demoshuffle_variant_name,
)
from tests.conftest import DEFAULT_LABELS, default_golden_rows, scaffold_task


def test_shuffle_demos_is_reproducible_for_same_seed():
    demos = [
        DemoCase(input="a", output="契約照会", id="1"),
        DemoCase(input="b", output="障害報告", id="2"),
        DemoCase(input="c", output="機能要望", id="3"),
    ]
    a = shuffle_demos(demos, seed=7)
    b = shuffle_demos(demos, seed=7)
    assert a == b
    assert [d.id for d in a] == [d.id for d in b]
    # Different seed should usually change order (3! = 6 permutations; seed 0 vs 1).
    other = shuffle_demos(demos, seed=0)
    assert {d.id for d in other} == {"1", "2", "3"}


def test_shuffle_demos_does_not_mutate_input():
    demos = [
        DemoCase(input="a", output="契約照会"),
        DemoCase(input="b", output="障害報告"),
    ]
    original = list(demos)
    shuffle_demos(demos, seed=1)
    assert demos == original


def _write_train_demos(paths, n: int = 3) -> None:
    rows = []
    for i in range(1, n + 1):
        rows.append(
            json.dumps(
                {
                    "id": f"case-{i:04d}",
                    "input": f"問い合わせ文サンプル{i}",
                    "output": "契約照会",
                },
                ensure_ascii=False,
            )
        )
    paths.demos.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_shuffle_demos_writes_named_variants(isolated_root):
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=8)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="Examples:\n{{demos}}Q:\n{{input}}\n",
    )
    _write_train_demos(paths, n=3)

    build_mod.build(cfg, paths, yes=True, shuffle_demos=3)

    for seed in range(3):
        name = demoshuffle_variant_name(paths.task, seed)
        variant_path = paths.variants_dir / f"{name}.yaml"
        assert variant_path.exists()
        resolved = paths.build_dir / f"demoshuffle_{seed}.txt"
        assert resolved.exists()
        cfg_yaml = yaml.safe_load(variant_path.read_text(encoding="utf-8"))
        assert any("demoshuffle" in p for p in cfg_yaml["prompts"])
        assert "demoshuffle seed=" in cfg_yaml["description"]

    # Same seed → same rendered order
    again = shuffle_demos(
        [
            DemoCase(input="問い合わせ文サンプル1", output="契約照会", id="case-0001"),
            DemoCase(input="問い合わせ文サンプル2", output="契約照会", id="case-0002"),
            DemoCase(input="問い合わせ文サンプル3", output="契約照会", id="case-0003"),
        ],
        seed=0,
    )
    text0 = (paths.build_dir / "demoshuffle_0.txt").read_text(encoding="utf-8")
    assert format_demos(again).strip() in text0


def test_build_shuffle_demos_errors_without_placeholder(isolated_root):
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=8)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="Q:\n{{input}}\n",
    )
    with pytest.raises(build_mod.BuildError, match="requires \\{\\{demos\\}\\}"):
        build_mod.build(cfg, paths, yes=True, shuffle_demos=2)


def test_build_shuffle_demos_errors_without_demos_file(isolated_root):
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=8)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="Examples:\n{{demos}}Q:\n{{input}}\n",
    )
    # Placeholder present but demos.jsonl missing → build fails (before or at shuffle).
    with pytest.raises(build_mod.BuildError, match="demos\\.jsonl"):
        build_mod.build(cfg, paths, yes=True, shuffle_demos=2)


def test_sensitivity_errors_without_placeholder_after_base_build(isolated_root):
    """Direct API: task without {{demos}} gets a clear SensitivityError."""
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=8)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="Q:\n{{input}}\n",
    )
    build_mod.build(cfg, paths, yes=True)
    with pytest.raises(SensitivityError, match="no demos placeholder"):
        build_demoshuffle_variants(cfg, paths, 2)


def test_build_demoshuffle_variants_rejects_non_positive(isolated_root):
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=8)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="Examples:\n{{demos}}Q:\n{{input}}\n",
    )
    _write_train_demos(paths)
    build_mod.build(cfg, paths, yes=True)
    with pytest.raises(SensitivityError, match="positive"):
        build_demoshuffle_variants(cfg, paths, 0)
