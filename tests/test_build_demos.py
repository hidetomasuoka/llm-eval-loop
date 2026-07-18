"""demos.jsonl + {{demos}} expansion (APO-16 / issue #75)."""

from __future__ import annotations

import json

import pytest
import yaml

from evalloop import build as build_mod
from evalloop.demos import (
    DemoCase,
    DemoError,
    assert_demos_do_not_leak_test,
    format_demos,
    load_demos_jsonl,
)
from tests.conftest import DEFAULT_LABELS, default_golden_rows, scaffold_task


def test_format_demos_is_pure_and_stable():
    text = format_demos(
        [
            DemoCase(input="hello", output="契約照会"),
            DemoCase(input="bye", output="その他", id="x"),
        ]
    )
    assert text == "Input: hello\nOutput: 契約照会\n\nInput: bye\nOutput: その他\n\n"
    assert format_demos([]) == ""


def test_assert_demos_do_not_leak_test_by_id_and_input():
    demos = [DemoCase(input="holdout text", output="x", id="case-0099")]
    with pytest.raises(DemoError, match="case-0099"):
        assert_demos_do_not_leak_test(demos, test_ids={"case-0099"}, test_inputs=set())
    with pytest.raises(DemoError, match="test-split input"):
        assert_demos_do_not_leak_test(
            [DemoCase(input="holdout text", output="x")],
            test_ids=set(),
            test_inputs={"holdout text"},
        )


def test_build_without_demos_keeps_prompt_path_to_task_txt(isolated_root):
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=8)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="Q:\n{{input}}\n",
    )
    build_mod.build(cfg, paths, yes=True)
    config = yaml.safe_load(paths.promptfoo_config.read_text(encoding="utf-8"))
    assert config["prompts"][0].endswith("prompts/task.txt")
    assert not paths.resolved_prompt.exists()


def test_build_embeds_demos_and_writes_resolved_prompt(isolated_root):
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=8)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="Examples:\n{{demos}}Q:\n{{input}}\n",
    )
    paths.demos.write_text(
        json.dumps({"id": "case-0001", "input": "demo in", "output": "契約照会"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    build_mod.build(cfg, paths, yes=True)
    resolved = paths.resolved_prompt.read_text(encoding="utf-8")
    assert "Input: demo in\nOutput: 契約照会" in resolved
    assert "{{demos}}" not in resolved
    config = yaml.safe_load(paths.promptfoo_config.read_text(encoding="utf-8"))
    assert "prompt.resolved.txt" in config["prompts"][0]


def test_build_errors_when_placeholder_without_demos_file(isolated_root):
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=4)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="{{demos}}\n{{input}}\n",
    )
    with pytest.raises(build_mod.BuildError, match="demos.jsonl"):
        build_mod.build(cfg, paths, yes=True)


def test_build_warns_when_demos_file_without_placeholder(isolated_root, capsys):
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=4)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="{{input}}\n",
    )
    paths.demos.write_text(
        json.dumps({"input": "x", "output": "契約照会"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_mod.build(cfg, paths, yes=True)
    captured = capsys.readouterr().out
    assert "WARN" in captured and "{{demos}}" in captured


def test_build_errors_on_test_split_leak(isolated_root):
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=4)
    test_id = next(r["id"] for r in rows if r["split"] == "test")
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="{{demos}}\n{{input}}\n",
    )
    paths.demos.write_text(
        json.dumps({"id": test_id, "input": "leak", "output": "契約照会"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(build_mod.BuildError, match="leaks test-split"):
        build_mod.build(cfg, paths, yes=True)


def test_load_demos_jsonl_requires_rows(tmp_path):
    p = tmp_path / "demos.jsonl"
    p.write_text("\n", encoding="utf-8")
    with pytest.raises(DemoError, match="no demo rows"):
        load_demos_jsonl(p)


def test_real_sample_inquiry_build_embeds_tracked_demos(isolated_root):
    import shutil

    from evalloop.paths import REPO_ROOT
    from evalloop.schemas import load_task

    shutil.copy(REPO_ROOT / "config.yaml", isolated_root / "config.yaml")
    shutil.copytree(REPO_ROOT / "tasks" / "sample-inquiry", isolated_root / "tasks" / "sample-inquiry")
    cfg, paths = load_task("sample-inquiry", root=isolated_root)
    build_mod.build(cfg, paths, yes=True)
    assert paths.demos.exists()
    assert paths.resolved_prompt.exists()
    assert "Input:" in paths.resolved_prompt.read_text(encoding="utf-8")
