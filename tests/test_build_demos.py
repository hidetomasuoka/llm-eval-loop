"""demos.jsonl + {{demos}} expansion (APO-16 / issue #75)."""

from __future__ import annotations

import json

import pytest
import yaml

from evalloop import build as build_mod
from evalloop.demos import (
    DEMOS_PLACEHOLDER,
    DemoCase,
    DemoError,
    assert_demos_do_not_leak_test,
    expand_demos_in_template,
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


def test_expand_demos_in_template_shared_by_build_and_optimize(tmp_path):
    demos_path = tmp_path / "demos.jsonl"
    demos_path.write_text(
        json.dumps({"id": "case-0001", "input": "demo in", "output": "契約照会"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    template = f"Examples:\n{DEMOS_PLACEHOLDER}Q:\n{{{{input}}}}\n"
    resolved, n = expand_demos_in_template(template, demos_path, test_ids={"case-0099"}, test_inputs={"holdout"})
    assert n == 1
    assert DEMOS_PLACEHOLDER not in resolved
    assert "Input: demo in\nOutput: 契約照会" in resolved
    unchanged, n_none = expand_demos_in_template(
        "no placeholder {{input}}", demos_path, test_ids=set(), test_inputs=set()
    )
    assert n_none is None
    assert unchanged == "no placeholder {{input}}"


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
        json.dumps({"id": "case-0001", "input": "demo in", "output": "契約照会"}, ensure_ascii=False) + "\n",
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


def test_load_demos_jsonl_rejects_non_string_fields(tmp_path):
    p = tmp_path / "demos.jsonl"
    p.write_text(
        json.dumps({"input": ["not", "a", "string"], "output": "契約照会"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DemoError, match="must be strings"):
        load_demos_jsonl(p)


def test_build_demo_leak_does_not_write_tests_yaml(isolated_root):
    """Failed demo validation must not refresh holdout YAML ahead of a good config."""
    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=4)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="{{demos}}\n{{input}}\n",
    )
    test_id = next(r["id"] for r in rows if r["split"] == "test")
    paths.demos.write_text(
        json.dumps({"id": test_id, "input": "leak", "output": "契約照会"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert not paths.tests_test.exists()
    with pytest.raises(build_mod.BuildError, match="leaks test-split"):
        build_mod.build(cfg, paths, yes=True)
    assert not paths.tests_test.exists()
    assert not paths.promptfoo_config.exists()


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


def test_optimize_embeds_demos_into_training_template(isolated_root, monkeypatch, capsys):
    """Bugbot #114: optimize must expand {{demos}} like build, not leave the placeholder."""
    import types

    from evalloop import optimize as optimize_mod
    from evalloop import run as run_mod

    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=8)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="Examples:\n{{demos}}Classify:\n{{input}}\n",
    )
    paths.demos.write_text(
        json.dumps({"id": "case-0001", "input": "demo in", "output": "契約照会"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_mod.build(cfg, paths, yes=True)

    seen = {}

    def fake_gepa(student, trainset, metric, reflection_lm, auto, seed=0):
        seen["instructions"] = student.signature.instructions
        return types.SimpleNamespace(signature=types.SimpleNamespace(instructions="optimized"))

    monkeypatch.setattr(optimize_mod, "run_gepa", fake_gepa)

    def fake_eval(config_path, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": {"results": []}}), encoding="utf-8")

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)
    monkeypatch.setattr(run_mod, "get_promptfoo_version", lambda: "0.0.0-test")
    monkeypatch.setattr(run_mod, "get_node_version", lambda: "v22.22.0")

    optimize_mod.optimize(cfg, paths, yes=True)
    out = capsys.readouterr().out
    assert "embedded 1 demos" in out
    assert DEMOS_PLACEHOLDER not in seen["instructions"]
    assert "Input: demo in\nOutput: 契約照会" in seen["instructions"]


def test_optimize_demo_leak_check_uses_build_yaml_holdout(isolated_root, monkeypatch):
    """Bugbot #114: leak check must cover tests_test.yaml, not only golden test rows."""
    import types

    from evalloop import optimize as optimize_mod
    from evalloop.optimizers.base import OptimizeError

    rows = default_golden_rows(labels=DEFAULT_LABELS, n_train=12, n_test=4)
    cfg, paths = scaffold_task(
        isolated_root,
        answer_type="label",
        labels=DEFAULT_LABELS,
        golden_rows=rows,
        prompt="{{demos}}\n{{input}}\n",
    )
    paths.demos.write_text(
        json.dumps(
            {"id": "stale-holdout", "input": "yaml-only holdout input", "output": "契約照会"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    build_mod.build(cfg, paths, yes=True)

    # Simulate golden drift without rebuild: no test rows left in golden, but
    # promptfoo still evaluates the previous tests_test.yaml holdout.
    train_only = [r for r in rows if r["split"] == "train"]
    paths.golden.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_only) + "\n",
        encoding="utf-8",
    )
    # Keep a stale holdout entry that is no longer in golden.
    paths.tests_test.write_text(
        yaml.safe_dump(
            [
                {
                    "description": "stale-holdout",
                    "vars": {
                        "case_id": "stale-holdout",
                        "input": "yaml-only holdout input",
                        "expected": "契約照会",
                        "category": "基本",
                    },
                }
            ],
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        optimize_mod,
        "run_gepa",
        lambda *a, **k: types.SimpleNamespace(signature=types.SimpleNamespace(instructions="x")),
    )

    with pytest.raises(OptimizeError, match="leaks test-split"):
        optimize_mod.optimize(cfg, paths, yes=True, force=True)
