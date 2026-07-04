from pathlib import Path

import pytest

from evalloop.schemas import (
    SchemaError,
    assert_split_disjoint,
    load_config,
    load_golden_jsonl,
    load_human_labels,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_config_parses_real_config():
    # config.yaml is the project's *active* task config, which is explicitly
    # meant to be swapped for a real task (currently CUAD-100 clause
    # extraction; previously the sample inquiry-classification demo). These
    # checks are deliberately generic rather than tied to one task's content.
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.task.name
    assert cfg.task.answer_type in {"label", "json", "text"}
    if cfg.task.answer_type == "label":
        assert cfg.task.labels
    aliases = [m.alias for m in cfg.models]
    assert "qwen7b" in aliases
    assert "haiku45" in aliases
    assert cfg.judge.provider
    assert cfg.optimize.target_alias in aliases


def test_load_config_parses_supports_sampling_params(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "task:",
                "  name: x",
                "  answer_type: label",
                "  prompt_file: p.txt",
                "  labels: [a, b]",
                "models:",
                "  - {provider: p1, alias: default_model, tier: t}",
                "  - {provider: p2, alias: no_sampling, tier: t, supports_sampling_params: false}",
                "judge:",
                "  provider: j",
                "optimize:",
                "  target_alias: default_model",
                "  reflection_provider: r",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.model_by_alias("default_model").supports_sampling_params is True
    assert cfg.model_by_alias("no_sampling").supports_sampling_params is False


def test_load_config_missing_top_level_key(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("task:\n  name: x\n  answer_type: label\n  prompt_file: p.txt\n  labels: [a]\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_config(bad)


def test_task_config_label_requires_labels_list(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text(
        "\n".join(
            [
                "task:",
                "  name: x",
                "  answer_type: label",
                "  prompt_file: p.txt",
                "  labels: []",
                "models:",
                "  - {provider: p, alias: a, tier: t}",
                "judge:",
                "  provider: j",
                "optimize:",
                "  target_alias: a",
                "  reflection_provider: r",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_config(bad)


def test_duplicate_alias_rejected(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text(
        "\n".join(
            [
                "task:",
                "  name: x",
                "  answer_type: label",
                "  prompt_file: p.txt",
                "  labels: [a, b]",
                "models:",
                "  - {provider: p1, alias: dup, tier: t}",
                "  - {provider: p2, alias: dup, tier: t}",
                "judge:",
                "  provider: j",
                "optimize:",
                "  target_alias: dup",
                "  reflection_provider: r",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_config(bad)


def test_load_golden_jsonl_parses_real_project_data():
    # data/golden.jsonl is the active task's dataset (currently CUAD-100;
    # see test_build.py's note on why these checks are kept generic).
    cases = load_golden_jsonl(REPO_ROOT / "data" / "golden.jsonl")
    assert len(cases) > 0
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(cases)
    assert all(c.source for c in cases)  # meta.source is required and non-empty
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))

    # the pristine sample task demo must still be intact and untouched
    sample_cases = load_golden_jsonl(REPO_ROOT / "data" / "sample" / "golden.jsonl")
    assert len(sample_cases) == 20
    assert all(c.source == "self-made" for c in sample_cases)


def test_load_golden_jsonl_rejects_duplicate_id(tmp_path):
    bad = tmp_path / "golden.jsonl"
    bad.write_text(
        '{"id": "case-0001", "input": "a", "expected": "x", "split": "train", "meta": {"category": "c", "source": "self-made"}}\n'
        '{"id": "case-0001", "input": "b", "expected": "y", "split": "test", "meta": {"category": "c", "source": "self-made"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_golden_jsonl(bad)


def test_load_golden_jsonl_rejects_invalid_split(tmp_path):
    bad = tmp_path / "golden.jsonl"
    bad.write_text(
        '{"id": "case-0001", "input": "a", "expected": "x", "split": "validation", "meta": {"category": "c", "source": "self-made"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_golden_jsonl(bad)


def test_load_golden_jsonl_requires_meta_source(tmp_path):
    bad = tmp_path / "golden.jsonl"
    bad.write_text(
        '{"id": "case-0001", "input": "a", "expected": "x", "split": "train", "meta": {"category": "c"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_golden_jsonl(bad)


def test_assert_split_disjoint_raises_on_overlap():
    with pytest.raises(SchemaError):
        assert_split_disjoint({"case-0001", "case-0002"}, {"case-0002", "case-0003"})


def test_assert_split_disjoint_passes_when_disjoint():
    assert_split_disjoint({"case-0001"}, {"case-0002"})


def test_load_human_labels_parses_real_project_file():
    # data/human_labels.jsonl is intentionally empty right now: the active
    # task (CUAD-100) hasn't had a human review pass yet, and fabricating
    # verdicts would defeat the point of `evalloop calibrate`. An empty file
    # must parse cleanly (not error) -- calibrate() itself raises a clear
    # CalibrateError if it's empty, which is the correct signal here.
    labels = load_human_labels(REPO_ROOT / "data" / "human_labels.jsonl")
    assert isinstance(labels, list)
    assert {l.human_verdict for l in labels} <= {"pass", "fail"}


def test_load_human_labels_parses_sample_demo_file():
    labels = load_human_labels(REPO_ROOT / "data" / "sample" / "human_labels.jsonl")
    assert len(labels) == 10
    assert {l.human_verdict for l in labels} <= {"pass", "fail"}


def test_load_human_labels_rejects_bad_verdict(tmp_path):
    bad = tmp_path / "human_labels.jsonl"
    bad.write_text(
        '{"case_id": "case-0001", "model_label": "m", "output_raw": "o", "human_verdict": "maybe"}\n',
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_human_labels(bad)


def test_load_human_labels_rejects_duplicate_case_model_pair(tmp_path):
    # (case_id, model_label)は複合主キー。重複はcalibrateの一致率を静かに
    # 汚染するため、ロード時点でエラーにする（issue #6）
    bad = tmp_path / "human_labels.jsonl"
    bad.write_text(
        "\n".join(
            [
                '{"case_id": "case-0001", "model_label": "haiku45", "output_raw": "a", "human_verdict": "pass"}',
                '{"case_id": "case-0001", "model_label": "haiku45", "output_raw": "b", "human_verdict": "fail"}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="duplicate"):
        load_human_labels(bad)


def test_load_human_labels_allows_same_case_across_models(tmp_path):
    ok = tmp_path / "human_labels.jsonl"
    ok.write_text(
        "\n".join(
            [
                '{"case_id": "case-0001", "model_label": "haiku45", "output_raw": "a", "human_verdict": "pass"}',
                '{"case_id": "case-0001", "model_label": "qwen7b", "output_raw": "b", "human_verdict": "fail"}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    labels = load_human_labels(ok)
    assert len(labels) == 2
