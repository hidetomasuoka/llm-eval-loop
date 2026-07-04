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
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.task.name == "sample-inquiry-classification"
    assert cfg.task.answer_type == "label"
    assert "契約照会" in cfg.task.labels
    aliases = [m.alias for m in cfg.models]
    assert "qwen7b" in aliases
    assert "haiku45" in aliases
    assert cfg.judge.provider
    assert cfg.optimize.target_alias == "qwen7b"


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


def test_load_golden_jsonl_parses_sample_dataset():
    cases = load_golden_jsonl(REPO_ROOT / "data" / "golden.jsonl")
    assert len(cases) == 20
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    assert len(train) == 8
    assert len(test) == 12
    assert all(c.source == "self-made" for c in cases)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


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


def test_load_human_labels_parses_sample():
    labels = load_human_labels(REPO_ROOT / "data" / "human_labels.jsonl")
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
