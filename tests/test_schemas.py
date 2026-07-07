from pathlib import Path

import pytest
import yaml

from evalloop.schemas import (
    SchemaError,
    assert_split_disjoint,
    load_global_config,
    load_golden_jsonl,
    load_human_labels,
    load_task,
    restrict_models,
)
from tests.conftest import scaffold_task

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# load_global_config
# ---------------------------------------------------------------------------


def test_load_global_config_parses_real_config():
    cfg = load_global_config(REPO_ROOT / "config.yaml")
    assert cfg.default_task  # a fresh clone must resolve to a working task
    aliases = [m.alias for m in cfg.models]
    assert "qwen7b" in aliases
    assert "haiku45" in aliases
    assert cfg.run.repeat >= 1


def test_load_global_config_parses_supports_sampling_params(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {
                "default_task": "t1",
                "models": [
                    {"provider": "p1", "alias": "default_model", "tier": "t"},
                    {"provider": "p2", "alias": "no_sampling", "tier": "t", "supports_sampling_params": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_global_config(cfg_file)
    by_alias = {m.alias: m for m in cfg.models}
    assert by_alias["default_model"].supports_sampling_params is True
    assert by_alias["no_sampling"].supports_sampling_params is False


def test_load_global_config_missing_models_raises(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("default_task: t1\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_global_config(bad)


def test_load_global_config_missing_file_raises(tmp_path):
    with pytest.raises(SchemaError):
        load_global_config(tmp_path / "nope.yaml")


def test_duplicate_alias_rejected(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"provider": "p1", "alias": "dup", "tier": "t"},
                    {"provider": "p2", "alias": "dup", "tier": "t"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_global_config(bad)


# ---------------------------------------------------------------------------
# load_task (global config.yaml + tasks/<name>/task.yaml merge)
# ---------------------------------------------------------------------------


def test_load_task_parses_real_sample_task():
    # tasks/sample-inquiry/ is the only task whose data is tracked in git
    # (fresh clone / CI). Keep the checks generic rather than tied to content.
    cfg, paths = load_task("sample-inquiry")
    assert cfg.task.name == "sample-inquiry"
    assert paths.task == "sample-inquiry"
    assert cfg.task.answer_type in {"label", "json", "text"}
    if cfg.task.answer_type == "label":
        assert cfg.task.labels
    aliases = [m.alias for m in cfg.models]
    assert "qwen7b" in aliases
    assert "haiku45" in aliases
    assert cfg.judge.provider
    assert cfg.optimize.target_alias in aliases
    # prompt/rubric come back as ABSOLUTE paths following the task convention
    assert Path(cfg.task.prompt_file).is_absolute()
    assert Path(cfg.task.prompt_file) == paths.prompt_file
    assert Path(cfg.judge.rubric_file) == paths.rubric_file
    assert paths.prompt_file.exists()


def test_load_task_missing_top_level_key_raises(tmp_path):
    scaffold_task(tmp_path)  # writes a valid workspace first
    task_yaml = tmp_path / "tasks" / "t1" / "task.yaml"
    raw = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
    del raw["judge"]
    task_yaml.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SchemaError, match="judge"):
        load_task("t1", root=tmp_path)


def test_load_task_label_requires_labels_list(tmp_path):
    with pytest.raises(SchemaError):
        scaffold_task(tmp_path, answer_type="label", labels=[])


def test_load_task_unknown_task_raises(tmp_path):
    scaffold_task(tmp_path)
    with pytest.raises(SchemaError, match="not found"):
        load_task("does-not-exist", root=tmp_path)


def test_load_task_models_subset_selection(tmp_path):
    cfg, _paths = scaffold_task(tmp_path, models=["haiku45"])
    assert [m.alias for m in cfg.models] == ["haiku45"]
    # omitted models: key means the full global registry
    cfg_all, _ = scaffold_task(tmp_path, name="t2", default_task="t2")
    assert [m.alias for m in cfg_all.models] == ["qwen7b", "haiku45"]


def test_load_task_unknown_model_alias_raises(tmp_path):
    with pytest.raises(SchemaError, match="not in the global registry"):
        scaffold_task(tmp_path, models=["nope"])


def test_load_task_run_overrides_merge_over_global_defaults(tmp_path):
    cfg, _paths = scaffold_task(
        tmp_path,
        global_run={"repeat": 2, "temperature": 0.5, "max_tokens": 512, "cost_warn_usd": 9.0},
        run_overrides={"repeat": 3},
    )
    assert cfg.run.repeat == 3  # task override wins
    assert cfg.run.temperature == 0.5  # global value survives
    assert cfg.run.max_tokens == 512
    assert cfg.run.cost_warn_usd == 9.0


# ---------------------------------------------------------------------------
# optimize.method / optimize.params ([APO-04])
# ---------------------------------------------------------------------------


def _set_optimize(root, name="t1", **extra):
    task_yaml = root / "tasks" / name / "task.yaml"
    raw = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
    raw["optimize"].update(extra)
    task_yaml.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")


def test_optimize_method_and_params_roundtrip(tmp_path):
    scaffold_task(tmp_path)
    _set_optimize(tmp_path, method="gepa", params={"auto": "medium", "seed": 7})
    cfg, _ = load_task("t1", root=tmp_path)
    assert cfg.optimize.method == "gepa"
    assert cfg.optimize.params == {"auto": "medium", "seed": 7}
    # backward-compat rule: params.auto takes precedence over the legacy
    # top-level auto (which scaffold_task writes as "light")
    assert cfg.optimize.auto == "medium"


def test_optimize_method_defaults_to_gepa_for_legacy_task_yaml(tmp_path):
    # a task.yaml without method/params keys (every pre-APO-04 task) must be
    # fully compatible
    cfg, _ = scaffold_task(tmp_path)
    assert cfg.optimize.method == "gepa"
    assert cfg.optimize.params == {}
    assert cfg.optimize.auto == "light"


def test_optimize_unknown_method_fails_fast_at_load(tmp_path):
    scaffold_task(tmp_path)
    _set_optimize(tmp_path, method="genetic-annealing")
    with pytest.raises(SchemaError, match="optimize.method"):
        load_task("t1", root=tmp_path)


def test_task_resolution_precedence_flag_env_default(tmp_path, monkeypatch):
    scaffold_task(tmp_path, name="from-default", default_task="from-default")
    scaffold_task(tmp_path, name="from-env", default_task="from-default")
    scaffold_task(tmp_path, name="from-flag", default_task="from-default")

    monkeypatch.delenv("EVALLOOP_TASK", raising=False)
    cfg, _ = load_task(None, root=tmp_path)
    assert cfg.task.name == "from-default"

    monkeypatch.setenv("EVALLOOP_TASK", "from-env")
    cfg, _ = load_task(None, root=tmp_path)
    assert cfg.task.name == "from-env"

    # explicit --task beats both the env var and default_task
    cfg, _ = load_task("from-flag", root=tmp_path)
    assert cfg.task.name == "from-flag"


def test_no_task_anywhere_raises(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"models": [{"provider": "p", "alias": "a", "tier": "t"}]}), encoding="utf-8"
    )
    monkeypatch.delenv("EVALLOOP_TASK", raising=False)
    with pytest.raises(SchemaError, match="no task specified"):
        load_task(None, root=tmp_path)


def test_restrict_models_narrows_and_rejects_unknown(tmp_path):
    cfg, _paths = scaffold_task(tmp_path)
    narrowed = restrict_models(cfg, ["haiku45"])
    assert [m.alias for m in narrowed.models] == ["haiku45"]
    # the original config object is untouched
    assert [m.alias for m in cfg.models] == ["qwen7b", "haiku45"]
    with pytest.raises(SchemaError, match="nope"):
        restrict_models(cfg, ["nope"])


# ---------------------------------------------------------------------------
# golden.jsonl
# ---------------------------------------------------------------------------


def test_load_golden_jsonl_parses_real_project_data():
    # the pristine sample task dataset must be intact (tracked demo data)
    cases = load_golden_jsonl(REPO_ROOT / "tasks" / "sample-inquiry" / "golden.jsonl")
    assert len(cases) == 24  # 12 train / 12 test (train extended to 3 per label for the APO-09 preflight)
    train = [c for c in cases if c.split == "train"]
    test = [c for c in cases if c.split == "test"]
    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(cases)
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


# ---------------------------------------------------------------------------
# human_labels.jsonl
# ---------------------------------------------------------------------------


def test_load_human_labels_parses_empty_file(tmp_path):
    # a task without a human review pass yet keeps an empty human_labels.jsonl;
    # it must parse cleanly (not error) -- calibrate() itself raises a clear
    # CalibrateError when there is nothing to compare, which is the right signal.
    empty = tmp_path / "human_labels.jsonl"
    empty.write_text("", encoding="utf-8")
    labels = load_human_labels(empty)
    assert labels == []


def test_load_human_labels_parses_sample_demo_file():
    labels = load_human_labels(REPO_ROOT / "tasks" / "sample-inquiry" / "human_labels.jsonl")
    assert len(labels) == 10
    assert {label.human_verdict for label in labels} <= {"pass", "fail"}


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
