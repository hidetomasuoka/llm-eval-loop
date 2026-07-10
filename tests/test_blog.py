import json
from pathlib import Path

import pytest
import yaml

from evalloop import blog as blog_mod
from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import (
    BlogConfig,
    Config,
    GoldenCase,
    JudgeConfig,
    ModelConfig,
    OptimizeConfig,
    RunConfig,
    TaskConfig,
)


def _case(id_, source="self-made"):
    return GoldenCase(id=id_, input="x", expected="契約照会", split="test", category="基本", difficulty="easy", source=source)


# ---------------------------------------------------------------------------
# guard 1: source
# ---------------------------------------------------------------------------


def test_check_source_guard_passes_when_all_self_made():
    blog_mod.check_source_guard([_case("case-0001"), _case("case-0002")])


def test_check_source_guard_raises_with_violating_ids():
    cases = [_case("case-0001"), _case("case-0002", source="scraped-from-web")]
    with pytest.raises(blog_mod.BlogGuardError, match="case-0002"):
        blog_mod.check_source_guard(cases)


# ---------------------------------------------------------------------------
# guard 2: secrets / local paths
# ---------------------------------------------------------------------------


def test_check_secret_guard_passes_on_clean_files(tmp_path):
    (tmp_path / "tables.md").write_text("| alias | accuracy |\n|---|---|\n| haiku45 | 90% |\n", encoding="utf-8")
    blog_mod.check_secret_guard(tmp_path)


def test_check_secret_guard_catches_anthropic_style_key(tmp_path):
    (tmp_path / "leak.md").write_text("key: sk-abcdefghijklmnopqrstuvwx", encoding="utf-8")
    with pytest.raises(blog_mod.BlogGuardError, match="secret pattern"):
        blog_mod.check_secret_guard(tmp_path)


def test_check_secret_guard_catches_aws_key(tmp_path):
    (tmp_path / "leak.md").write_text("AKIAABCDEFGHIJKLMNOP", encoding="utf-8")
    with pytest.raises(blog_mod.BlogGuardError, match="secret pattern"):
        blog_mod.check_secret_guard(tmp_path)


def test_check_secret_guard_catches_home_directory_path(tmp_path, monkeypatch):
    fake_home = tmp_path / "home" / "someone"
    fake_home.mkdir(parents=True)
    monkeypatch.setattr(blog_mod.Path, "home", staticmethod(lambda: fake_home))
    leak_dir = tmp_path / "staging"
    leak_dir.mkdir()
    (leak_dir / "leak.md").write_text(f"see {fake_home}/repo/config.yaml", encoding="utf-8")
    with pytest.raises(blog_mod.BlogGuardError, match="home directory"):
        blog_mod.check_secret_guard(leak_dir)


def test_check_secret_guard_ignores_png_binaries(tmp_path):
    (tmp_path / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\nsk-shouldnotmatterbecausebinary")
    blog_mod.check_secret_guard(tmp_path)


# ---------------------------------------------------------------------------
# labels / font fallback
# ---------------------------------------------------------------------------


def test_labels_japanese_when_cjk_font_found():
    labels = blog_mod._labels(has_cjk_font=True)
    assert labels.accuracy == "精度"


def test_labels_english_fallback_when_no_cjk_font():
    labels = blog_mod._labels(has_cjk_font=False)
    assert labels.accuracy == "Accuracy"
    assert all(ord(ch) < 128 for ch in labels.accuracy + labels.cost + labels.model + labels.category + labels.unassigned)


# ---------------------------------------------------------------------------
# conditions.md reproduce block
# ---------------------------------------------------------------------------


def _mk_config(answer_type, judge_provider, model_provider):
    return Config(
        task=TaskConfig(
            name="t1",
            answer_type=answer_type,
            prompt_file="tasks/sample-inquiry/prompts/task.txt",
            labels=["契約照会", "解約"] if answer_type == "label" else [],
        ),
        models=[ModelConfig(provider=model_provider, alias="m1", tier="small")],
        run=RunConfig(),
        judge=JudgeConfig(provider=judge_provider),
        optimize=OptimizeConfig(target_alias="m1", reflection_provider="r"),
        blog=BlogConfig(),
        path=Path("config.yaml"),
    )


def _mk_run_data(run_id="run-1", answer_type="label", judge_provider="j", model_providers=None):
    models = [{"provider": p, "alias": f"m{i}", "tier": "small"} for i, p in enumerate(model_providers or [])]
    meta = {
        "run_id": run_id,
        "task": "t1",
        "answer_type": answer_type,
        "repeat": 1,
        "prompt_file": "tasks/t1/prompts/task.txt",
        "prompt_sha256": "a" * 64,
        "promptfoo_config_sha256": "b" * 64,
        "models": models,
        "promptfoo_version": "0.0.0-test",
        "judge": {"provider": judge_provider},
    }
    return blog_mod.RunData(run_id=run_id, meta=meta, stats=[])


def test_conditions_reproduce_adds_allow_same_judge_for_same_judge_text_config():
    # run snapshot records answer_type=text and judge/model sharing the same
    # provider -- the reproduce block must carry --allow-same-judge or it won't
    # be copy-pastable (build.py iron rule #2 would abort)
    config = _mk_config("text", judge_provider="p:shared", model_provider="p:shared")
    run = _mk_run_data(answer_type="text", judge_provider="p:shared", model_providers=["p:shared"])
    md = blog_mod.render_conditions_md([run], config, fig03_written=False)
    assert "evalloop build --task t1 --allow-same-judge" in md


def test_conditions_reproduce_plain_build_when_judge_is_independent():
    config = _mk_config("text", judge_provider="p:judge", model_provider="p:model")
    run = _mk_run_data(answer_type="text", judge_provider="p:judge", model_providers=["p:model"])
    md = blog_mod.render_conditions_md([run], config, fig03_written=False)
    assert "evalloop build --task t1\n" in md
    assert "--allow-same-judge" not in md


def test_conditions_reproduce_plain_build_for_label_config():
    # same provider on both sides is irrelevant outside answer_type=text:
    # build.py only enforces iron rule #2 for the llm-rubric path
    config = _mk_config("label", judge_provider="p:shared", model_provider="p:shared")
    run = _mk_run_data(answer_type="label", judge_provider="p:shared", model_providers=["p:shared"])
    md = blog_mod.render_conditions_md([run], config, fig03_written=False)
    assert "--allow-same-judge" not in md


def test_conditions_reproduce_same_judge_uses_meta_not_config():
    # When the passed config disagrees with primary.meta, the meta (run
    # snapshot) must win.  Here the config says "same judge" but the snapshot
    # says the run actually used different providers.
    config = _mk_config("text", judge_provider="p:shared", model_provider="p:shared")
    run = _mk_run_data(answer_type="text", judge_provider="p:judge", model_providers=["p:model"])
    md = blog_mod.render_conditions_md([run], config, fig03_written=False)
    assert "--allow-same-judge" not in md


def test_conditions_reproduce_commands_are_task_scoped():
    # the reproduce block derives --task from the run snapshot's meta["task"]
    config = _mk_config("label", judge_provider="p:judge", model_provider="p:model")
    md = blog_mod.render_conditions_md([_mk_run_data("run-xyz")], config, fig03_written=False)
    assert "evalloop build --task t1" in md
    assert "evalloop run --task t1 --repeat 1" in md
    assert "evalloop report --task t1 run-xyz" in md
    assert "--run" not in md  # report takes a positional run_id, not --run
    assert "--config" not in md  # per-task workspaces replaced --config flags
    assert "prompt sha256 (first 8): `aaaaaaaa`" in md
    assert "promptfoo config sha256 (first 8): `bbbbbbbb`" in md


def test_conditions_preserves_explicit_unknown_prompt_identity():
    config = _mk_config("label", judge_provider="p:judge", model_provider="p:model")
    run = _mk_run_data()
    run.meta["prompt_file"] = None
    run.meta["prompt_sha256"] = None

    md = blog_mod.render_conditions_md([run], config, fig03_written=False)

    assert "prompt sha256 (first 8): `unknown`" in md


# ---------------------------------------------------------------------------
# full blog() orchestration
# ---------------------------------------------------------------------------


@pytest.fixture
def blog_env(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    paths.task_dir.mkdir(parents=True)
    cfg = Config(
        task=TaskConfig(
            name="t1",
            answer_type="label",
            prompt_file="tasks/sample-inquiry/prompts/task.txt",
            labels=["契約照会", "障害報告", "機能要望", "その他"],
        ),
        models=[
            ModelConfig(provider="ollama:chat:qwen2.5:7b", alias="qwen7b", tier="local"),
            ModelConfig(provider="anthropic:messages:claude-haiku-4-5-20251001", alias="haiku45", tier="small"),
        ],
        run=RunConfig(),
        judge=JudgeConfig(provider="anthropic:messages:claude-sonnet-4-6"),
        optimize=OptimizeConfig(target_alias="qwen7b", reflection_provider="r"),
        blog=BlogConfig(),
        path=REPO_ROOT / "config.yaml",
    )
    return {"paths": paths, "cfg": cfg}


def _write_golden(path, sources):
    with path.open("w", encoding="utf-8") as f:
        for i, source in enumerate(sources, start=1):
            f.write(
                json.dumps(
                    {
                        "id": f"case-{i:04d}",
                        "input": "x",
                        "expected": "契約照会",
                        "split": "test",
                        "meta": {"category": "基本", "source": source},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_run(runs_dir, run_id, aliases, variant=None):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    rows = [
        {
            "vars": {"case_id": "case-0001", "expected": "契約照会", "category": "基本"},
            "provider": {"id": alias, "label": alias},
            "response": {"output": "契約照会", "cached": False},
            "gradingResult": {"pass": True, "score": 1, "reason": "ok"},
            "success": True,
            "cost": 0.001,
            "latencyMs": 100,
        }
        for alias in aliases
    ]
    (run_dir / "output.json").write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task": "t1",
                "task_name": "t1",
                "answer_type": "label",
                "variant": variant,
                "repeat": 1,
                "prompt_file": "tasks/t1/prompts/task.txt",
                "prompt_sha256": "a" * 64,
                "models": [{"alias": a, "provider": f"p:{a}", "tier": "small"} for a in aliases],
                "promptfoo_version": "0.0.0-test",
                "judge": {"provider": "j", "calibration_status": "uncalibrated", "agreement_rate": None},
            }
        ),
        encoding="utf-8",
    )


def test_blog_aborts_on_source_guard_violation(blog_env):
    paths, cfg = blog_env["paths"], blog_env["cfg"]
    _write_golden(paths.golden, ["self-made", "scraped"])
    _write_run(paths.runs_dir, "run-1", ["haiku45"])

    with pytest.raises(blog_mod.BlogGuardError):
        blog_mod.blog(cfg, paths, run_ids=["run-1"])

    assert not paths.blog_dir.exists()


def test_blog_success_writes_expected_files(blog_env):
    paths, cfg = blog_env["paths"], blog_env["cfg"]
    _write_golden(paths.golden, ["self-made", "self-made"])
    _write_run(paths.runs_dir, "run-1", ["qwen7b", "haiku45"])

    out_dir = blog_mod.blog(cfg, paths, run_ids=["run-1"], slug="unit-test")

    assert paths.blog_dir in out_dir.parents
    assert (out_dir / "fig01_accuracy_by_model.png").exists()
    assert (out_dir / "fig01_accuracy_by_model.svg").exists()
    assert (out_dir / "fig02_cost_vs_accuracy.png").exists()
    assert not (out_dir / "fig03_failure_heatmap.png").exists()  # no taxonomy.yaml -> skipped

    tables = (out_dir / "tables.md").read_text(encoding="utf-8")
    assert "haiku45" in tables and "qwen7b" in tables

    conditions = (out_dir / "conditions.md").read_text(encoding="utf-8")
    assert "evalloop build --task t1" in conditions
    assert "evalloop run --task t1" in conditions

    article = (out_dir / "article_draft.md").read_text(encoding="utf-8")
    assert blog_mod.REVIEW_COMMENT in article
    assert "fig01_accuracy_by_model.png" in article


def test_blog_two_runs_produces_comparison_arrows_without_crashing(blog_env):
    paths, cfg = blog_env["paths"], blog_env["cfg"]
    _write_golden(paths.golden, ["self-made"])
    _write_run(paths.runs_dir, "before", ["qwen7b"])
    _write_run(paths.runs_dir, "after", ["qwen7b"], variant="qwen7b_opt")

    out_dir = blog_mod.blog(cfg, paths, run_ids=["before", "after"], slug="ab")
    assert (out_dir / "fig02_cost_vs_accuracy.png").exists()
    tables = (out_dir / "tables.md").read_text(encoding="utf-8")
    assert "before" in tables and "after" in tables


def test_blog_includes_fig03_when_taxonomy_defined(blog_env):
    paths, cfg = blog_env["paths"], blog_env["cfg"]
    _write_golden(paths.golden, ["self-made"])
    _write_run(paths.runs_dir, "run-1", ["qwen7b"])
    paths.taxonomy.write_text(
        yaml.safe_dump({"categories": [{"id": "c1", "name": "カテゴリ1", "definition": "d"}], "assignments": {}}),
        encoding="utf-8",
    )
    # make case-0001 actually fail so the heatmap has something to plot
    run_dir = paths.runs_dir / "run-1"
    data = json.loads((run_dir / "output.json").read_text(encoding="utf-8"))
    data["results"]["results"][0]["gradingResult"]["pass"] = False
    data["results"]["results"][0]["success"] = False
    (run_dir / "output.json").write_text(json.dumps(data), encoding="utf-8")

    out_dir = blog_mod.blog(cfg, paths, run_ids=["run-1"], slug="withfig3")
    assert (out_dir / "fig03_failure_heatmap.png").exists()


def test_blog_rejects_more_than_two_runs(blog_env):
    paths, cfg = blog_env["paths"], blog_env["cfg"]
    _write_golden(paths.golden, ["self-made"])
    with pytest.raises(blog_mod.BlogGuardError):
        blog_mod.blog(cfg, paths, run_ids=["a", "b", "c"])


def test_blog_rerun_same_slug_regenerates_not_accumulates(blog_env):
    paths, cfg = blog_env["paths"], blog_env["cfg"]
    _write_golden(paths.golden, ["self-made"])
    _write_run(paths.runs_dir, "run-1", ["qwen7b"])

    out_dir_1 = blog_mod.blog(cfg, paths, run_ids=["run-1"], slug="dup")
    (out_dir_1 / "stale_extra_file.txt").write_text("should be gone after regen", encoding="utf-8")

    out_dir_2 = blog_mod.blog(cfg, paths, run_ids=["run-1"], slug="dup")
    assert out_dir_1 == out_dir_2
    assert not (out_dir_2 / "stale_extra_file.txt").exists()
