import json
from pathlib import Path

import pytest
import yaml

from evalloop import analyze as analyze_mod
from evalloop import blog as blog_mod
from evalloop import build as build_mod
from evalloop import run as run_mod
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

REPO_ROOT = build_mod.REPO_ROOT


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
            name="t",
            answer_type=answer_type,
            prompt_file="prompts/base/task.txt",
            labels=["契約照会", "解約"] if answer_type == "label" else [],
        ),
        models=[ModelConfig(provider=model_provider, alias="m1", tier="small")],
        run=RunConfig(),
        judge=JudgeConfig(provider=judge_provider),
        optimize=OptimizeConfig(target_alias="m1", reflection_provider="r"),
        blog=BlogConfig(),
        path=Path("config.yaml"),
    )


def _mk_run_data(run_id="run-1"):
    meta = {
        "run_id": run_id,
        "repeat": 1,
        "prompt_file": "prompts/base/task.txt",
        "prompt_sha256": "a" * 64,
        "models": [],
        "promptfoo_version": "0.0.0-test",
        "judge": {"provider": "j"},
    }
    return blog_mod.RunData(run_id=run_id, meta=meta, stats=[])


def test_conditions_reproduce_adds_allow_same_judge_for_same_judge_text_config():
    # config.yaml-style setup: llm-rubric judge is also an evaluated model, so
    # a bare `evalloop build` aborts on iron rule #2 -- the reproduce block
    # must carry the override or it isn't copy-pastable
    config = _mk_config("text", judge_provider="p:shared", model_provider="p:shared")
    md = blog_mod.render_conditions_md([_mk_run_data()], config, fig03_written=False)
    assert "evalloop build --allow-same-judge" in md


def test_conditions_reproduce_plain_build_when_judge_is_independent():
    config = _mk_config("text", judge_provider="p:judge", model_provider="p:model")
    md = blog_mod.render_conditions_md([_mk_run_data()], config, fig03_written=False)
    assert "evalloop build\n" in md
    assert "--allow-same-judge" not in md


def test_conditions_reproduce_plain_build_for_label_config():
    # same provider on both sides is irrelevant outside answer_type=text:
    # build.py only enforces iron rule #2 for the llm-rubric path
    config = _mk_config("label", judge_provider="p:shared", model_provider="p:shared")
    md = blog_mod.render_conditions_md([_mk_run_data()], config, fig03_written=False)
    assert "--allow-same-judge" not in md


def test_conditions_reproduce_report_uses_positional_run_id():
    config = _mk_config("label", judge_provider="p:judge", model_provider="p:model")
    md = blog_mod.render_conditions_md([_mk_run_data("run-xyz")], config, fig03_written=False)
    assert "evalloop report run-xyz" in md
    assert "--run" not in md


# ---------------------------------------------------------------------------
# full blog() orchestration
# ---------------------------------------------------------------------------


@pytest.fixture
def blog_env(tmp_path, monkeypatch):
    golden_path = tmp_path / "golden.jsonl"
    runs_dir = tmp_path / "runs"
    blog_dir = tmp_path / "blog"
    taxonomy_path = tmp_path / "taxonomy.yaml"

    monkeypatch.setattr(build_mod, "GOLDEN_PATH", golden_path)
    monkeypatch.setattr(run_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(blog_mod, "BLOG_DIR", blog_dir)
    monkeypatch.setattr(analyze_mod, "TAXONOMY_PATH", taxonomy_path)

    return {"golden_path": golden_path, "runs_dir": runs_dir, "blog_dir": blog_dir, "taxonomy_path": taxonomy_path}


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
                "task_name": "sample-inquiry-classification",
                "answer_type": "label",
                "variant": variant,
                "repeat": 1,
                "prompt_file": "prompts/base/task.txt",
                "prompt_sha256": "a" * 64,
                "models": [{"alias": a, "provider": f"p:{a}", "tier": "small"} for a in aliases],
                "promptfoo_version": "0.0.0-test",
                "judge": {"provider": "j", "calibration_status": "uncalibrated", "agreement_rate": None},
            }
        ),
        encoding="utf-8",
    )


def test_blog_aborts_on_source_guard_violation(blog_env):
    _write_golden(blog_env["golden_path"], ["self-made", "scraped"])
    _write_run(blog_env["runs_dir"], "run-1", ["haiku45"])

    with pytest.raises(blog_mod.BlogGuardError):
        blog_mod.blog(run_ids=["run-1"], config_path=REPO_ROOT / "config.yaml")

    assert not blog_env["blog_dir"].exists()


def test_blog_success_writes_expected_files(blog_env):
    _write_golden(blog_env["golden_path"], ["self-made", "self-made"])
    _write_run(blog_env["runs_dir"], "run-1", ["qwen7b", "haiku45"])

    out_dir = blog_mod.blog(run_ids=["run-1"], slug="unit-test", config_path=REPO_ROOT / "config.yaml")

    assert (out_dir / "fig01_accuracy_by_model.png").exists()
    assert (out_dir / "fig01_accuracy_by_model.svg").exists()
    assert (out_dir / "fig02_cost_vs_accuracy.png").exists()
    assert not (out_dir / "fig03_failure_heatmap.png").exists()  # no taxonomy.yaml -> skipped

    tables = (out_dir / "tables.md").read_text(encoding="utf-8")
    assert "haiku45" in tables and "qwen7b" in tables

    conditions = (out_dir / "conditions.md").read_text(encoding="utf-8")
    assert "evalloop build" in conditions
    assert "evalloop run" in conditions

    article = (out_dir / "article_draft.md").read_text(encoding="utf-8")
    assert blog_mod.REVIEW_COMMENT in article
    assert "fig01_accuracy_by_model.png" in article


def test_blog_two_runs_produces_comparison_arrows_without_crashing(blog_env):
    _write_golden(blog_env["golden_path"], ["self-made"])
    _write_run(blog_env["runs_dir"], "before", ["qwen7b"])
    _write_run(blog_env["runs_dir"], "after", ["qwen7b"], variant="qwen7b_opt")

    out_dir = blog_mod.blog(run_ids=["before", "after"], slug="ab", config_path=REPO_ROOT / "config.yaml")
    assert (out_dir / "fig02_cost_vs_accuracy.png").exists()
    tables = (out_dir / "tables.md").read_text(encoding="utf-8")
    assert "before" in tables and "after" in tables


def test_blog_includes_fig03_when_taxonomy_defined(blog_env):
    _write_golden(blog_env["golden_path"], ["self-made"])
    _write_run(blog_env["runs_dir"], "run-1", ["qwen7b"])
    blog_env["taxonomy_path"].write_text(
        yaml.safe_dump({"categories": [{"id": "c1", "name": "カテゴリ1", "definition": "d"}], "assignments": {}}),
        encoding="utf-8",
    )
    # make case-0001 actually fail so the heatmap has something to plot
    run_dir = blog_env["runs_dir"] / "run-1"
    data = json.loads((run_dir / "output.json").read_text(encoding="utf-8"))
    data["results"]["results"][0]["gradingResult"]["pass"] = False
    data["results"]["results"][0]["success"] = False
    (run_dir / "output.json").write_text(json.dumps(data), encoding="utf-8")

    out_dir = blog_mod.blog(run_ids=["run-1"], slug="withfig3", config_path=REPO_ROOT / "config.yaml")
    assert (out_dir / "fig03_failure_heatmap.png").exists()


def test_blog_rejects_more_than_two_runs(blog_env):
    _write_golden(blog_env["golden_path"], ["self-made"])
    with pytest.raises(blog_mod.BlogGuardError):
        blog_mod.blog(run_ids=["a", "b", "c"], config_path=REPO_ROOT / "config.yaml")


def test_blog_rerun_same_slug_regenerates_not_accumulates(blog_env):
    _write_golden(blog_env["golden_path"], ["self-made"])
    _write_run(blog_env["runs_dir"], "run-1", ["qwen7b"])

    out_dir_1 = blog_mod.blog(run_ids=["run-1"], slug="dup", config_path=REPO_ROOT / "config.yaml")
    (out_dir_1 / "stale_extra_file.txt").write_text("should be gone after regen", encoding="utf-8")

    out_dir_2 = blog_mod.blog(run_ids=["run-1"], slug="dup", config_path=REPO_ROOT / "config.yaml")
    assert out_dir_1 == out_dir_2
    assert not (out_dir_2 / "stale_extra_file.txt").exists()
