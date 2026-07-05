import csv
import json

import pytest
import yaml

from evalloop import analyze as analyze_mod
from evalloop import run as run_mod
from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import (
    BlogConfig,
    Config,
    JudgeConfig,
    ModelConfig,
    OptimizeConfig,
    RunConfig,
    TaskConfig,
)


def _make_config(judge_provider="anthropic:messages:claude-sonnet-4-6", models=None):
    return Config(
        task=TaskConfig(name="t1", answer_type="text", prompt_file="tasks/sample-inquiry/prompts/task.txt"),
        models=models
        or [
            ModelConfig(provider="ollama:chat:qwen2.5:7b", alias="qwen7b", tier="local"),
            ModelConfig(provider="anthropic:messages:claude-haiku-4-5-20251001", alias="haiku45", tier="small"),
        ],
        run=RunConfig(),
        judge=JudgeConfig(provider=judge_provider),
        optimize=OptimizeConfig(target_alias="qwen7b", reflection_provider="r"),
        blog=BlogConfig(),
        path=REPO_ROOT / "config.yaml",
    )


@pytest.fixture
def analyze_env(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    # cluster()/pivot() write into the task workspace (taxonomy draft, notes);
    # the fixture pre-creates the task dir like a real scaffolded task would
    paths.task_dir.mkdir(parents=True)
    return paths


def _write_run_output(paths, run_id, rows):
    run_dir = paths.runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "output.json").write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")


def _row(case_id, alias, passed, category="基本", output="x", error=None):
    return {
        "vars": {"case_id": case_id, "expected": "契約照会", "category": category},
        "provider": {"id": "p", "label": alias},
        "response": {"output": output},
        "gradingResult": {"pass": passed, "score": 1 if passed else 0, "reason": "why"},
        "success": passed,
        "error": error,
    }


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------


def test_failures_writes_jsonl_and_notes_csv(analyze_env):
    _write_run_output(
        analyze_env,
        "run-1",
        [
            _row("case-0001", "haiku45", True),
            _row("case-0002", "haiku45", False),
            _row("case-0003", "qwen7b", False),
        ],
    )

    failures_path, notes_path = analyze_mod.failures("run-1", analyze_env)

    assert notes_path == analyze_env.notes
    lines = failures_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    ids = {json.loads(line)["case_id"] for line in lines}
    assert ids == {"case-0002", "case-0003"}

    with notes_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["case_id"] for r in rows} == {"case-0002", "case-0003"}
    assert all(r["note"] == "" for r in rows)


def test_failures_includes_errored_rows(analyze_env):
    errored_row = _row("case-0005", "haiku45", passed=None, error="rate limited")
    errored_row["gradingResult"] = {}
    errored_row["success"] = None
    _write_run_output(analyze_env, "run-1", [errored_row])

    failures_path, _ = analyze_mod.failures("run-1", analyze_env)

    lines = failures_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["case_id"] == "case-0005"
    assert entry["error"] == "rate limited"


def test_failures_is_idempotent_no_duplicate_notes_rows(analyze_env):
    _write_run_output(analyze_env, "run-1", [_row("case-0002", "haiku45", False)])
    analyze_mod.failures("run-1", analyze_env)

    # hand-annotate the note column, like a human would
    notes_path = analyze_env.notes
    with notes_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["note"] = "typo in label"
    with notes_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=analyze_mod.NOTES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # re-running failures for the same run must not duplicate or clobber the hand note
    analyze_mod.failures("run-1", analyze_env)

    with notes_path.open(encoding="utf-8", newline="") as f:
        rows_after = list(csv.DictReader(f))
    assert len(rows_after) == 1
    assert rows_after[0]["note"] == "typo in label"


def test_failures_missing_run_raises(analyze_env):
    with pytest.raises(analyze_mod.AnalyzeError):
        analyze_mod.failures("does-not-exist", analyze_env)


def test_failures_fills_input_head_from_golden(analyze_env):
    long_input = "本契約は解約可能である。" * 30  # far longer than INPUT_HEAD_LEN
    with analyze_env.golden.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": "case-0002",
                    "input": long_input,
                    "expected": "契約照会",
                    "split": "test",
                    "meta": {"category": "基本", "source": "self-made"},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    _write_run_output(
        analyze_env,
        "run-1",
        [
            _row("case-0002", "haiku45", False),
            _row("case-9999", "haiku45", False),  # not in golden.jsonl
        ],
    )

    _, notes_path = analyze_mod.failures("run-1", analyze_env)

    with notes_path.open(encoding="utf-8", newline="") as f:
        rows = {r["case_id"]: r for r in csv.DictReader(f)}
    assert rows["case-0002"]["input_head"] == long_input[: analyze_mod.INPUT_HEAD_LEN] + "..."
    assert rows["case-9999"]["input_head"] == ""


def test_failures_tolerates_missing_golden(analyze_env):
    # analyze_env leaves golden.jsonl missing; triage must still work
    _write_run_output(analyze_env, "run-1", [_row("case-0002", "haiku45", False)])

    _, notes_path = analyze_mod.failures("run-1", analyze_env)

    with notes_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["input_head"] == ""


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------


def test_cluster_writes_draft_never_touches_taxonomy_yaml(analyze_env, monkeypatch):
    analyze_env.notes.write_text(
        "case_id,model,input_head,output_head,expected,note\n"
        "case-0002,haiku45,foo,bar,契約照会,label swap\n",
        encoding="utf-8",
    )
    analyze_env.taxonomy.write_text("categories: []\nassignments: {}\n", encoding="utf-8")

    fake_taxonomy = {
        "categories": [{"id": "label_swap", "name": "ラベル取り違え", "definition": "似たラベルを混同する"}],
        "assignments": {"case-0002": "label_swap"},
    }

    def fake_eval(config_path, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"results": {"results": [{"vars": {}, "provider": {"id": "j", "label": "cluster_judge"},
                                                    "response": {"output": json.dumps(fake_taxonomy, ensure_ascii=False)},
                                                    "gradingResult": {"pass": True, "score": 1}, "success": True}]}}),
            encoding="utf-8",
        )

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    draft_path = analyze_mod.cluster(_make_config(), analyze_env, notes_path=analyze_env.notes)

    assert draft_path == analyze_env.taxonomy_draft
    draft = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
    assert draft["categories"][0]["id"] == "label_swap"
    # taxonomy.yaml (the real, human-merged file) must be untouched
    real = yaml.safe_load(analyze_env.taxonomy.read_text(encoding="utf-8"))
    assert real == {"categories": [], "assignments": {}}
    assert not list(analyze_env.promptfoo_dir.glob("_cluster_tmp.yaml"))


def test_cluster_missing_notes_raises(analyze_env):
    with pytest.raises(analyze_mod.AnalyzeError):
        analyze_mod.cluster(_make_config(), analyze_env)


def test_cluster_invalid_json_output_raises(analyze_env, monkeypatch):
    analyze_env.notes.write_text(
        "case_id,model,input_head,output_head,expected,note\ncase-0002,haiku45,foo,bar,契約照会,x\n",
        encoding="utf-8",
    )

    def fake_eval(config_path, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"results": {"results": [{"vars": {}, "provider": {"id": "j", "label": "cluster_judge"},
                                                    "response": {"output": "not json"},
                                                    "gradingResult": {"pass": False, "score": 0}, "success": False}]}}),
            encoding="utf-8",
        )

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    with pytest.raises(analyze_mod.AnalyzeError):
        analyze_mod.cluster(_make_config(), analyze_env, notes_path=analyze_env.notes)


def test_cluster_omits_temperature_when_judge_lacks_sampling_support(analyze_env, monkeypatch):
    # judge.provider also appears in models[] with supports_sampling_params:
    # false (opus48/fable5-style) -- the throwaway cluster eval must not send
    # temperature or the provider rejects it with HTTP 400
    analyze_env.notes.write_text(
        "case_id,model,input_head,output_head,expected,note\ncase-0002,haiku45,foo,bar,契約照会,x\n",
        encoding="utf-8",
    )
    cfg = _make_config(
        judge_provider="p:nosample",
        models=[ModelConfig(provider="p:nosample", alias="nosample", tier="frontier", supports_sampling_params=False)],
    )

    fake_taxonomy = {"categories": [{"id": "c1", "name": "c1", "definition": "d"}], "assignments": {"case-0002": "c1"}}
    captured = {}

    def fake_eval(cfg_path, output_path, **kwargs):
        captured["config"] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"results": {"results": [{"vars": {}, "provider": {"id": "j", "label": "cluster_judge"},
                                                    "response": {"output": json.dumps(fake_taxonomy, ensure_ascii=False)},
                                                    "gradingResult": {"pass": True, "score": 1}, "success": True}]}}),
            encoding="utf-8",
        )

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    analyze_mod.cluster(cfg, analyze_env, notes_path=analyze_env.notes)

    provider_config = captured["config"]["providers"][0]["config"]
    assert "temperature" not in provider_config
    assert provider_config == {"max_tokens": 2048}


# ---------------------------------------------------------------------------
# pivot
# ---------------------------------------------------------------------------


def test_pivot_cross_tab_with_unassigned_bucket(analyze_env):
    _write_run_output(
        analyze_env,
        "run-1",
        [
            _row("case-0001", "haiku45", False),
            _row("case-0002", "haiku45", False),
            _row("case-0003", "qwen7b", False),
            _row("case-0004", "qwen7b", True),  # passing row must be excluded
        ],
    )
    analyze_env.taxonomy.write_text(
        yaml.safe_dump(
            {
                "categories": [{"id": "label_swap", "name": "ラベル取り違え", "definition": "d"}],
                "assignments": {"case-0001": "label_swap", "case-0002": "label_swap"},
                # case-0003 intentionally left unassigned
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    report_path = analyze_mod.pivot("run-1", analyze_env)
    content = report_path.read_text(encoding="utf-8")

    assert report_path.parent == analyze_env.reports_dir
    assert "ラベル取り違え" in content
    assert "unassigned" in content.lower() or "未割当" in content
    assert "haiku45" in content and "qwen7b" in content
    # the footer must print the taxonomy path actually used, not None (issue #52)
    assert "`None`" not in content


def test_pivot_missing_taxonomy_raises(analyze_env):
    _write_run_output(analyze_env, "run-1", [_row("case-0001", "haiku45", False)])
    with pytest.raises(analyze_mod.AnalyzeError):
        analyze_mod.pivot("run-1", analyze_env)


def test_pivot_missing_run_raises(analyze_env):
    analyze_env.taxonomy.write_text("categories: []\nassignments: {}\n", encoding="utf-8")
    with pytest.raises(analyze_mod.AnalyzeError):
        analyze_mod.pivot("does-not-exist", analyze_env)


def test_load_taxonomy_requires_explicit_path(analyze_env):
    analyze_env.taxonomy.write_text("categories: []\n", encoding="utf-8")
    taxonomy = analyze_mod.load_taxonomy(analyze_env.taxonomy)
    assert taxonomy == {"categories": [], "assignments": {}}
    with pytest.raises(analyze_mod.AnalyzeError):
        analyze_mod.load_taxonomy(analyze_env.task_dir / "nope.yaml")
