import csv
import json

import pytest
import yaml

from evalloop import analyze as analyze_mod
from evalloop import build as build_mod
from evalloop import run as run_mod

REPO_ROOT = build_mod.REPO_ROOT


@pytest.fixture
def analyze_env(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    notes_path = tmp_path / "notes.csv"
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_draft_path = tmp_path / "taxonomy.draft.yaml"
    reports_dir = tmp_path / "reports"
    golden_path = tmp_path / "golden.jsonl"  # left missing unless a test writes it

    monkeypatch.setattr(run_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(analyze_mod, "NOTES_PATH", notes_path)
    monkeypatch.setattr(analyze_mod, "TAXONOMY_PATH", taxonomy_path)
    monkeypatch.setattr(analyze_mod, "TAXONOMY_DRAFT_PATH", taxonomy_draft_path)
    monkeypatch.setattr(analyze_mod, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(build_mod, "GOLDEN_PATH", golden_path)

    return {
        "runs_dir": runs_dir,
        "notes_path": notes_path,
        "taxonomy_path": taxonomy_path,
        "taxonomy_draft_path": taxonomy_draft_path,
        "reports_dir": reports_dir,
        "golden_path": golden_path,
    }


def _write_run_output(runs_dir, run_id, rows):
    run_dir = runs_dir / run_id
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
        analyze_env["runs_dir"],
        "run-1",
        [
            _row("case-0001", "haiku45", True),
            _row("case-0002", "haiku45", False),
            _row("case-0003", "qwen7b", False),
        ],
    )

    failures_path, notes_path = analyze_mod.failures("run-1")

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
    _write_run_output(analyze_env["runs_dir"], "run-1", [errored_row])

    failures_path, _ = analyze_mod.failures("run-1")

    lines = failures_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["case_id"] == "case-0005"
    assert entry["error"] == "rate limited"


def test_failures_is_idempotent_no_duplicate_notes_rows(analyze_env):
    _write_run_output(analyze_env["runs_dir"], "run-1", [_row("case-0002", "haiku45", False)])
    analyze_mod.failures("run-1")

    # hand-annotate the note column, like a human would
    notes_path = analyze_env["notes_path"]
    with notes_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["note"] = "typo in label"
    with notes_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=analyze_mod.NOTES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # re-running failures for the same run must not duplicate or clobber the hand note
    analyze_mod.failures("run-1")

    with notes_path.open(encoding="utf-8", newline="") as f:
        rows_after = list(csv.DictReader(f))
    assert len(rows_after) == 1
    assert rows_after[0]["note"] == "typo in label"


def test_failures_missing_run_raises(analyze_env):
    with pytest.raises(analyze_mod.AnalyzeError):
        analyze_mod.failures("does-not-exist")


def test_failures_fills_input_head_from_golden(analyze_env):
    long_input = "本契約は解約可能である。" * 30  # far longer than INPUT_HEAD_LEN
    with analyze_env["golden_path"].open("w", encoding="utf-8") as f:
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
        analyze_env["runs_dir"],
        "run-1",
        [
            _row("case-0002", "haiku45", False),
            _row("case-9999", "haiku45", False),  # not in golden.jsonl
        ],
    )

    _, notes_path = analyze_mod.failures("run-1")

    with notes_path.open(encoding="utf-8", newline="") as f:
        rows = {r["case_id"]: r for r in csv.DictReader(f)}
    assert rows["case-0002"]["input_head"] == long_input[: analyze_mod.INPUT_HEAD_LEN] + "..."
    assert rows["case-9999"]["input_head"] == ""


def test_failures_tolerates_missing_golden(analyze_env):
    # analyze_env leaves golden_path missing; triage must still work
    _write_run_output(analyze_env["runs_dir"], "run-1", [_row("case-0002", "haiku45", False)])

    _, notes_path = analyze_mod.failures("run-1")

    with notes_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["input_head"] == ""


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------


def test_cluster_writes_draft_never_touches_taxonomy_yaml(analyze_env, monkeypatch):
    analyze_env["notes_path"].write_text(
        "case_id,model,input_head,output_head,expected,note\n"
        "case-0002,haiku45,foo,bar,契約照会,label swap\n",
        encoding="utf-8",
    )
    analyze_env["taxonomy_path"].write_text("categories: []\nassignments: {}\n", encoding="utf-8")

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

    draft_path = analyze_mod.cluster(notes_path=analyze_env["notes_path"], config_path=REPO_ROOT / "config.yaml")

    draft = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
    assert draft["categories"][0]["id"] == "label_swap"
    # taxonomy.yaml (the real, human-merged file) must be untouched
    real = yaml.safe_load(analyze_env["taxonomy_path"].read_text(encoding="utf-8"))
    assert real == {"categories": [], "assignments": {}}
    assert not list(build_mod.PROMPTFOO_DIR.glob("_cluster_tmp.yaml"))


def test_cluster_missing_notes_raises(analyze_env):
    with pytest.raises(analyze_mod.AnalyzeError):
        analyze_mod.cluster(notes_path=analyze_env["notes_path"], config_path=REPO_ROOT / "config.yaml")


def test_cluster_invalid_json_output_raises(analyze_env, monkeypatch):
    analyze_env["notes_path"].write_text(
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
        analyze_mod.cluster(notes_path=analyze_env["notes_path"], config_path=REPO_ROOT / "config.yaml")


def test_cluster_omits_temperature_when_judge_lacks_sampling_support(analyze_env, monkeypatch, tmp_path):
    # judge.provider also appears in models[] with supports_sampling_params:
    # false (opus48/fable5-style) -- the throwaway cluster eval must not send
    # temperature or the provider rejects it with HTTP 400
    analyze_env["notes_path"].write_text(
        "case_id,model,input_head,output_head,expected,note\ncase-0002,haiku45,foo,bar,契約照会,x\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "t", "answer_type": "text", "prompt_file": "prompts/base/task.txt"},
                "models": [
                    {"provider": "p:nosample", "alias": "nosample", "tier": "frontier", "supports_sampling_params": False}
                ],
                "judge": {"provider": "p:nosample"},
                "optimize": {"target_alias": "nosample", "reflection_provider": "r"},
            }
        ),
        encoding="utf-8",
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

    analyze_mod.cluster(notes_path=analyze_env["notes_path"], config_path=config_path)

    provider_config = captured["config"]["providers"][0]["config"]
    assert "temperature" not in provider_config
    assert provider_config == {"max_tokens": 2048}


# ---------------------------------------------------------------------------
# pivot
# ---------------------------------------------------------------------------


def test_pivot_cross_tab_with_unassigned_bucket(analyze_env):
    _write_run_output(
        analyze_env["runs_dir"],
        "run-1",
        [
            _row("case-0001", "haiku45", False),
            _row("case-0002", "haiku45", False),
            _row("case-0003", "qwen7b", False),
            _row("case-0004", "qwen7b", True),  # passing row must be excluded
        ],
    )
    analyze_env["taxonomy_path"].write_text(
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

    report_path = analyze_mod.pivot("run-1")
    content = report_path.read_text(encoding="utf-8")

    assert "ラベル取り違え" in content
    assert "unassigned" in content.lower() or "未割当" in content
    assert "haiku45" in content and "qwen7b" in content


def test_pivot_missing_taxonomy_raises(analyze_env):
    _write_run_output(analyze_env["runs_dir"], "run-1", [_row("case-0001", "haiku45", False)])
    with pytest.raises(analyze_mod.AnalyzeError):
        analyze_mod.pivot("run-1")


def test_pivot_missing_run_raises(analyze_env):
    analyze_env["taxonomy_path"].write_text("categories: []\nassignments: {}\n", encoding="utf-8")
    with pytest.raises(analyze_mod.AnalyzeError):
        analyze_mod.pivot("does-not-exist")
