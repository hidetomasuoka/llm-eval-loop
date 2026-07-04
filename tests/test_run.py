import json

import pytest

from evalloop import run as run_mod

REPO_ROOT = run_mod.REPO_ROOT


class _FakeCompletedProcess:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(run_mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_mod, "RUNS_DIR", tmp_path / "results" / "runs")
    monkeypatch.setattr(run_mod, "INDEX_PATH", tmp_path / "results" / "index.jsonl")
    monkeypatch.setattr(run_mod, "get_promptfoo_version", lambda: "0.0.0-test")
    monkeypatch.setattr(run_mod, "get_node_version", lambda: "v22.22.0")


def test_npx_base_cmd_uses_pinned_version_not_latest(monkeypatch):
    # @latestはサプライチェーン露出＋再現性ドリフトのため禁止（issue #19）
    monkeypatch.setattr(run_mod.shutil, "which", lambda name: "/fake/npx")
    cmd = run_mod._npx_base_cmd()
    assert cmd == ["/fake/npx", f"promptfoo@{run_mod.PROMPTFOO_VERSION}"]
    assert "latest" not in cmd[1]
    # 固定値は具体的なバージョン番号であること（"latest"等のタグではない）
    assert run_mod.PROMPTFOO_VERSION[0].isdigit()


def test_run_missing_promptfoo_config_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(run_mod, "PROMPTFOO_CONFIG_PATH", tmp_path / "nope.yaml")
    with pytest.raises(run_mod.RunError):
        run_mod.run(config_path=REPO_ROOT / "config.yaml")


def test_run_variant_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "VARIANTS_DIR", tmp_path / "variants")
    with pytest.raises(run_mod.RunError):
        run_mod.run(variant="does-not-exist", config_path=REPO_ROOT / "config.yaml")


def test_run_total_failure_still_records_ledger_entry(tmp_path, monkeypatch):
    """Iron rule #3: even a run that produces no output.json must leave an
    audit trail (meta.json + index.jsonl), not a silent orphaned directory.
    """
    _patch_dirs(monkeypatch, tmp_path)

    def fake_eval(config_path, output_path, **kwargs):
        return _FakeCompletedProcess(returncode=1, stdout="", stderr="promptfoo boom")

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    with pytest.raises(run_mod.RunError, match="boom"):
        run_mod.run(config_path=REPO_ROOT / "config.yaml")

    index_lines = (tmp_path / "results" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 1
    entry = json.loads(index_lines[0])
    assert entry["promptfoo_exit_code"] == 1
    assert entry["actual_cost_usd"] == 0.0

    run_dirs = list((tmp_path / "results" / "runs").iterdir())
    assert len(run_dirs) == 1
    meta = json.loads((run_dirs[0] / "meta.json").read_text(encoding="utf-8"))
    assert meta["promptfoo_exit_code"] == 1
    assert "boom" in meta["promptfoo_stderr_tail"]
    assert not (run_dirs[0] / "output.json").exists()


def test_run_success_records_cost_and_returns_outcome(tmp_path, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)

    def fake_eval(config_path, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "results": {
                        "results": [
                            {
                                "vars": {"case_id": "case-0001", "expected": "契約照会"},
                                "provider": {"id": "p", "label": "haiku45"},
                                "response": {"output": "契約照会"},
                                "gradingResult": {"pass": True, "score": 1},
                                "success": True,
                                "cost": 0.002,
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    outcome = run_mod.run(config_path=REPO_ROOT / "config.yaml", limit=1)

    assert outcome.output_path.exists()
    assert outcome.meta["actual_cost_usd"] == pytest.approx(0.002)
    index_lines = (tmp_path / "results" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 1
