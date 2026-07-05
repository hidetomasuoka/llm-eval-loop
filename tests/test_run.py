import hashlib
import json

import pytest

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


class _FakeCompletedProcess:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_config():
    return Config(
        task=TaskConfig(
            name="t1",
            answer_type="label",
            # must point at a real, tracked file: run() records its sha256
            prompt_file="tasks/sample-inquiry/prompts/task.txt",
            labels=["契約照会", "障害報告", "機能要望", "その他"],
        ),
        models=[
            ModelConfig(provider="anthropic:messages:claude-haiku-4-5-20251001", alias="haiku45", tier="small"),
        ],
        run=RunConfig(),
        judge=JudgeConfig(provider="anthropic:messages:claude-sonnet-4-6"),
        optimize=OptimizeConfig(target_alias="haiku45", reflection_provider="r"),
        blog=BlogConfig(),
        path=REPO_ROOT / "config.yaml",
    )


def test_run_meta_models_reflect_built_provider_subset(isolated_root, monkeypatch):
    """`build --models` narrows the built promptfoo config; meta.json must list
    only what was actually evaluated, not the full task config (issue #49)."""
    paths = TaskPaths(root=isolated_root, task="t1")
    cfg = _make_config()
    cfg.models.append(ModelConfig(provider="ollama:chat:qwen2.5:7b", alias="qwen7b", tier="local"))
    _prepare_env(monkeypatch, paths)
    # the built artifact contains only haiku45, as `build --models haiku45` would emit
    paths.promptfoo_config.write_text(
        "providers:\n  - id: anthropic:messages:claude-haiku-4-5-20251001\n    label: haiku45\n",
        encoding="utf-8",
    )

    def fake_eval(config_path, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": {"results": []}}), encoding="utf-8")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    outcome = run_mod.run(cfg, paths)

    assert [m["alias"] for m in outcome.meta["models"]] == ["haiku45"]


def _prepare_env(monkeypatch, paths):
    monkeypatch.setattr(run_mod, "get_promptfoo_version", lambda: "0.0.0-test")
    monkeypatch.setattr(run_mod, "get_node_version", lambda: "v22.22.0")
    # run() refuses to start without a built promptfoo config. Create a dummy
    # one in the isolated tree instead of the real (gitignored, generated)
    # promptfoo/<task>/promptfooconfig.yaml -- on a fresh CI clone it doesn't exist.
    paths.promptfoo_config.parent.mkdir(parents=True, exist_ok=True)
    paths.promptfoo_config.write_text("providers: []\n", encoding="utf-8")


def test_npx_base_cmd_uses_pinned_version_not_latest(monkeypatch):
    # @latestはサプライチェーン露出＋再現性ドリフトのため禁止（issue #19）
    monkeypatch.setattr(run_mod.shutil, "which", lambda name: "/fake/npx")
    cmd = run_mod._npx_base_cmd()
    assert cmd == ["/fake/npx", f"promptfoo@{run_mod.PROMPTFOO_VERSION}"]
    assert "latest" not in cmd[1]
    # 固定値は具体的なバージョン番号であること（"latest"等のタグではない）
    assert run_mod.PROMPTFOO_VERSION[0].isdigit()


def test_run_missing_promptfoo_config_raises(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")  # nothing built here
    with pytest.raises(run_mod.RunError):
        run_mod.run(_make_config(), paths)


def test_run_variant_missing_raises(isolated_root):
    paths = TaskPaths(root=isolated_root, task="t1")
    with pytest.raises(run_mod.RunError):
        run_mod.run(_make_config(), paths, variant="does-not-exist")


def test_run_total_failure_still_records_ledger_entry(isolated_root, monkeypatch):
    """Iron rule #3: even a run that produces no output.json must leave an
    audit trail (meta.json + index.jsonl), not a silent orphaned directory.
    """
    paths = TaskPaths(root=isolated_root, task="t1")
    _prepare_env(monkeypatch, paths)

    def fake_eval(config_path, output_path, **kwargs):
        return _FakeCompletedProcess(returncode=1, stdout="", stderr="promptfoo boom")

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    with pytest.raises(run_mod.RunError, match="boom"):
        run_mod.run(_make_config(), paths)

    index_lines = paths.index.read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 1
    entry = json.loads(index_lines[0])
    assert entry["promptfoo_exit_code"] == 1
    assert entry["actual_cost_usd"] == 0.0
    assert entry["task"] == "t1"

    run_dirs = list(paths.runs_dir.iterdir())
    assert len(run_dirs) == 1
    meta = json.loads((run_dirs[0] / "meta.json").read_text(encoding="utf-8"))
    assert meta["promptfoo_exit_code"] == 1
    assert "boom" in meta["promptfoo_stderr_tail"]
    assert meta["task"] == "t1"
    # no golden.jsonl in the isolated tree -> dataset hash recorded as null
    assert meta["golden_sha256"] is None
    assert not (run_dirs[0] / "output.json").exists()


def test_run_success_records_cost_and_returns_outcome(isolated_root, monkeypatch):
    paths = TaskPaths(root=isolated_root, task="t1")
    _prepare_env(monkeypatch, paths)
    # dataset-version reproducibility (issue #47): meta must hash the golden
    golden_bytes = b'{"id": "case-0001"}\n'
    paths.golden.parent.mkdir(parents=True, exist_ok=True)
    paths.golden.write_bytes(golden_bytes)

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

    outcome = run_mod.run(_make_config(), paths, limit=1)

    assert outcome.output_path.exists()
    assert outcome.meta["actual_cost_usd"] == pytest.approx(0.002)
    assert outcome.meta["task"] == "t1"
    assert outcome.meta["golden_sha256"] == hashlib.sha256(golden_bytes).hexdigest()
    # reproduce command is task-scoped now
    assert "--task t1" in outcome.meta["evalloop_command"]
    index_lines = paths.index.read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 1
    assert json.loads(index_lines[0])["task"] == "t1"
