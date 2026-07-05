import shutil
import uuid

import pytest

from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import report as report_mod
from evalloop import run as run_mod


@pytest.fixture
def isolated_artifact_paths(monkeypatch):
    """Redirect every artifact path that build()/optimize()/run()/report()
    write through to an isolated throwaway tree. Without this, tests that
    exercise the real orchestration pollute the developer's checkout (and CI
    workspace): entries appended to results/index.jsonl, junk results/runs/
    and prompts/optimized/ dirs, and promptfoo/promptfooconfig.yaml silently
    replaced -- which also makes a second consecutive pytest run non-idempotent.

    The isolated tree lives INSIDE the repo (.pytest_isolated/, gitignored)
    rather than in pytest's tmp_path: build/optimize compute os.path.relpath
    between these dirs and repo files like prompts/base/task.txt, and on
    GitHub's Windows runners the workspace (D:) and temp (C:) are different
    drives, where a cross-drive relpath raises ValueError.
    """
    root = build_mod.REPO_ROOT / ".pytest_isolated" / uuid.uuid4().hex[:12]
    build_dir = root / "data" / "build"
    promptfoo_dir = root / "promptfoo"
    results_dir = root / "results"

    monkeypatch.setattr(build_mod, "BUILD_DIR", build_dir)
    monkeypatch.setattr(build_mod, "TESTS_TEST_PATH", build_dir / "tests_test.yaml")
    monkeypatch.setattr(build_mod, "TESTS_TRAIN_PATH", build_dir / "tests_train.yaml")
    monkeypatch.setattr(build_mod, "PROMPTFOO_DIR", promptfoo_dir)
    monkeypatch.setattr(build_mod, "PROMPTFOO_CONFIG_PATH", promptfoo_dir / "promptfooconfig.yaml")
    monkeypatch.setattr(run_mod, "PROMPTFOO_CONFIG_PATH", promptfoo_dir / "promptfooconfig.yaml")
    monkeypatch.setattr(run_mod, "VARIANTS_DIR", promptfoo_dir / "variants")
    monkeypatch.setattr(run_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(run_mod, "RUNS_DIR", results_dir / "runs")
    monkeypatch.setattr(run_mod, "INDEX_PATH", results_dir / "index.jsonl")
    monkeypatch.setattr(report_mod, "RUNS_DIR", results_dir / "runs")
    monkeypatch.setattr(report_mod, "REPORTS_DIR", results_dir / "reports")
    monkeypatch.setattr(optimize_mod, "OPTIMIZED_DIR", root / "prompts" / "optimized")
    monkeypatch.setattr(optimize_mod, "VARIANTS_DIR", promptfoo_dir / "variants")

    yield root

    shutil.rmtree(root, ignore_errors=True)
