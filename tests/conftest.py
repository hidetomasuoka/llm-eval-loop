"""Shared fixtures/helpers for the multi-task workspace test suite (issue #47).

`isolated_root` gives each test a throwaway repo-shaped root; `scaffold_task`
writes a minimal but complete task workspace (config.yaml + tasks/<name>/)
under such a root and resolves it through the real schemas.load_task().
"""

import json
import shutil
import uuid

import pytest
import yaml

from evalloop.paths import REPO_ROOT
from evalloop.schemas import load_task

DEFAULT_LABELS = ["契約照会", "障害報告", "機能要望", "その他"]

# mirrors the shape of the real global config.yaml models[] registry
DEFAULT_GLOBAL_MODELS = [
    {
        "provider": "ollama:chat:qwen2.5:7b",
        "alias": "qwen7b",
        "tier": "local",
        "price_in_per_mtok": 0.0,
        "price_out_per_mtok": 0.0,
    },
    {
        "provider": "anthropic:messages:claude-haiku-4-5-20251001",
        "alias": "haiku45",
        "tier": "small",
        "price_in_per_mtok": 1.0,
        "price_out_per_mtok": 5.0,
    },
]


@pytest.fixture
def isolated_root():
    """A throwaway root directory for TaskPaths(root=..., task=...).

    CRITICAL: the isolated tree lives INSIDE the repo (.pytest_isolated/,
    gitignored, rmtree'd in teardown) rather than in pytest's tmp_path:
    build/optimize compute os.path.relpath between artifact dirs and repo
    files (e.g. src/evalloop/asserts/label_match.js), and on GitHub's Windows
    runners the workspace (D:) and temp (C:) are different drives, where a
    cross-drive relpath raises ValueError.
    """
    root = REPO_ROOT / ".pytest_isolated" / uuid.uuid4().hex[:12]
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(autouse=True)
def disable_external_token_counting(monkeypatch):
    """Unit tests never call provider APIs even when the developer has a key."""
    monkeypatch.setenv("EVALLOOP_TOKEN_COUNT_API", "off")


def default_golden_rows(labels=None, n_train=4, n_test=4):
    labels = labels or DEFAULT_LABELS
    rows = []
    for i in range(n_train):
        rows.append(
            {
                "id": f"case-{i + 1:04d}",
                "input": f"問い合わせ文サンプル{i + 1}",
                "expected": labels[i % len(labels)],
                "split": "train",
                "meta": {"category": "基本", "source": "self-made"},
            }
        )
    for i in range(n_test):
        rows.append(
            {
                "id": f"case-{i + 101:04d}",
                "input": f"問い合わせ文サンプル{i + 101}",
                "expected": labels[i % len(labels)],
                "split": "test",
                "meta": {"category": "基本", "source": "self-made"},
            }
        )
    return rows


def scaffold_task(
    root,
    name="t1",
    answer_type="label",
    labels=None,
    golden_rows=None,
    models=None,  # alias subset for task.yaml models: (None = omit key = all global models)
    judge_provider="anthropic:messages:claude-sonnet-4-6",
    prompt="問い合わせ文:\n{{input}}\n",
    rubric=None,
    run_overrides=None,
    global_models=None,
    global_run=None,
    default_task=None,
    optimize_target="qwen7b",
    reflection_provider="anthropic/claude-opus-4-8",
    blog=None,
):
    """Write <root>/config.yaml + tasks/<name>/{task.yaml,golden.jsonl,prompts/}
    and return (Config, TaskPaths) via the real schemas.load_task().
    """
    if labels is None:
        labels = list(DEFAULT_LABELS) if answer_type == "label" else []
    if golden_rows is None:
        golden_rows = default_golden_rows(labels or None)

    global_raw = {
        "default_task": default_task if default_task is not None else name,
        "models": global_models or DEFAULT_GLOBAL_MODELS,
    }
    if global_run:
        global_raw["run"] = global_run
    (root / "config.yaml").write_text(
        yaml.safe_dump(global_raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    task_dir = root / "tasks" / name
    (task_dir / "prompts").mkdir(parents=True, exist_ok=True)

    task_raw = {
        "task": {"answer_type": answer_type, "labels": labels, "json_schema_file": None},
        "judge": {"provider": judge_provider, "threshold": 0.8, "agreement_threshold": 0.85},
        "optimize": {"target_alias": optimize_target, "reflection_provider": reflection_provider, "auto": "light"},
    }
    if models is not None:
        task_raw["models"] = models
    if run_overrides:
        task_raw["run"] = run_overrides
    if blog:
        task_raw["blog"] = blog
    (task_dir / "task.yaml").write_text(
        yaml.safe_dump(task_raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    with (task_dir / "golden.jsonl").open("w", encoding="utf-8") as f:
        for row in golden_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (task_dir / "prompts" / "task.txt").write_text(prompt, encoding="utf-8")
    if rubric is not None:
        (task_dir / "prompts" / "judge_rubric.txt").write_text(rubric, encoding="utf-8")

    return load_task(name, root=root)
