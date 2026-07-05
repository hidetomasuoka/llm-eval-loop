"""Single source of truth for every task-scoped path (issue #47).

One task = one workspace directory under tasks/<name>/, plus per-task
generated-artifact subtrees (all gitignored). Modules never hold their own
path constants for task-scoped files -- they take a TaskPaths. Tests build a
TaskPaths rooted somewhere else (tests/conftest.py) and nothing needs to be
monkeypatched.

Data policy (issue #47 section 1.5): golden.jsonl / human_labels.jsonl /
notes.csv / taxonomy*.yaml are gitignored by default -- only task.yaml,
prompts/ and PROVENANCE.md are tracked. The synthetic sample-inquiry task is
the only opt-in tracked dataset (quickstart + CI smoke rely on it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# path safety: task names become directory components everywhere below
TASK_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


class TaskNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskPaths:
    root: Path  # repo root (or an isolated root under test)
    task: str

    # --- task workspace (tracked: task_config, prompts, provenance) ---------
    @property
    def task_dir(self) -> Path:
        return self.root / "tasks" / self.task

    @property
    def task_config(self) -> Path:
        return self.task_dir / "task.yaml"

    @property
    def golden(self) -> Path:
        return self.task_dir / "golden.jsonl"

    @property
    def prompt_file(self) -> Path:
        return self.task_dir / "prompts" / "task.txt"

    @property
    def rubric_file(self) -> Path:
        return self.task_dir / "prompts" / "judge_rubric.txt"

    @property
    def human_labels(self) -> Path:
        return self.task_dir / "human_labels.jsonl"

    @property
    def taxonomy(self) -> Path:
        return self.task_dir / "taxonomy.yaml"

    @property
    def taxonomy_draft(self) -> Path:
        return self.task_dir / "taxonomy.draft.yaml"

    @property
    def notes(self) -> Path:
        return self.task_dir / "notes.csv"

    @property
    def failures_notes_dir(self) -> Path:
        return self.task_dir

    @property
    def optimized_dir(self) -> Path:
        return self.task_dir / "optimized"

    # --- generated artifacts (all gitignored) --------------------------------
    @property
    def build_dir(self) -> Path:
        return self.root / "data" / "build" / self.task

    @property
    def tests_test(self) -> Path:
        return self.build_dir / "tests_test.yaml"

    @property
    def tests_train(self) -> Path:
        return self.build_dir / "tests_train.yaml"

    @property
    def promptfoo_dir(self) -> Path:
        return self.root / "promptfoo" / self.task

    @property
    def promptfoo_config(self) -> Path:
        return self.promptfoo_dir / "promptfooconfig.yaml"

    @property
    def variants_dir(self) -> Path:
        return self.promptfoo_dir / "variants"

    @property
    def results_dir(self) -> Path:
        return self.root / "results" / self.task

    @property
    def runs_dir(self) -> Path:
        return self.results_dir / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.results_dir / "reports"

    @property
    def index(self) -> Path:
        return self.results_dir / "index.jsonl"

    @property
    def blog_dir(self) -> Path:
        return self.root / "blog" / self.task


def validate_task_name(name: str) -> str:
    if not TASK_NAME_RE.match(name):
        raise TaskNotFoundError(
            f"invalid task name {name!r}: must match {TASK_NAME_RE.pattern} "
            "(lowercase alphanumerics and hyphens; it becomes a directory name)"
        )
    return name


def for_task(name: str, root: Path = REPO_ROOT) -> TaskPaths:
    validate_task_name(name)
    paths = TaskPaths(root=root, task=name)
    if not paths.task_config.exists():
        known = ", ".join(list_tasks(root)) or "(none)"
        raise TaskNotFoundError(
            f"task {name!r} not found: {paths.task_config} does not exist. Known tasks: {known}"
        )
    return paths


def list_tasks(root: Path = REPO_ROOT) -> list[str]:
    tasks_dir = root / "tasks"
    if not tasks_dir.exists():
        return []
    return sorted(d.name for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.yaml").exists())
