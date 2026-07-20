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
    def demos(self) -> Path:
        """Optional few-shot demos (APO-16). Usually gitignored; see PROVENANCE.md."""
        return self.task_dir / "demos.jsonl"

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

    @property
    def optimized_index(self) -> Path:
        """Append-only ledger of optimize variants (slug, summary, run links)."""
        return self.optimized_dir / "index.jsonl"

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
    def tests_dev(self) -> Path:
        """3-way split (improvement plan #4): dev is the optimize shipping-gate
        holdout; test stays reserved for the final confirmation run."""
        return self.build_dir / "tests_dev.yaml"

    @property
    def resolved_prompt(self) -> Path:
        """Build-time prompt with ``{{demos}}`` expanded (gitignored under data/build/)."""
        return self.build_dir / "prompt.resolved.txt"

    @property
    def promptfoo_dir(self) -> Path:
        return self.root / "promptfoo" / self.task

    @property
    def promptfoo_config(self) -> Path:
        return self.promptfoo_dir / "promptfooconfig.yaml"

    @property
    def promptfoo_config_dev(self) -> Path:
        """Same providers/prompt/grading as promptfoo_config but tests point at
        tests_dev.yaml. Only written when golden.jsonl has split=='dev' cases."""
        return self.promptfoo_dir / "promptfooconfig.dev.yaml"

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
    def calibration(self) -> Path:
        """Task-level judge calibration snapshot written by ``evalloop calibrate``."""
        return self.results_dir / "calibration.json"

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
        raise TaskNotFoundError(f"task {name!r} not found: {paths.task_config} does not exist. Known tasks: {known}")
    return paths


def list_tasks(root: Path = REPO_ROOT) -> list[str]:
    tasks_dir = root / "tasks"
    if not tasks_dir.exists():
        return []
    return sorted(d.name for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.yaml").exists())


class TaskExistsError(RuntimeError):
    pass


_TASK_YAML_TEMPLATE = """\
# =============================================================================
# {name} タスク設定（`evalloop task init` が生成した雛形）
#
# データ（golden.jsonl / human_labels.jsonl 等）は git 管理外（issue #47 の
# データポリシー）。出典・再取得手順は PROVENANCE.md に記録すること。
# プロンプトは prompts/task.txt（textタスクは prompts/judge_rubric.txt も）に規約固定。
# =============================================================================

task:
  answer_type: {answer_type}
{labels_block}  json_schema_file: null

# グローバル registry（config.yaml）からの alias 選択。省略時 = 全モデル
# models: [qwen7b, haiku45]

judge:
  # answer_type=text（llm-rubric）のときの grader。label/json では実際には
  # 呼ばれないが、スキーマ上必須のため明示しておく
  provider: anthropic:messages:claude-sonnet-4-6
  threshold: 0.8
  agreement_threshold: 0.85

optimize:
  target_alias: qwen7b                               # GEPAで最適化する対象モデル
  reflection_provider: anthropic/claude-opus-4-8      # dspy側の表記
  auto: light

blog:
  jpy_per_usd: 150
  slug_prefix: llm-eval
  allowed_sources: ["self-made"]   # 公開ガードが許可する meta.source の値
"""

_TASK_PROMPT_TEMPLATE = """\
ここにタスクの指示を書く（このファイルはgit追跡される）。
出力形式の指定まで含めて、モデルに与える指示のすべてをここに書くこと。

入力:
{{input}}
"""

_RUBRIC_TEMPLATE = """\
ここに llm-rubric ジャッジ用の採点基準を書く。
{{input}} と {{expected}} のプレースホルダは promptfoo が実行時に置換する。

入力: {{input}}
期待される答え: {{expected}}
"""

_PROVENANCE_TEMPLATE = """\
# {name} — データ出自と再取得手順

このタスクのデータ（`golden.jsonl` 等）は **git 管理外**（issue #47 のデータポリシー:
タスクデータは既定でコミット禁止）。以下を必ず埋めること。

## 出典

- **Source**: （配布元・URL・アーカイブ名）
- **License**: （ライセンスと、公開時の制約があれば明記）
- **Retrieved**: （取得日と取得方法）

## サンプリング方法（再現用）

（元データからどう抽出したか。乱数シード・件数・train/test分割を再現可能に書く）

## ファイル指紋（検証用）

- `golden.jsonl` sha256: （`python -c "import hashlib;print(hashlib.sha256(open('tasks/{name}/golden.jsonl','rb').read()).hexdigest())"`）

## 再取得

（データを失ったとき、上記だけで復元できる手順）
"""


def init_task_workspace(name: str, root: Path = REPO_ROOT, answer_type: str = "label") -> TaskPaths:
    """Scaffold tasks/<name>/ (task.yaml + prompts/ + PROVENANCE.md).

    golden.jsonl is deliberately NOT created: the data policy keeps it out of
    git, and an empty file would only defer the real error from build time.
    """
    validate_task_name(name)
    # keep in sync with schemas.VALID_ANSWER_TYPES (importing it here would be circular)
    if answer_type not in {"label", "json", "text"}:
        raise ValueError(f"unknown answer_type {answer_type!r} (expected label/json/text)")
    paths = TaskPaths(root=root, task=name)
    # duplicate = a task.yaml exists, not merely the directory: an empty dir or
    # a half-written scaffold isn't a task (list_tasks won't show it either),
    # and refusing to re-run init there would leave it unrecoverable (issue #55)
    if paths.task_config.exists():
        raise TaskExistsError(f"task {name!r} already exists: {paths.task_config}")

    (paths.task_dir / "prompts").mkdir(parents=True, exist_ok=True)
    labels_block = (
        '  labels: ["ラベルA", "ラベルB"]   # answer_type=label では必須。実際のラベルに置き換えること\n'
        if answer_type == "label"
        else "  labels: []\n"
    )
    paths.task_config.write_text(
        _TASK_YAML_TEMPLATE.format(name=name, answer_type=answer_type, labels_block=labels_block),
        encoding="utf-8",
    )
    paths.prompt_file.write_text(_TASK_PROMPT_TEMPLATE, encoding="utf-8")
    if answer_type == "text":
        paths.rubric_file.write_text(_RUBRIC_TEMPLATE, encoding="utf-8")
    (paths.task_dir / "PROVENANCE.md").write_text(_PROVENANCE_TEMPLATE.format(name=name), encoding="utf-8")
    return paths
