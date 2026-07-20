"""Interactive APO readiness checklist (symptom → granularity → method).

Pure text Q&A — no LLM calls. Content sourced from docs/APO_GUIDE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from rich.console import Console
from rich.prompt import Confirm, IntPrompt

console = Console()


class DiagnoseOutcome(str, Enum):
    DEFER_NON_PROMPT = "defer_non_prompt"
    OUT_OF_SCOPE = "out_of_scope"
    NEED_EVAL_SET = "need_eval_set"
    RECOMMEND_METHOD = "recommend_method"


@dataclass(frozen=True)
class SymptomChoice:
    key: int
    label: str
    granularity: str
    methods: tuple[str, ...]
    out_of_scope: bool = False


SYMPTOM_CHOICES: tuple[SymptomChoice, ...] = (
    SymptomChoice(
        1,
        "指示が曖昧で分類・抽出がぶれる",
        "7a. Instruction",
        ("gepa", "copro"),
    ),
    SymptomChoice(
        2,
        "例の入れ替え・順序で性能がぶれる",
        "7b. Exemplar",
        ("miprov2",),
    ),
    SymptomChoice(
        3,
        "長いsystem promptの局所修正で別セクションが壊れる",
        "7c. 長文構造",
        (),
        out_of_scope=True,
    ),
    SymptomChoice(
        4,
        "コスト・長さ制約が厳しい",
        "7d. 多目的",
        (),
        out_of_scope=True,
    ),
    SymptomChoice(
        5,
        "Agent軌跡が破綻",
        "7e. Agent/Multi-step",
        (),
        out_of_scope=True,
    ),
)

Q1_PROMPT = (
    "Q1: 症状はプロンプト以外（検索未ヒット / パース欠損 / ツール誤選択 / "
    "ワークフロー破綻等）が主因の可能性がありますか?"
)
DEFER_NON_PROMPT_MESSAGE = (
    "APO適用は保留。先にRAG / パース / ツール説明 / ワークフローを修正してください。（docs/APO_GUIDE.md 第1章参照）"
)
NEED_EVAL_SET_MESSAGE = (
    "train / holdout 分割が取れる評価セット整備が先です。"
    "APOは評価セットの上に成り立ちます。（docs/APO_GUIDE.md 第1章③参照）"
)

METHOD_SNIPPETS: dict[str, str] = {
    "gepa": """optimize:
  method: gepa
  auto: light
  # params: {}  # params.auto が上の auto より優先される""",
    "copro": """optimize:
  method: copro
  params:
    breadth: 10
    depth: 3
    init_temperature: 1.4""",
    "miprov2": """optimize:
  method: miprov2
  params:
    max_bootstrapped_demos: 4
    max_labeled_demos: 4
    val_ratio: 0.2
    seed: 0""",
}


def _symptom_by_key(key: int) -> SymptomChoice:
    for choice in SYMPTOM_CHOICES:
        if choice.key == key:
            return choice
    raise ValueError(f"symptom choice must be 1-{len(SYMPTOM_CHOICES)}, got {key}")


def _format_methods(methods: tuple[str, ...]) -> str:
    if not methods:
        return "対象外"
    return " / ".join(methods)


def _print_symptom_menu() -> None:
    console.print("\nQ2: 症状の種類を選択してください:")
    for choice in SYMPTOM_CHOICES:
        console.print(f"  {choice.key}. {choice.label}")


def _print_symptom_result(symptom: SymptomChoice) -> None:
    console.print(f"\n粒度: [bold]{symptom.granularity}[/bold]")
    console.print(f"推奨 method: [bold]{_format_methods(symptom.methods)}[/bold]")
    if symptom.out_of_scope:
        console.print(
            "[yellow]evalloop optimize では未対応の粒度です。根本原因の設計見直しを先に検討してください。[/yellow]"
        )


def _print_method_snippet(method: str) -> None:
    console.print("\n推奨 task.yaml スニペット（optimize 節）:")
    console.print(METHOD_SNIPPETS[method])


def _ask_q1_non_prompt_cause(*, answer: int | None, ask_confirm: Callable[..., bool]) -> bool:
    if answer is not None:
        if answer not in (1, 2):
            raise ValueError("Q1 answer must be 1 (yes) or 2 (no)")
        return answer == 1
    return ask_confirm(Q1_PROMPT, default=False)


def _ask_q2_symptom(*, answer: int | None, ask_int: Callable[..., int]) -> SymptomChoice:
    _print_symptom_menu()
    if answer is not None:
        return _symptom_by_key(answer)
    choice = ask_int(
        "番号",
        choices=[str(c.key) for c in SYMPTOM_CHOICES],
        show_choices=False,
    )
    return _symptom_by_key(choice)


def _ask_q3_split_available(*, answer: int | None, ask_confirm: Callable[..., bool]) -> bool:
    prompt = "Q3: train / holdout 分割が取れる評価セットはありますか?"
    if answer is not None:
        if answer not in (1, 2):
            raise ValueError("Q3 answer must be 1 (yes) or 2 (no)")
        return answer == 1
    return ask_confirm(prompt, default=True)


def _recommended_method(symptom: SymptomChoice) -> str:
    return symptom.methods[0]


def run_diagnose(
    *,
    answers: list[int] | None = None,
    ask_confirm: Callable[..., bool] | None = None,
    ask_int: Callable[..., int] | None = None,
) -> DiagnoseOutcome:
    """Run the symptom → granularity → method checklist.

    ``answers`` supplies fixed replies for tests: [Q1, Q2?, Q3?] where
    Q1/Q3 use 1=yes / 2=no and Q2 uses symptom keys 1-5.
    """
    ask_confirm = ask_confirm or Confirm.ask
    ask_int = ask_int or IntPrompt.ask

    q1 = answers[0] if answers else None
    q2 = answers[1] if answers and len(answers) > 1 else None
    q3 = answers[2] if answers and len(answers) > 2 else None

    console.print("[bold]APO 適用診断[/bold]（docs/APO_GUIDE.md の3段階フロー）\n")

    if _ask_q1_non_prompt_cause(answer=q1, ask_confirm=ask_confirm):
        console.print(f"\n[bold yellow]{DEFER_NON_PROMPT_MESSAGE}[/bold yellow]")
        return DiagnoseOutcome.DEFER_NON_PROMPT

    symptom = _ask_q2_symptom(answer=q2, ask_int=ask_int)
    _print_symptom_result(symptom)

    if symptom.out_of_scope:
        return DiagnoseOutcome.OUT_OF_SCOPE

    if not _ask_q3_split_available(answer=q3, ask_confirm=ask_confirm):
        console.print(f"\n[bold yellow]{NEED_EVAL_SET_MESSAGE}[/bold yellow]")
        return DiagnoseOutcome.NEED_EVAL_SET

    method = _recommended_method(symptom)
    _print_method_snippet(method)
    if len(symptom.methods) > 1:
        alternatives = ", ".join(symptom.methods[1:])
        console.print(f"\n代替候補: {alternatives}（task.yaml の optimize.method を変更）")
    # Q3 already confirmed train/holdout exists — do not re-ask for eval-set setup.
    console.print("\n次のステップ: task.yaml に上記 optimize 設定を反映 → evalloop optimize")
    return DiagnoseOutcome.RECOMMEND_METHOD


def parse_answers(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--answers must contain at least one value")
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise ValueError("--answers must be comma-separated integers") from e
