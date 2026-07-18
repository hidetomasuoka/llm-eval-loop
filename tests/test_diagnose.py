import pytest
from typer.testing import CliRunner

from evalloop import cli
from evalloop import diagnose as diagnose_mod


def _run(answers: list[int]) -> tuple[diagnose_mod.DiagnoseOutcome, str]:
    test_console = diagnose_mod.Console(record=True, width=120)
    original = diagnose_mod.console
    diagnose_mod.console = test_console
    try:
        outcome = diagnose_mod.run_diagnose(answers=answers)
        return outcome, test_console.export_text()
    finally:
        diagnose_mod.console = original


def test_parse_answers_splits_integers():
    assert diagnose_mod.parse_answers("2,1,1") == [2, 1, 1]


def test_parse_answers_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        diagnose_mod.parse_answers(",")


def test_q1_yes_defers_non_prompt():
    outcome, text = _run([1])
    assert outcome == diagnose_mod.DiagnoseOutcome.DEFER_NON_PROMPT
    assert "APO適用は保留" in text
    assert "RAG / パース" in text


@pytest.mark.parametrize(
    ("answers", "expected_method"),
    [
        ([2, 1, 1], "gepa"),
        ([2, 2, 1], "miprov2"),
    ],
)
def test_supported_symptoms_recommend_method(answers, expected_method):
    outcome, text = _run(answers)
    assert outcome == diagnose_mod.DiagnoseOutcome.RECOMMEND_METHOD
    assert diagnose_mod.METHOD_SNIPPETS[expected_method] in text
    assert "次のステップ" in text


@pytest.mark.parametrize(
    "answers",
    [
        [2, 1, 2],
        [2, 2, 2],
    ],
)
def test_supported_symptoms_need_eval_set_first(answers):
    outcome, text = _run(answers)
    assert outcome == diagnose_mod.DiagnoseOutcome.NEED_EVAL_SET
    assert "評価セット整備が先" in text
    assert "APOは評価セットの上に成り立ちます" in text


@pytest.mark.parametrize(
    ("answers", "fragment"),
    [
        ([2, 3], "7c. 長文構造"),
        ([2, 4], "7d. 多目的"),
        ([2, 5], "7e. Agent/Multi-step"),
    ],
)
def test_out_of_scope_symptoms(answers, fragment):
    outcome, text = _run(answers)
    assert outcome == diagnose_mod.DiagnoseOutcome.OUT_OF_SCOPE
    assert fragment in text
    assert "未対応" in text
    assert "method: gepa" not in text


def test_instruction_symptom_shows_copro_alternative():
    _, text = _run([2, 1, 1])
    assert "7a. Instruction" in text
    assert "gepa / copro" in text
    assert "代替候補: copro" in text


def test_exemplar_symptom_shows_miprov2_granularity():
    _, text = _run([2, 2, 1])
    assert "7b. Exemplar" in text
    assert "miprov2" in text


def test_invalid_q1_answer_raises():
    with pytest.raises(ValueError, match="Q1"):
        diagnose_mod.run_diagnose(answers=[9])


def test_invalid_q2_answer_raises():
    with pytest.raises(ValueError):
        diagnose_mod.run_diagnose(answers=[2, 9])


def test_invalid_q3_answer_raises():
    with pytest.raises(ValueError, match="Q3"):
        diagnose_mod.run_diagnose(answers=[2, 1, 9])


def test_cli_diagnose_answers_mode():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["diagnose", "--answers", "2,1,1"])
    assert result.exit_code == 0
    assert "method: gepa" in result.stdout


def test_cli_diagnose_invalid_answers_exits():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["diagnose", "--answers", "bad"])
    assert result.exit_code == 1
    assert "diagnose failed" in result.stdout
