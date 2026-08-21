"""Agent trajectory proxy metric + instruction-conditioned rollout."""

from __future__ import annotations

from evalloop.optimizers.agent import (
    classify_intent,
    first_failing_step,
    parse_agent_trajectory,
    rollout_from_instruction,
)
from evalloop.optimizers.metrics import agent_score_and_feedback
from evalloop.schemas import load_task

WEAK_INSTRUCTION = (
    "あなたはサポートエージェントです。よく分からないときは kb.search を使ってください。"
    " JSON のみを返してください。"
)

STRONG_INSTRUCTION = """\
障害報告: kb.search → ticket.create。回答: インシデントを起票しました。
契約照会: kb.search。回答: 契約FAQを案内しました。
機能要望: ツールなし。回答: 要望を記録しました。
その他: ツールなし。回答: 担当へ引き継ぎます。
"""


def test_classify_intent_matches_sample_agent_texts():
    assert classify_intent("ログインできません。パスワードを何度入力してもエラーになります。") == "outage"
    assert classify_intent("契約書の解約条項について教えてください。") == "contract"
    assert classify_intent("フィルタを追加してほしいです。") == "feature"
    assert classify_intent("営業時間を教えてください。") == "other"


def test_parse_agent_trajectory_accepts_fenced_json():
    raw = '```json\n{"tools": ["kb.search"], "answer": "ok"}\n```'
    assert parse_agent_trajectory(raw) == {"tools": ["kb.search"], "answer": "ok"}
    assert parse_agent_trajectory("not json") == {"tools": [], "answer": ""}


def test_first_failing_step_names_missing_ticket():
    pred = {"tools": ["kb.search"], "answer": "インシデントを起票しました。"}
    gold = {"tools": ["kb.search", "ticket.create"], "answer": "インシデントを起票しました。"}
    fail = first_failing_step(pred, gold)
    assert fail["kind"] == "tool"
    assert fail["index"] == 1
    assert fail["expected"] == "ticket.create"
    assert fail["predicted"] is None


def test_weak_instruction_scores_below_strong_on_sample_agent():
    cfg, paths = load_task("sample-agent")
    from evalloop.schemas import load_golden_jsonl

    train = [c for c in load_golden_jsonl(paths.golden) if c.split == "train"]
    weak_total = 0.0
    strong_total = 0.0
    for case in train:
        weak = rollout_from_instruction(WEAK_INSTRUCTION, case.input)
        strong = rollout_from_instruction(STRONG_INSTRUCTION, case.input)
        w, _ = agent_score_and_feedback(weak, case.expected)
        s, _ = agent_score_and_feedback(strong, case.expected)
        weak_total += w
        strong_total += s
        assert s == 1.0
    assert strong_total / len(train) == 1.0
    assert weak_total / len(train) < 0.8
    assert cfg.task.answer_type == "agent"
