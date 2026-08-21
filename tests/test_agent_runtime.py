"""In-process agent loop: tools actually execute; APO trains against this runtime."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from evalloop import cli
from evalloop.agent import AgentEnv, LmStepPolicy, run_agent
from evalloop.optimizers.agent import rollout_from_instruction

STRONG_INSTRUCTION = """\
障害報告: kb.search → ticket.create。回答: インシデントを起票しました。
契約照会: kb.search。回答: 契約FAQを案内しました。
機能要望: ツールなし。回答: 要望を記録しました。
その他: ツールなし。回答: 担当へ引き継ぎます。
"""

OUTAGE = "システムにログインできません。パスワードを何度入力してもエラーになります。"


def test_outage_loop_executes_search_then_ticket():
    env = AgentEnv()
    traj = run_agent(STRONG_INSTRUCTION, OUTAGE, env=env)
    assert traj.tools == ["kb.search", "ticket.create"]
    assert traj.answer == "インシデントを起票しました。"
    assert traj.finished and not traj.truncated
    assert "インシデント" in traj.steps[0].observation
    assert traj.steps[1].observation.startswith("created TICKET-")
    assert env.tickets == ["TICKET-0001"]


def test_weak_instruction_stops_after_search():
    weak = "よく分からないときは kb.search を使ってください。"
    traj = run_agent(weak, OUTAGE)
    assert traj.tools == ["kb.search"]
    assert traj.steps[0].observation
    assert not any(s.tool == "ticket.create" for s in traj.steps)


def test_feature_request_runs_no_tools():
    traj = run_agent(STRONG_INSTRUCTION, "検索結果に絞り込みフィルタを追加してほしいです。")
    assert traj.tools == []
    assert traj.answer == "要望を記録しました。"
    assert traj.as_eval() == rollout_from_instruction(
        STRONG_INSTRUCTION, "検索結果に絞り込みフィルタを追加してほしいです。"
    )


def test_max_steps_truncates_when_policy_never_finishes():
    class NeverFinish:
        def next_action(self, *, instruction, user_input, history):
            from evalloop.agent import Action

            return Action(kind="tool", tool="kb.search", args={"query": user_input})

    traj = run_agent("x", OUTAGE, policy=NeverFinish(), max_steps=2)
    assert traj.truncated
    assert not traj.finished
    assert len(traj.steps) == 2


def test_lm_step_policy_stub_runs_tools_then_finishes():
    calls = {"n": 0}

    def fake_lm(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"kind": "tool", "tool": "kb.search", "args": {"query": "ログイン"}})
        return json.dumps({"kind": "finish", "answer": "案内しました。"})

    traj = run_agent("live", OUTAGE, policy=LmStepPolicy(fake_lm))
    assert traj.tools == ["kb.search"]
    assert traj.answer == "案内しました。"
    assert calls["n"] == 2
    assert traj.steps[0].observation


def test_cli_agent_rollout_sample_agent():
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["agent", "rollout", "--task", "sample-agent", OUTAGE],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["task"] == "sample-agent"
    assert payload["tools"] == ["kb.search"]
    assert payload["steps"][0]["tool"] == "kb.search"
    assert payload["steps"][0]["observation"]


def test_cli_agent_rollout_rejects_non_agent_task():
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["agent", "rollout", "--task", "sample-inquiry", "hello"],
    )
    assert result.exit_code == 1
    assert "answer_type=agent" in result.stdout
