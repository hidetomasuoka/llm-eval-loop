"""In-process agent loop: policy chooses an action, tools execute, history grows.

This is the runtime APO trains against. PROMST mutates the instruction; the
loop re-rolls the same tools with the new plan. Candidate fitness does not
call a hosted LLM (InstructionRoutingPolicy is the default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from evalloop.agent.tools import AgentEnv


@dataclass
class Action:
    kind: str  # "tool" | "finish"
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    answer: str = ""


@dataclass
class AgentStep:
    tool: str
    args: dict[str, Any]
    observation: str

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "observation": self.observation}


@dataclass
class Trajectory:
    steps: list[AgentStep]
    answer: str
    finished: bool
    truncated: bool = False

    @property
    def tools(self) -> list[str]:
        return [s.tool for s in self.steps]

    def as_eval(self) -> dict[str, Any]:
        return {"tools": self.tools, "answer": self.answer}

    def as_steps(self) -> list[dict[str, Any]]:
        return [s.as_dict() for s in self.steps]


class Policy(Protocol):
    def next_action(
        self,
        *,
        instruction: str,
        user_input: str,
        history: list[AgentStep],
    ) -> Action: ...


DEFAULT_MAX_STEPS = 4


def run_agent(
    instruction: str,
    user_input: str,
    *,
    policy: Policy | None = None,
    env: AgentEnv | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Trajectory:
    """Execute tools in a loop until the policy finishes or max_steps is hit."""
    if policy is None:
        from evalloop.agent.policy import InstructionRoutingPolicy

        policy = InstructionRoutingPolicy()
    env = env or AgentEnv()
    steps: list[AgentStep] = []
    for _ in range(max_steps):
        action = policy.next_action(
            instruction=instruction,
            user_input=user_input,
            history=steps,
        )
        if action.kind == "finish":
            return Trajectory(steps=steps, answer=action.answer or "", finished=True)
        tool = (action.tool or "").strip()
        if action.kind != "tool" or not tool:
            return Trajectory(steps=steps, answer="", finished=True)
        observation = env.execute(tool, action.args or {})
        steps.append(AgentStep(tool=tool, args=dict(action.args or {}), observation=observation))
    return Trajectory(steps=steps, answer="", finished=False, truncated=True)
