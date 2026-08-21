"""In-process agent runtime that APO (PROMST) trains against.

The loop executes local tools step by step. The default policy reads a tool
plan from the instruction text so instruction mutations change the trace
without calling a hosted model. Final holdout eval is still promptfoo JSON
deep-equality against ``{tools, answer}``.
"""

from evalloop.agent.loop import Action, AgentStep, Policy, Trajectory, run_agent
from evalloop.agent.policy import (
    INTENT_ANSWERS,
    InstructionRoutingPolicy,
    LmStepPolicy,
    classify_intent,
    routing_for_intent,
)
from evalloop.agent.tools import AgentEnv
from evalloop.agent.trajectory import (
    first_failing_step,
    parse_agent_trajectory,
    tools_prefix_score,
    trajectory_json,
)


def rollout_from_instruction(instruction: str, user_input: str) -> dict:
    """Run the agent loop and return the eval-shaped ``{tools, answer}`` dict."""
    return run_agent(instruction, user_input).as_eval()


__all__ = [
    "Action",
    "AgentEnv",
    "AgentStep",
    "INTENT_ANSWERS",
    "InstructionRoutingPolicy",
    "LmStepPolicy",
    "Policy",
    "Trajectory",
    "classify_intent",
    "first_failing_step",
    "parse_agent_trajectory",
    "rollout_from_instruction",
    "routing_for_intent",
    "run_agent",
    "tools_prefix_score",
    "trajectory_json",
]
