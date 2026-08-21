"""Backward-compatible agent helpers used by PROMST / GEPA proxy metrics.

The runtime lives in ``evalloop.agent`` (loop + local tools + policies).
This module re-exports the names the optimizer package historically imported.
"""

from evalloop.agent import (
    INTENT_ANSWERS,
    classify_intent,
    first_failing_step,
    parse_agent_trajectory,
    rollout_from_instruction,
    routing_for_intent,
    run_agent,
    tools_prefix_score,
    trajectory_json,
)

__all__ = [
    "INTENT_ANSWERS",
    "classify_intent",
    "first_failing_step",
    "parse_agent_trajectory",
    "rollout_from_instruction",
    "routing_for_intent",
    "run_agent",
    "tools_prefix_score",
    "trajectory_json",
]
