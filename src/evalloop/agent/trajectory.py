"""Parse and score agent trajectories ({tools, answer} plus optional steps)."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def parse_agent_trajectory(output: Any) -> dict[str, Any]:
    """Parse a model/policy output into ``{"tools": [str], "answer": str}``.

    Invalid JSON or a non-object yields empty tools and empty answer so the
    trajectory metric can still name the first failing step.
    """
    parsed: Any = output
    if isinstance(output, str):
        text = output.strip()
        text = _FENCE_RE.sub("", text).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"tools": [], "answer": ""}
    if not isinstance(parsed, dict):
        return {"tools": [], "answer": ""}
    raw_tools = parsed.get("tools", [])
    tools: list[str] = []
    if isinstance(raw_tools, list):
        for item in raw_tools:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if not name or name.lower() == "none":
                continue
            tools.append(name)
    answer = parsed.get("answer", "")
    return {"tools": tools, "answer": answer if isinstance(answer, str) else ""}


def _normalize_answer(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().replace("。", "").replace(".", "")


def first_failing_step(pred: dict[str, Any], expected: Any) -> dict[str, Any]:
    gold = parse_agent_trajectory(expected)
    pred_t = pred.get("tools") or []
    gold_t = gold.get("tools") or []
    n = max(len(pred_t), len(gold_t))
    for i in range(n):
        predicted = pred_t[i] if i < len(pred_t) else None
        wanted = gold_t[i] if i < len(gold_t) else None
        if predicted != wanted:
            return {
                "kind": "tool",
                "index": i,
                "expected": wanted,
                "predicted": predicted,
            }
    if _normalize_answer(pred.get("answer")) != _normalize_answer(gold.get("answer")):
        return {
            "kind": "answer",
            "expected": gold.get("answer"),
            "predicted": pred.get("answer"),
        }
    return {"kind": "none"}


def tools_prefix_score(pred_tools: list[str], gold_tools: list[str]) -> float:
    if not pred_tools and not gold_tools:
        return 1.0
    n = max(len(pred_tools), len(gold_tools), 1)
    matched = 0
    for i in range(min(len(pred_tools), len(gold_tools))):
        if pred_tools[i] == gold_tools[i]:
            matched += 1
        else:
            break
    return matched / n


def trajectory_json(traj: dict[str, Any]) -> str:
    return json.dumps({"tools": traj.get("tools") or [], "answer": traj.get("answer") or ""}, ensure_ascii=False)
