"""Agent-trajectory helpers for answer_type=agent (PROMST / GEPA proxy).

Training never shells out to promptfoo and never calls a hosted model for
candidate fitness. A deterministic, instruction-conditioned policy reads
routing hints from the instruction text, then emits a JSON trajectory
``{"tools": [...], "answer": "..."}``. Instruction mutations that add the
right tool-order rules change the proxy score; the final promptfoo eval still
uses exact JSON deep-equality against golden.jsonl.
"""

from __future__ import annotations

import json
import re
from typing import Any

INTENT_ANSWERS = {
    "outage": "インシデントを起票しました。",
    "contract": "契約FAQを案内しました。",
    "feature": "要望を記録しました。",
    "other": "担当へ引き継ぎます。",
}

INTENT_ALIASES = {
    "outage": ("障害報告", "障害", "インシデント", "outage", "incident"),
    "contract": ("契約照会", "契約", "contract"),
    "feature": ("機能要望", "要望", "feature"),
    "other": ("その他", "other", "greeting"),
}

_OUTAGE_INPUT = re.compile(
    r"ログインできない|パスワードを何度|クラッシュ|真っ白|フリーズ|エラーになります|"
    r"起動しなく|通知メール|障害"
)
_CONTRACT_INPUT = re.compile(r"契約|解約|条項|更新料|支払いサイト|見積")
_FEATURE_INPUT = re.compile(r"追加して|ほしい|ダークモード|フィルタ|プッシュ通知|Excel|エクスポート")

_TOOL_TOKEN_RE = re.compile(r"kb\.search|ticket\.create")
_NO_TOOLS_RE = re.compile(r"ツールなし|no tools|tools\s*[:=]\s*\[\s*\]|\bnone\b", re.I)
_ANSWER_RE = re.compile(r"(?:回答|answer)\s*[:：]\s*[「\"']?([^」\"'\n]+)[」\"']?", re.I)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def classify_intent(text: str) -> str:
    if _OUTAGE_INPUT.search(text or ""):
        return "outage"
    if _CONTRACT_INPUT.search(text or ""):
        return "contract"
    if _FEATURE_INPUT.search(text or ""):
        return "feature"
    return "other"


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
    gold = parse_agent_trajectory(expected if not isinstance(expected, dict) else expected)
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


def _lines(instruction: str) -> list[str]:
    return [ln.strip() for ln in (instruction or "").splitlines() if ln.strip()]


def _tools_from_text(text: str) -> list[str] | None:
    if _NO_TOOLS_RE.search(text):
        return []
    found = _TOOL_TOKEN_RE.findall(text)
    return list(found) if found else None


def _window_around(blob: str, alias: str) -> str | None:
    low_blob = blob.lower()
    low_alias = alias.lower()
    idx = low_blob.find(low_alias)
    if idx < 0:
        idx = blob.find(alias)
    if idx < 0:
        return None
    return blob[max(0, idx - 48) : idx + 160]


def _snippets_for_intent(instruction: str, intent: str) -> list[str]:
    aliases = INTENT_ALIASES[intent]
    snippets: list[str] = []
    for ln in _lines(instruction):
        if any(alias in ln or alias.lower() in ln.lower() for alias in aliases):
            snippets.append(ln)
    if snippets:
        return snippets
    blob = instruction or ""
    for alias in aliases:
        window = _window_around(blob, alias)
        if window:
            snippets.append(window)
            break
    return snippets


def routing_for_intent(instruction: str, intent: str) -> tuple[list[str] | None, str | None]:
    """Return (tools, answer_template) parsed from instruction text for intent."""
    tools: list[str] | None = None
    answer: str | None = None
    for text in _snippets_for_intent(instruction, intent):
        if tools is None:
            parsed_tools = _tools_from_text(text)
            if parsed_tools is not None:
                tools = parsed_tools
        if answer is None:
            parsed_answer = _ANSWER_RE.search(text)
            if parsed_answer:
                answer = parsed_answer.group(1).strip()
        if tools is not None and answer is not None:
            break
    return tools, answer


def rollout_from_instruction(instruction: str, user_input: str) -> dict[str, Any]:
    """Deterministic policy: classify the user input, then apply instruction routing.

    Missing routing falls back to always calling ``kb.search`` (the typical
    under-specified agent prompt). Answers default to the intent template so
    tool-order is the main training signal.
    """
    intent = classify_intent(user_input)
    tools, answer = routing_for_intent(instruction, intent)
    if tools is None:
        tools = ["kb.search"]
    if not answer:
        answer = INTENT_ANSWERS[intent]
    return {"tools": list(tools), "answer": answer}


def trajectory_json(traj: dict[str, Any]) -> str:
    return json.dumps({"tools": traj.get("tools") or [], "answer": traj.get("answer") or ""}, ensure_ascii=False)
