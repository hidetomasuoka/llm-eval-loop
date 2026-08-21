"""Policies that pick the next agent action from instruction + history."""

from __future__ import annotations

import json
import re

from evalloop.agent.loop import Action, AgentStep

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


def classify_intent(text: str) -> str:
    if _OUTAGE_INPUT.search(text or ""):
        return "outage"
    if _CONTRACT_INPUT.search(text or ""):
        return "contract"
    if _FEATURE_INPUT.search(text or ""):
        return "feature"
    return "other"


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


def _args_for_tool(tool: str, user_input: str) -> dict[str, str]:
    if tool == "ticket.create":
        return {"summary": (user_input or "").strip().replace("\n", " ")[:120]}
    return {"query": user_input or ""}


class InstructionRoutingPolicy:
    """Deterministic policy: instruction text supplies the tool plan; tools actually run.

    Missing routing falls back to a single ``kb.search`` (under-specified agent prompt).
    This is the APO training proxy: instruction mutations change the plan, and the
    loop executes that plan step by step.
    """

    def next_action(
        self,
        *,
        instruction: str,
        user_input: str,
        history: list[AgentStep],
    ) -> Action:
        intent = classify_intent(user_input)
        tools, answer = routing_for_intent(instruction, intent)
        if tools is None:
            tools = ["kb.search"]
        if not answer:
            answer = INTENT_ANSWERS[intent]
        done = len(history)
        if done < len(tools):
            tool = tools[done]
            return Action(kind="tool", tool=tool, args=_args_for_tool(tool, user_input))
        return Action(kind="finish", answer=answer)


def _lm_text(prompt_model, prompt: str) -> str:
    raw = prompt_model(prompt)
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list) and raw:
        first = raw[0]
        return str(first).strip() if not isinstance(first, dict) else str(first.get("text", first)).strip()
    if isinstance(raw, dict):
        for key in ("text", "content", "output"):
            if key in raw and raw[key]:
                return str(raw[key]).strip()
    return str(raw).strip()


class LmStepPolicy:
    """One LM call per step (dspy/reflection-style). Used for live agent rollouts.

    Candidate fitness during PROMST stays on InstructionRoutingPolicy so tests and
    the iron-rule proxy never call a hosted model per tool step.
    """

    def __init__(self, prompt_model):
        self.prompt_model = prompt_model

    def next_action(
        self,
        *,
        instruction: str,
        user_input: str,
        history: list[AgentStep],
    ) -> Action:
        hist = [
            {"tool": s.tool, "args": s.args, "observation": s.observation}
            for s in history
        ]
        prompt = (
            f"{instruction}\n\nUser input:\n{user_input}\n\n"
            f"History so far:\n{json.dumps(hist, ensure_ascii=False)}\n\n"
            "Return ONLY JSON for the next action, no fences:\n"
            '{"kind":"tool","tool":"kb.search","args":{"query":"..."} }\n'
            'or {"kind":"finish","answer":"..."}\n'
            "Allowed tools: kb.search, ticket.create."
        )
        raw = _lm_text(self.prompt_model, prompt)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return Action(kind="finish", answer="")
        if not isinstance(parsed, dict):
            return Action(kind="finish", answer="")
        kind = str(parsed.get("kind") or "").strip().lower()
        if kind == "finish":
            return Action(kind="finish", answer=str(parsed.get("answer") or ""))
        tool = parsed.get("tool")
        args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
        if not isinstance(tool, str) or not tool.strip():
            return Action(kind="finish", answer="")
        return Action(kind="tool", tool=tool.strip(), args=args)
