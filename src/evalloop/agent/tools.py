"""Local tools for the in-process support agent (no hosted APIs).

APO candidate evaluation must stay in-process (iron rule: Python does not
call model providers except via promptfoo; optimize uses dspy only for the
reflection LM). Tool execution itself is deterministic and local.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KnowledgeDoc:
    doc_id: str
    keywords: tuple[str, ...]
    text: str


SUPPORT_DOCS: tuple[KnowledgeDoc, ...] = (
    KnowledgeDoc(
        "outage-login",
        ("ログイン", "パスワード", "エラー", "クラッシュ", "真っ白", "フリーズ"),
        "認証・画面障害は既知のインシデント対象です。kb.search のあと ticket.create で起票してください。",
    ),
    KnowledgeDoc(
        "contract-faq",
        ("契約", "解約", "条項", "更新料", "見積", "支払い"),
        "契約・解約・更新の案内は契約FAQにあります。検索結果を案内してください。",
    ),
    KnowledgeDoc(
        "hours",
        ("営業時間", "資料請求", "ISO", "担当営業"),
        "一般的な案内事項です。ツールは不要で担当へ引き継いでください。",
    ),
)


@dataclass
class AgentEnv:
    """Mutable tool sandbox for one rollout (fresh tickets per run)."""

    docs: tuple[KnowledgeDoc, ...] = SUPPORT_DOCS
    tickets: list[str] = field(default_factory=list)

    def execute(self, tool: str, args: dict) -> str:
        if tool == "kb.search":
            return self._search(str(args.get("query") or ""))
        if tool == "ticket.create":
            return self._create_ticket(str(args.get("summary") or ""))
        return f"unknown tool: {tool}"

    def _search(self, query: str) -> str:
        hits = [doc for doc in self.docs if any(k in query for k in doc.keywords)]
        if not hits:
            return "該当するFAQはありません。"
        return " / ".join(doc.text for doc in hits)

    def _create_ticket(self, summary: str) -> str:
        n = len(self.tickets) + 1
        ticket_id = f"TICKET-{n:04d}"
        self.tickets.append(ticket_id)
        clipped = (summary or "").strip().replace("\n", " ")[:80]
        return f"created {ticket_id}: {clipped or '(no summary)'}"
