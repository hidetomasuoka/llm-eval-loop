"""Seed a studio workspace with a working Dify/LangChain/DataRobot demo.

Uses the bundled sample-inquiry task when present, plus a synthetic tabular
churn set. Trains AutoML models and registers processes/apps so
`evalloop studio seed && evalloop studio app run inquiry-bot --input ...`
works without API keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalloop.paths import REPO_ROOT
from evalloop.studio.apps import save_app
from evalloop.studio.errors import StudioError
from evalloop.studio.automl import load_trained_model, register_trained_model, train_job
from evalloop.studio.data import import_dataset, import_from_task
from evalloop.studio.knowledge import import_knowledge
from evalloop.studio.models import import_from_registry
from evalloop.studio.processes import save_process
from evalloop.studio.store import StudioStore

CHURN_CSV = """id,tenure_months,monthly_spend,open_tickets,plan,churn
c01,1,18,6,basic,yes
c02,2,22,4,basic,yes
c03,3,25,5,basic,yes
c04,4,19,3,basic,yes
c05,6,40,2,plus,yes
c06,7,38,3,plus,yes
c07,18,90,0,plus,no
c08,20,95,1,plus,no
c09,24,140,0,pro,no
c10,30,155,0,pro,no
c11,36,160,1,pro,no
c12,12,70,1,plus,no
c13,2,21,7,basic,yes
c14,28,150,0,pro,no
c15,8,42,4,plus,yes
c16,22,110,0,pro,no
c17,3,17,8,basic,yes
c18,16,88,1,plus,no
c19,5,33,5,basic,yes
c20,26,148,0,pro,no
c21,9,55,2,plus,no
c22,1,16,9,basic,yes
c23,32,170,0,pro,no
c24,14,80,1,plus,no
"""

INQUIRY_FAQ = [
    {
        "id": "faq-contract",
        "title": "契約・解約",
        "text": "契約書の解約条項、更新手続き、違約金、支払いサイトの変更は契約照会です。契約チームが一次対応します。",
    },
    {
        "id": "faq-incident",
        "title": "障害・ログイン",
        "text": "ログインできない、画面が真っ白、クラッシュ、タイムアウトは障害報告です。至急対応キューに入れます。",
    },
    {
        "id": "faq-feature",
        "title": "機能要望",
        "text": "フィルタ追加、ダークモード、CSV/Excelエクスポート、プッシュ通知などの改善希望は機能要望です。",
    },
    {
        "id": "faq-other",
        "title": "その他の問い合わせ",
        "text": "営業時間、資料請求、担当者の連絡先、ISO取得有無など分類しにくい内容はその他です。",
    },
]


def _alias_winner(store: StudioStore, alias: str, source_id: str, description: str) -> dict[str, Any]:
    artifact = load_trained_model(store, source_id)
    meta = store.get("models", source_id)
    return register_trained_model(
        store,
        alias,
        artifact,
        description=description,
        job_id=str(meta.get("job_id") or "seed"),
        metrics=meta.get("metrics") or {},
        deployed=True,
    )


def seed(store: StudioStore, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or REPO_ROOT
    store.init()
    summary: dict[str, Any] = {"root": str(store.root), "imported": []}

    try:
        import_from_registry(store, repo_root=root)
        summary["imported"].append("llm-registry")
    except (StudioError, ValueError, OSError) as exc:
        summary["registry_error"] = str(exc)

    inquiry_ok = False
    try:
        import_from_task(store, "sample-inquiry", dataset_id="inquiry", repo_root=root)
        inquiry_ok = True
        summary["imported"].append("dataset:inquiry")
    except (StudioError, ValueError, RuntimeError, OSError) as exc:
        summary["inquiry_error"] = str(exc)

    from evalloop.studio.data import load_tabular_text

    columns, rows = load_tabular_text(CHURN_CSV, "csv")
    import_dataset(
        store,
        "churn-toy",
        rows=rows,
        columns=columns,
        origin="seed:churn-toy",
        description="Synthetic SaaS churn table for AutoML demo (self-made)",
    )
    summary["imported"].append("dataset:churn-toy")

    import_knowledge(store, "inquiry-faq", INQUIRY_FAQ, description="Inquiry routing FAQ (self-made)")
    summary["imported"].append("knowledge:inquiry-faq")

    if inquiry_ok:
        job = train_job(store, "inquiry", "expected", job_id="job-inquiry-seed", seed=0, feature_columns=["input"])
        _alias_winner(store, "inquiry-clf", job["winner_model_id"], "Stable alias for the inquiry AutoML winner")
        summary["inquiry_job"] = job["id"]
        summary["inquiry_winner"] = job["winner"]
        save_process(store, _inquiry_process())
        save_app(
            store,
            {
                "id": "inquiry-bot",
                "name": "問い合わせ仕分けボット",
                "description": "Dify-like app: classify + retrieve FAQ + template reply",
                "process": "inquiry-triage",
                "knowledge": "inquiry-faq",
                "models": {"classifier": "inquiry-clf"},
                "inputs": [{"name": "input", "type": "string"}],
            },
        )
        summary["imported"].extend(["process:inquiry-triage", "app:inquiry-bot"])

    churn_job = train_job(store, "churn-toy", "churn", job_id="job-churn-seed", seed=0)
    _alias_winner(store, "churn-clf", churn_job["winner_model_id"], "Stable alias for the churn AutoML winner")
    summary["churn_job"] = churn_job["id"]
    summary["churn_winner"] = churn_job["winner"]
    save_process(store, _churn_process())
    save_app(
        store,
        {
            "id": "churn-watch",
            "name": "解約リスク監視",
            "description": "DataRobot-like scoring app over the churn classifier",
            "process": "churn-risk",
            "models": {"classifier": "churn-clf"},
            "inputs": [
                {"name": "tenure_months", "type": "number"},
                {"name": "monthly_spend", "type": "number"},
                {"name": "open_tickets", "type": "number"},
                {"name": "plan", "type": "string"},
            ],
        },
    )
    summary["imported"].extend(["process:churn-risk", "app:churn-watch"])
    return summary


def _inquiry_process() -> dict[str, Any]:
    return {
        "id": "inquiry-triage",
        "name": "問い合わせトリアージ",
        "description": "LangChain-like chain: classify → retrieve → branch reply → format",
        "inputs": [{"name": "input", "type": "string"}],
        "outputs": ["label", "reply", "answer"],
        "steps": [
            {
                "id": "label",
                "kind": "classify",
                "model": "inquiry-clf",
                "feature": "input",
                "input": "input",
                "output": "label",
            },
            {
                "id": "passages",
                "kind": "retrieve",
                "knowledge": "inquiry-faq",
                "query": "{{input}}",
                "k": 2,
                "output": "passages",
            },
            {
                "id": "context",
                "kind": "transform",
                "field": "passages",
                "op": "join",
                "sep": " / ",
                "output": "context",
            },
            {
                "id": "reply",
                "kind": "branch",
                "on": "{{label}}",
                "cases": {
                    "障害報告": "至急対応が必要です。障害キューに投入しました。",
                    "契約照会": "契約チームへ転送します。解約・更新・支払い条件を確認します。",
                    "機能要望": "プロダクトバックログに登録します。",
                    "その他": "受付しました。担当からご連絡します。",
                },
                "default": "受付しました。",
                "output": "reply",
            },
            {
                "id": "answer",
                "kind": "llm",
                "provider": "template",
                "template": "[{{label}}] {{reply}}\n入力: {{input}}\n参考: {{context}}",
                "output": "answer",
            },
        ],
    }


def _churn_process() -> dict[str, Any]:
    return {
        "id": "churn-risk",
        "name": "解約リスク判定",
        "description": "Score a customer row with the AutoML churn model",
        "inputs": [
            {"name": "tenure_months", "type": "number"},
            {"name": "monthly_spend", "type": "number"},
            {"name": "open_tickets", "type": "number"},
            {"name": "plan", "type": "string"},
        ],
        "outputs": ["prediction", "risk"],
        "steps": [
            {
                "id": "scored",
                "kind": "predict",
                "model": "churn-clf",
                "features": {
                    "tenure_months": "{{tenure_months}}",
                    "monthly_spend": "{{monthly_spend}}",
                    "open_tickets": "{{open_tickets}}",
                    "plan": "{{plan}}",
                },
                "output": "scored",
            },
            {
                "id": "prediction",
                "kind": "expr",
                "expr": "scored['prediction']",
                "output": "prediction",
            },
            {
                "id": "risk",
                "kind": "branch",
                "on": "{{prediction}}",
                "cases": {"yes": "high", "no": "low"},
                "default": "unknown",
                "output": "risk",
            },
        ],
    }
