"""Studio: process / data / model platform (Dify + LangChain + DataRobot analog)."""

import pytest
from typer.testing import CliRunner

from evalloop.cli import app
from evalloop.paths import REPO_ROOT
from evalloop.studio.apps import run_app
from evalloop.studio.automl import predict_row, train_job
from evalloop.studio.data import import_dataset, load_profile, load_tabular_text
from evalloop.studio.errors import StudioError
from evalloop.studio.expr import safe_eval
from evalloop.studio.ids import validate_id
from evalloop.studio.knowledge import import_knowledge, search
from evalloop.studio.processes import run_process, save_process
from evalloop.studio.render import render
from evalloop.studio.seed import CHURN_CSV, seed
from evalloop.studio.serve import handle_request
from evalloop.studio.store import StudioStore

runner = CliRunner()


def test_validate_id_rejects_uppercase():
    with pytest.raises(StudioError):
        validate_id("Inquiry")


def test_render_nested_and_missing():
    assert render("hello {{user.name}}", {"user": {"name": "Ada"}}) == "hello Ada"
    assert render("x={{missing}}", {}) == "x="


def test_store_put_get_list(isolated_root):
    store = StudioStore(isolated_root / "studio")
    store.init()
    store.put("datasets", "iris", {"kind": "dataset", "n_rows": 3})
    got = store.get("datasets", "iris")
    assert got["n_rows"] == 3
    assert store.list("datasets")[0]["id"] == "iris"
    with pytest.raises(StudioError):
        store.get("datasets", "missing")


def test_dataset_profile_and_types(isolated_root):
    store = StudioStore(isolated_root / "studio")
    path = isolated_root / "tiny.csv"
    path.write_text("id,text,label,score\na,hello world,yes,1.5\nb,hi,no,2.0\n", encoding="utf-8")
    import_dataset(store, "tiny", source=path, origin="test")
    profile = load_profile(store, "tiny")
    types = {c["name"]: c["type"] for c in profile["columns"]}
    assert types["score"] == "numeric"
    assert types["label"] == "categorical"
    assert profile["n_rows"] == 2


def test_knowledge_search_ranks_relevant_doc(isolated_root):
    store = StudioStore(isolated_root / "studio")
    import_knowledge(
        store,
        "faq",
        [
            {"id": "d1", "title": "契約", "text": "解約条項と更新手続きは契約チームが担当します。"},
            {"id": "d2", "title": "障害", "text": "ログインできない、画面が真っ白なときは障害報告です。"},
        ],
    )
    hits = search(store, "faq", "ログインできません", k=1)
    assert hits[0]["id"] == "d2"
    assert hits[0]["score"] > 0


def test_safe_eval_allowlist_and_rejects_import():
    assert safe_eval("len(items) + 1", {"items": [1, 2]}) == 3
    assert safe_eval("scored['prediction']", {"scored": {"prediction": "yes"}}) == "yes"
    with pytest.raises(StudioError):
        safe_eval("__import__('os').system('pwd')", {})
    with pytest.raises(StudioError):
        safe_eval("(1).__class__", {})


def test_automl_churn_leaderboard_and_predict(isolated_root):
    store = StudioStore(isolated_root / "studio")
    columns, rows = load_tabular_text(CHURN_CSV, "csv")
    import_dataset(store, "churn-toy", rows=rows, columns=columns, origin="test")
    job = train_job(store, "churn-toy", "churn", job_id="job-churn-test", seed=0)
    assert job["problem"] == "classification"
    assert job["winner"]
    assert job["n_train"] >= 8 and job["n_test"] >= 4
    winner_meta = store.get("models", job["winner_model_id"])
    assert winner_meta["metrics"]["accuracy"] >= 0.6
    pred = predict_row(
        store,
        job["winner_model_id"],
        {"tenure_months": 1, "monthly_spend": 16, "open_tickets": 9, "plan": "basic"},
    )
    assert pred["prediction"] in {"yes", "no"}
    assert "probabilities" in pred


def test_process_rejects_hosted_llm_provider(isolated_root):
    store = StudioStore(isolated_root / "studio")
    save_process(
        store,
        {
            "id": "bad-llm",
            "steps": [{"id": "g", "kind": "llm", "provider": "anthropic", "prompt": "hi", "output": "out"}],
            "outputs": ["out"],
        },
    )
    with pytest.raises(StudioError, match="not a local studio provider"):
        run_process(store, "bad-llm", {}, record=False)


def test_process_unknown_kind(isolated_root):
    store = StudioStore(isolated_root / "studio")
    with pytest.raises(StudioError, match="unknown step kind"):
        save_process(
            store,
            {"id": "bad", "steps": [{"id": "x", "kind": "agent-executor", "output": "y"}]},
        )


def test_seed_inquiry_bot_classifies_incident(isolated_root):
    store = StudioStore(isolated_root / "studio")
    summary = seed(store, repo_root=REPO_ROOT)
    assert "app:inquiry-bot" in summary["imported"]
    assert "app:churn-watch" in summary["imported"]
    result = run_app(
        store,
        "inquiry-bot",
        {"input": "システムにログインできません。パスワードを何度入力してもエラーになります。"},
    )
    assert result["outputs"]["label"] == "障害報告"
    assert "至急" in result["outputs"]["reply"]
    assert result["outputs"]["answer"].startswith("[障害報告]")

    churn = run_app(
        store,
        "churn-watch",
        {"tenure_months": 1, "monthly_spend": 16, "open_tickets": 9, "plan": "basic"},
    )
    assert churn["outputs"]["prediction"] == "yes"
    assert churn["outputs"]["risk"] == "high"


def test_batch_process_on_inquiry_dataset(isolated_root):
    from evalloop.studio.processes import run_process_on_dataset

    store = StudioStore(isolated_root / "studio")
    seed(store, repo_root=REPO_ROOT)
    summary = run_process_on_dataset(store, "inquiry-triage", "inquiry", limit=8)
    assert summary["n"] == 8
    assert summary["n_scored"] == 8
    assert summary["accuracy"] is not None
    assert summary["accuracy"] >= 0.5


def test_http_health_and_app_run(isolated_root):
    store = StudioStore(isolated_root / "studio")
    seed(store, repo_root=REPO_ROOT)
    status, headers, body = handle_request(store, "GET", "/api/health")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    html_status, html_headers, html_body = handle_request(store, "GET", "/")
    assert html_status == 200
    assert "evalloop studio" in html_body.decode("utf-8")
    import json

    status, _, body = handle_request(
        store,
        "POST",
        "/api/apps/inquiry-bot/run",
        json.dumps({"inputs": {"input": "ダークモードに対応してほしいです。"}}).encode("utf-8"),
    )
    assert status == 200
    payload = json.loads(body.decode("utf-8"))
    assert payload["outputs"]["label"] == "機能要望"
    status, _, body = handle_request(store, "GET", "/api/nope")
    assert status == 404


def test_cli_init_status_seed_and_app_run(isolated_root):
    root = isolated_root / "studio"
    init = runner.invoke(app, ["studio", "init", "--root", str(root)])
    assert init.exit_code == 0, init.output
    seeded = runner.invoke(app, ["studio", "seed", "--root", str(root)])
    assert seeded.exit_code == 0, seeded.output
    status = runner.invoke(app, ["studio", "status", "--root", str(root)])
    assert status.exit_code == 0
    assert "datasets" in status.output
    listed = runner.invoke(app, ["studio", "app", "list", "--root", str(root)])
    assert listed.exit_code == 0
    assert "inquiry-bot" in listed.output
    ran = runner.invoke(
        app,
        ["studio", "app", "run", "inquiry-bot", "--root", str(root), "--input", "契約書の解約条項を教えてください。"],
    )
    assert ran.exit_code == 0, ran.output
    assert "契約照会" in ran.output
    board = runner.invoke(app, ["studio", "train", "leaderboard", "job-churn-seed", "--root", str(root)])
    assert board.exit_code == 0, board.output
    assert "majority" in board.output


def test_cli_data_import(isolated_root):
    root = isolated_root / "studio"
    csv_path = isolated_root / "rows.csv"
    csv_path.write_text("name,y\na,1\nb,0\n", encoding="utf-8")
    result = runner.invoke(app, ["studio", "data", "import", str(csv_path), "--id", "rows", "--root", str(root)])
    assert result.exit_code == 0, result.output
    show = runner.invoke(app, ["studio", "data", "show", "rows", "--root", str(root)])
    assert show.exit_code == 0
    assert "n_rows" in show.output


def test_switch_and_map_steps(isolated_root):
    store = StudioStore(isolated_root / "studio")
    save_process(
        store,
        {
            "id": "switch-map",
            "outputs": ["lines", "flag"],
            "steps": [
                {"id": "items", "kind": "set", "value": ["a", "b"], "output": "items"},
                {
                    "id": "lines",
                    "kind": "map",
                    "over": "items",
                    "as": "item",
                    "collect": "line",
                    "steps": [{"id": "line", "kind": "prompt", "template": "- {{item}}", "output": "line"}],
                    "output": "lines",
                },
                {
                    "id": "flag",
                    "kind": "switch",
                    "on": "{{name}}",
                    "cases": {
                        "ada": [{"id": "yes", "kind": "set", "value": "ok", "output": "flag"}],
                        "default": [{"id": "no", "kind": "set", "value": "no", "output": "flag"}],
                    },
                    "output": "flag",
                },
            ],
        },
    )
    result = run_process(store, "switch-map", {"name": "ada"}, record=False)
    assert result["outputs"]["lines"] == ["- a", "- b"]
    assert result["outputs"]["flag"] == "ok"
