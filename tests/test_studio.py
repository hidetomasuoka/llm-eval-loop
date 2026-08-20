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
    assert "app:inquiry-desk" in summary["imported"]
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


def test_chat_session_and_openai_compat(isolated_root):
    import json

    from evalloop.studio.apps import chat_app, openai_chat_completion

    store = StudioStore(isolated_root / "studio")
    seed(store, repo_root=REPO_ROOT)
    first = chat_app(store, "inquiry-bot", "システムにログインできません。パスワードを何度入力してもエラーになります。")
    assert first["outputs"]["label"] == "障害報告"
    assert first["session_id"]
    second = chat_app(store, "inquiry-bot", "契約書の解約条項を教えてください。", session_id=first["session_id"])
    assert second["session_id"] == first["session_id"]
    assert len(second["messages"]) == 4
    assert "システムにログインできません" in second["outputs"]["answer"]
    payload = openai_chat_completion(
        store,
        {"model": "inquiry-bot", "messages": [{"role": "user", "content": "ダークモードに対応してほしいです。"}]},
    )
    assert payload["object"] == "chat.completion"
    assert "機能要望" in payload["choices"][0]["message"]["content"]
    status, _, body = handle_request(
        store,
        "POST",
        "/v1/chat/completions",
        json.dumps({"model": "inquiry-bot", "messages": [{"role": "user", "content": "資料請求したいです。"}]}).encode(),
    )
    assert status == 200
    assert json.loads(body.decode())["choices"][0]["message"]["content"]


def test_subprocess_desk_and_cycle(isolated_root):
    store = StudioStore(isolated_root / "studio")
    seed(store, repo_root=REPO_ROOT)
    from evalloop.studio.apps import run_app

    result = run_app(store, "inquiry-desk", {"input": "システムにログインできません。パスワードを何度入力してもエラーになります。"})
    assert result["outputs"]["label"] == "障害報告"
    assert "ticket=" in result["outputs"]["answer"]
    save_process(
        store,
        {"id": "loop-a", "steps": [{"id": "go", "kind": "subprocess", "process": "loop-b", "output": "x"}]},
    )
    save_process(
        store,
        {"id": "loop-b", "steps": [{"id": "go", "kind": "subprocess", "process": "loop-a", "output": "x"}]},
    )
    with pytest.raises(StudioError, match="cycle"):
        run_process(store, "loop-a", {}, record=False)


def test_parse_memory_agent_and_graph(isolated_root):
    from evalloop.studio.processes import load_process, process_mermaid

    store = StudioStore(isolated_root / "studio")
    seed(store, repo_root=REPO_ROOT)
    save_process(
        store,
        {
            "id": "parse-mem",
            "outputs": ["parsed", "hist", "acted"],
            "steps": [
                {"id": "parsed", "kind": "parse", "field": "blob", "format": "json", "output": "parsed"},
                {"id": "hist", "kind": "memory", "output": "hist"},
                {
                    "id": "acted",
                    "kind": "agent",
                    "model": "inquiry-clf",
                    "knowledge": "inquiry-faq",
                    "feature": "input",
                    "query": "{{input}}",
                    "routes": {"障害報告": "retrieve", "契約照会": "retrieve", "機能要望": "retrieve", "その他": "finish"},
                    "tools": ["retrieve"],
                    "output": "acted",
                },
            ],
        },
    )
    result = run_process(
        store,
        "parse-mem",
        {
            "blob": '{"ok": true, "n": 1}',
            "input": "システムにログインできません。パスワードを何度入力してもエラーになります。",
            "history": [{"role": "user", "content": "前の話"}],
        },
        record=False,
    )
    assert result["outputs"]["parsed"] == {"ok": True, "n": 1}
    assert "user: 前の話" in result["outputs"]["hist"]
    assert result["outputs"]["acted"]["action"] == "retrieve"
    assert result["outputs"]["acted"]["result"]
    mermaid = process_mermaid(load_process(store, "inquiry-triage"))
    assert "flowchart TD" in mermaid
    assert "label" in mermaid
    save_process(
        store,
        {
            "id": "re-parse",
            "outputs": ["hit"],
            "steps": [{"id": "hit", "kind": "parse", "field": "input", "format": "regex", "pattern": r"id=(?P<id>\w+)", "output": "hit"}],
        },
    )
    parsed = run_process(store, "re-parse", {"input": "id=abc123"}, record=False)
    assert parsed["outputs"]["hit"]["id"] == "abc123"


def test_knowledge_file_import_and_automl_diagnostics(isolated_root):
    from evalloop.studio.automl import batch_score, compare_jobs, load_leaderboard
    from evalloop.studio.knowledge import import_knowledge_from_files, search

    store = StudioStore(isolated_root / "studio")
    seed(store, repo_root=REPO_ROOT)
    md = isolated_root / "faq.md"
    md.write_text("# FAQ\n\n## ログイン\nログインできないときは障害です。\n", encoding="utf-8")
    import_knowledge_from_files(store, "file-faq", [md], description="from file")
    hits = search(store, "file-faq", "ログインできない", k=1)
    assert hits and "障害" in hits[0]["text"]
    board = load_leaderboard(store, "job-churn-seed")
    assert board[0]["confusion"]["matrix"]
    assert "cv_mean" in board[0] or board[0]["model"] == "majority"
    compared = compare_jobs(store, ["job-inquiry-seed", "job-churn-seed"])
    assert len(compared["jobs"]) == 2
    scored = batch_score(store, "churn-clf", "churn-toy")
    assert scored["n"] == 24
    assert scored["accuracy"] is not None
    status, _, body = handle_request(store, "GET", "/api/jobs/job-churn-seed")
    assert status == 200
    import json

    payload = json.loads(body.decode())
    assert payload["leaderboard"]
    status, _, body = handle_request(store, "GET", "/api/processes/inquiry-triage/graph")
    assert status == 200
    assert "flowchart TD" in json.loads(body.decode())["mermaid"]


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
