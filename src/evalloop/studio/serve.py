"""Local studio HTTP API + dashboard. Never uploads; never calls hosted LLMs."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from evalloop.studio.apps import chat_app, load_app, openai_chat_completion, run_app
from evalloop.studio.automl import load_leaderboard, predict_row
from evalloop.studio.data import load_dataset, load_profile
from evalloop.studio.errors import StudioError
from evalloop.studio.knowledge import search
from evalloop.studio.processes import load_process, process_mermaid, run_process
from evalloop.studio.store import StudioStore

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>evalloop studio</title>
<style>
:root {
  --bg: #161410;
  --panel: #211c16;
  --ink: #f3eadf;
  --muted: #b7a894;
  --line: #3a3228;
  --accent: #e2a336;
  --ok: #7cb389;
  --bad: #d37a6a;
  --mono: "IBM Plex Mono", "ui-monospace", monospace;
  --sans: "Iowan Old Style", "Palatino Linotype", "Hiragino Mincho ProN", serif;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); }
body { min-height: 100vh; }
header {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 28px 32px 16px; border-bottom: 1px solid var(--line);
}
header h1 { margin: 0; font-size: 28px; letter-spacing: 0.02em; font-weight: 600; }
header p { margin: 0; color: var(--muted); font-size: 14px; }
main { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 90px); }
nav { border-right: 1px solid var(--line); padding: 20px; }
nav button {
  display: block; width: 100%; text-align: left; background: transparent;
  color: var(--ink); border: 0; border-left: 3px solid transparent;
  padding: 10px 12px; margin: 4px 0; font: inherit; cursor: pointer;
}
nav button.active { border-left-color: var(--accent); background: #2a241c; }
section { padding: 24px 32px 48px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.card {
  background: var(--panel); border: 1px solid var(--line); padding: 16px 16px 14px;
  min-height: 120px;
}
.card h3 { margin: 0 0 6px; font-size: 16px; }
.card .id { font-family: var(--mono); color: var(--accent); font-size: 12px; }
.card p { margin: 8px 0 0; color: var(--muted); font-size: 13px; line-height: 1.45; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { border-bottom: 1px solid var(--line); text-align: left; padding: 8px 6px; vertical-align: top; }
th { color: var(--muted); font-weight: 500; }
textarea, input, select {
  width: 100%; background: #12100c; color: var(--ink); border: 1px solid var(--line);
  padding: 8px; font-family: var(--mono); font-size: 13px;
}
button.run {
  margin-top: 10px; background: var(--accent); color: #1a140c; border: 0;
  padding: 8px 14px; font-weight: 700; cursor: pointer;
}
pre {
  background: #12100c; border: 1px solid var(--line); padding: 12px; overflow: auto;
  font-family: var(--mono); font-size: 12px; line-height: 1.4;
}
.kpis { display: flex; gap: 16px; margin: 0 0 18px; }
.kpi { background: var(--panel); border: 1px solid var(--line); padding: 12px 16px; min-width: 110px; }
.kpi b { display: block; font-size: 22px; }
.kpi span { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>evalloop studio</h1>
  <p>プロセス · データ · モデル — Dify / LangChain / DataRobot 相当のローカル管理面</p>
</header>
<main>
  <nav id="nav"></nav>
  <section id="view">読み込み中…</section>
</main>
<script>
const views = ["概要","会話","データ","ナレッジ","モデル","プロセス","アプリ","学習ジョブ","実行"];
let catalog = {datasets:{}, knowledge:{}, models:{}, processes:{}, apps:{}, jobs:{}, runs:{}, sessions:{}};
let current = "概要";
let sessionId = "";

async function load() {
  catalog = await (await fetch("/api/catalog")).json();
  const nav = document.getElementById("nav");
  nav.innerHTML = views.map(v => `<button data-v="${v}">${v}</button>`).join("");
  nav.onclick = (e) => { if (e.target.dataset.v) { current = e.target.dataset.v; render(); } };
  render();
}
function entries(kind) { return Object.values(catalog[kind] || {}); }
function kpi(label, n) { return `<div class="kpi"><b>${n}</b><span>${label}</span></div>`; }
function cards(kind, extra) {
  return `<div class="grid">${entries(kind).map(e => `
    <div class="card">
      <div class="id">${e.id}</div>
      <h3>${e.name || e.kind || e.id}</h3>
      <p>${e.description || extra(e) || ""}</p>
    </div>`).join("") || "<p>まだありません。 <code>evalloop studio seed</code> を実行してください。</p>"}</div>`;
}
function table(rows, cols) {
  if (!rows.length) return "<p>なし</p>";
  return `<table><thead><tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr></thead><tbody>${
    rows.map(r => `<tr>${cols.map(c=>`<td>${esc(fmt(r[c]))}</td>`).join("")}</tr>`).join("")
  }</tbody></table>`;
}
function fmt(v) { return v === undefined || v === null ? "" : (typeof v === "object" ? JSON.stringify(v) : String(v)); }
function esc(s) { return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;"); }

function render() {
  [...document.querySelectorAll("nav button")].forEach(b => b.classList.toggle("active", b.dataset.v === current));
  const v = document.getElementById("view");
  if (current === "概要") {
    v.innerHTML = `<div class="kpis">
      ${kpi("データ", entries("datasets").length)}
      ${kpi("モデル", entries("models").length)}
      ${kpi("プロセス", entries("processes").length)}
      ${kpi("アプリ", entries("apps").length)}
      ${kpi("学習", entries("jobs").length)}
    </div>
    <p>ホストされた LLM にはここから直接アクセスしません。分類・予測は AutoML 成果物、生成は template/echo、評価は <code>evalloop run</code>（promptfoo）です。</p>
    ${runPanel()}`;
    bindRun();
    return;
  }
  if (current === "会話") { v.innerHTML = chatPanel(); bindChat(); return; }
  if (current === "データ") v.innerHTML = cards("datasets", e => `${e.n_rows || 0} 行 · origin ${e.origin || ""}`);
  if (current === "ナレッジ") v.innerHTML = cards("knowledge", e => `${e.n_docs || 0} docs`);
  if (current === "モデル") v.innerHTML = table(entries("models"), ["id","kind","algorithm","problem","deployed","description"]);
  if (current === "プロセス") { v.innerHTML = processPanel(); bindProcess(); return; }
  if (current === "アプリ") v.innerHTML = cards("apps", e => `process ${e.process || ""}`);
  if (current === "学習ジョブ") { v.innerHTML = jobsPanel(); bindJobs(); return; }
  if (current === "実行") v.innerHTML = table(entries("runs"), ["id","kind","process_id","dataset_id","accuracy"]);
}

function runPanel() {
  const apps = entries("apps");
  const opts = apps.map(a => `<option value="${a.id}">${a.id}</option>`).join("");
  return `<h2>アプリを実行</h2>
    <p>JSON 入力を渡してローカルでチェーンを動かします。</p>
    <select id="appId">${opts}</select>
    <textarea id="payload" rows="8">{
  "input": "ログインできません。パスワードを何度入力してもエラーになります。"
}</textarea>
    <button class="run" id="runBtn">実行</button>
    <pre id="out">（結果がここに出ます）</pre>`;
}
function bindRun() {
  const btn = document.getElementById("runBtn");
  if (!btn) return;
  btn.onclick = async () => {
    const id = document.getElementById("appId").value;
    const payload = JSON.parse(document.getElementById("payload").value || "{}");
    const res = await fetch("/api/apps/" + id + "/run", {
      method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({inputs: payload})
    });
    const body = await res.json();
    document.getElementById("out").textContent = JSON.stringify(body, null, 2);
    catalog = await (await fetch("/api/catalog")).json();
  };
}
function chatPanel() {
  const apps = entries("apps");
  const opts = apps.map(a => `<option value="${a.id}">${a.id}</option>`).join("");
  return `<h2>会話</h2>
    <p>Dify 相当のセッション付きチャット。履歴はプロセスへ <code>history_text</code> として渡します。</p>
    <select id="chatApp">${opts}</select>
    <input id="chatMsg" placeholder="メッセージ"/>
    <button class="run" id="chatBtn">送信</button>
    <pre id="chatLog">session: (new)</pre>`;
}
function bindChat() {
  const btn = document.getElementById("chatBtn");
  if (!btn) return;
  btn.onclick = async () => {
    const id = document.getElementById("chatApp").value;
    const message = document.getElementById("chatMsg").value;
    const res = await fetch("/api/apps/" + id + "/chat", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({message, session_id: sessionId || undefined})
    });
    const body = await res.json();
    sessionId = body.session_id || sessionId;
    const lines = (body.messages || []).map(m => m.role + ": " + m.content).join("\\n");
    document.getElementById("chatLog").textContent = "session " + sessionId + "\\n" + lines;
    catalog = await (await fetch("/api/catalog")).json();
  };
}
function processPanel() {
  const opts = entries("processes").map(p => `<option value="${p.id}">${p.id}</option>`).join("");
  return `${cards("processes", e => `${e.n_steps || 0} steps`)}
    <h2>フロー</h2>
    <select id="procId">${opts}</select>
    <button class="run" id="graphBtn">DAG を表示</button>
    <pre id="graphOut"></pre>`;
}
function bindProcess() {
  const btn = document.getElementById("graphBtn");
  if (!btn) return;
  btn.onclick = async () => {
    const id = document.getElementById("procId").value;
    const body = await (await fetch("/api/processes/" + id + "/graph")).json();
    document.getElementById("graphOut").textContent = body.mermaid || JSON.stringify(body, null, 2);
  };
}
function jobsPanel() {
  return `${table(entries("jobs"), ["id","dataset_id","target","problem","winner","n_train","n_test"])}
    <h2>リーダーボード</h2>
    <select id="jobId">${entries("jobs").map(j => `<option value="${j.id}">${j.id}</option>`).join("")}</select>
    <button class="run" id="boardBtn">表示</button>
    <pre id="boardOut"></pre>`;
}
function bindJobs() {
  const btn = document.getElementById("boardBtn");
  if (!btn) return;
  btn.onclick = async () => {
    const id = document.getElementById("jobId").value;
    const body = await (await fetch("/api/jobs/" + id)).json();
    document.getElementById("boardOut").textContent = JSON.stringify(body.leaderboard || body, null, 2);
  };
}
load().catch(err => { document.getElementById("view").textContent = String(err); });
</script>
</body>
</html>
"""


def _json_ok(payload: Any, status: int = 200) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return status, {"Content-Type": "application/json; charset=utf-8"}, body


def _json_err(message: str, status: int = 400) -> tuple[int, dict[str, str], bytes]:
    return _json_ok({"error": message}, status=status)


def handle_request(
    store: StudioStore,
    method: str,
    path: str,
    body: bytes = b"",
    query: dict[str, list[str]] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    query = query or {}
    if method == "GET" and path in {"/", "/index.html"}:
        return 200, {"Content-Type": "text/html; charset=utf-8"}, DASHBOARD_HTML.encode("utf-8")
    try:
        return _dispatch(store, method, path, body, query)
    except StudioError as e:
        return _json_err(str(e), 400)
    except json.JSONDecodeError as e:
        return _json_err(f"invalid JSON: {e}", 400)
    except FileNotFoundError as e:
        return _json_err(str(e), 404)


def _read_json(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise StudioError("JSON body must be an object")
    return payload


def _parts(path: str) -> list[str]:
    return [p for p in path.split("/") if p]


def _dispatch(
    store: StudioStore,
    method: str,
    path: str,
    body: bytes,
    query: dict[str, list[str]],
) -> tuple[int, dict[str, str], bytes]:
    parts = _parts(path)
    if method == "GET" and path == "/api/health":
        return _json_ok({"ok": True, "root": str(store.root)})
    if method == "GET" and path == "/api/catalog":
        return _json_ok(store.load_catalog())
    if method == "GET" and parts == ["api", "datasets"]:
        return _json_ok(store.list("datasets"))
    if method == "GET" and len(parts) == 3 and parts[:2] == ["api", "datasets"]:
        meta, rows = load_dataset(store, parts[2])
        limit = int((query.get("limit") or ["20"])[0])
        return _json_ok({"meta": meta, "rows": rows[:limit], "n": len(rows)})
    if method == "GET" and len(parts) == 4 and parts[:2] == ["api", "datasets"] and parts[3] == "profile":
        return _json_ok(load_profile(store, parts[2]))
    if method == "GET" and parts == ["api", "models"]:
        return _json_ok(store.list("models"))
    if method == "GET" and parts == ["api", "processes"]:
        return _json_ok(store.list("processes"))
    if method == "GET" and len(parts) == 3 and parts[:2] == ["api", "processes"]:
        return _json_ok(load_process(store, parts[2]))
    if method == "GET" and parts == ["api", "apps"]:
        return _json_ok(store.list("apps"))
    if method == "GET" and parts == ["api", "jobs"]:
        return _json_ok(store.list("jobs"))
    if method == "GET" and parts == ["api", "runs"]:
        return _json_ok(store.list("runs"))
    if method == "GET" and parts == ["api", "knowledge"]:
        return _json_ok(store.list("knowledge"))
    if method == "GET" and parts == ["api", "sessions"]:
        return _json_ok(store.list("sessions"))
    if method == "GET" and len(parts) == 4 and parts[:2] == ["api", "processes"] and parts[3] == "graph":
        spec = load_process(store, parts[2])
        return _json_ok({"id": parts[2], "mermaid": process_mermaid(spec)})
    if method == "GET" and len(parts) == 3 and parts[:2] == ["api", "jobs"]:
        job = store.get("jobs", parts[2])
        board = load_leaderboard(store, parts[2])
        return _json_ok({**job, "leaderboard": board})
    if method == "GET" and len(parts) == 3 and parts[:2] == ["api", "apps"]:
        return _json_ok(load_app(store, parts[2]))
    if method == "POST" and len(parts) == 4 and parts[:2] == ["api", "apps"] and parts[3] == "run":
        payload = _read_json(body)
        return _json_ok(run_app(store, parts[2], payload.get("inputs") or payload))
    if method == "POST" and len(parts) == 4 and parts[:2] == ["api", "apps"] and parts[3] == "chat":
        payload = _read_json(body)
        return _json_ok(
            chat_app(
                store,
                parts[2],
                str(payload.get("message") or payload.get("input") or ""),
                session_id=payload.get("session_id"),
                extra_inputs=payload.get("inputs") if isinstance(payload.get("inputs"), dict) else None,
            )
        )
    if method == "POST" and path == "/v1/chat/completions":
        return _json_ok(openai_chat_completion(store, _read_json(body)))
    if method == "POST" and len(parts) == 4 and parts[:2] == ["api", "processes"] and parts[3] == "run":
        payload = _read_json(body)
        return _json_ok(run_process(store, parts[2], payload.get("inputs") or payload))
    if method == "POST" and len(parts) == 4 and parts[:2] == ["api", "models"] and parts[3] == "predict":
        payload = _read_json(body)
        row = payload.get("row") or payload.get("inputs") or payload
        return _json_ok(predict_row(store, parts[2], row))
    if method == "POST" and len(parts) == 4 and parts[:2] == ["api", "knowledge"] and parts[3] == "search":
        payload = _read_json(body)
        hits = search(store, parts[2], str(payload.get("query") or ""), k=int(payload.get("k") or 3))
        return _json_ok({"hits": hits})
    return _json_err(f"not found: {method} {path}", 404)


class StudioHandler(BaseHTTPRequestHandler):
    store: StudioStore

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _write(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        status, headers, body = handle_request(self.store, "GET", parsed.path, b"", parse_qs(parsed.query))
        self._write(status, headers, body)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length) if length else b""
        status, headers, body = handle_request(self.store, "POST", parsed.path, payload, parse_qs(parsed.query))
        self._write(status, headers, body)


def serve(store: StudioStore, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    handler = type("BoundStudioHandler", (StudioHandler,), {"store": store})
    server = ThreadingHTTPServer((host, port), handler)
    return server
