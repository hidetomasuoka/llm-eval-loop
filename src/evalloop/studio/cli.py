"""`evalloop studio ...` — manage processes, data, and models locally."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from evalloop.studio.errors import StudioError
from evalloop.studio.store import StudioStore, resolve_root

studio_app = typer.Typer(
    help=(
        "Manage processes, datasets, knowledge, AutoML models, and apps "
        "(Dify / LangChain / DataRobot analog). Local only; hosted LLMs stay on promptfoo."
    )
)
data_app = typer.Typer(help="Datasets (import, profile, list)")
knowledge_app = typer.Typer(help="Knowledge bases / retrieval")
model_app = typer.Typer(help="Model catalog (LLM registry + trained AutoML)")
process_app = typer.Typer(help="LangChain-like process (chain) specs")
apps_app = typer.Typer(help="Dify-like apps")
train_app = typer.Typer(help="DataRobot-like AutoML jobs")

studio_app.add_typer(data_app, name="data")
studio_app.add_typer(knowledge_app, name="knowledge")
studio_app.add_typer(model_app, name="model")
studio_app.add_typer(process_app, name="process")
studio_app.add_typer(apps_app, name="app")
studio_app.add_typer(train_app, name="train")

console = Console()

_ROOT = typer.Option(None, "--root", help="Studio workspace directory (default: <repo>/studio)")


def _store(root: str | None) -> StudioStore:
    return StudioStore(resolve_root(root))


def _die(exc: Exception) -> None:
    console.print(f"[bold red]studio error:[/bold red] {exc}")
    raise typer.Exit(1) from exc


def _print_json(payload: Any) -> None:
    console.print(json.dumps(payload, ensure_ascii=False, indent=2))


def _table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    table = Table(title=title)
    for col in columns:
        table.add_column(col, overflow="fold")
    if not rows:
        console.print(f"(no {title})")
        return
    for row in rows:
        table.add_row(*[str(row.get(col, "") or "") for col in columns])
    console.print(table)


@studio_app.command("init")
def studio_init(root: str = _ROOT) -> None:
    """Create an empty studio workspace (catalog.json)."""
    store = _store(root)
    store.init()
    console.print(f"initialized {store.root}")


@studio_app.command("seed")
def studio_seed(root: str = _ROOT) -> None:
    """Load demo datasets, train AutoML, and register inquiry/churn apps."""
    from evalloop.studio.seed import seed

    store = _store(root)
    try:
        summary = seed(store)
    except StudioError as e:
        _die(e)
        return
    console.print(f"seeded {store.root}")
    _print_json({k: v for k, v in summary.items() if k != "root"})


@studio_app.command("status")
def studio_status(root: str = _ROOT) -> None:
    """Show counts of datasets, models, processes, apps, jobs."""
    store = _store(root)
    try:
        catalog = store.load_catalog()
    except StudioError as e:
        _die(e)
        return
    rows = [{"kind": kind, "count": len(catalog.get(kind) or {})} for kind in
            ("datasets", "knowledge", "models", "processes", "apps", "jobs", "runs")]
    _table("studio", rows, ["kind", "count"])
    console.print(f"root: {store.root}")


@studio_app.command("serve")
def studio_serve(
    root: str = _ROOT,
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8787, "--port"),
) -> None:
    """Local dashboard + JSON API (never uploads)."""
    from evalloop.studio.serve import serve

    store = _store(root)
    try:
        store.init()
    except StudioError as e:
        _die(e)
        return
    server = serve(store, host=host, port=port)
    console.print(f"studio UI: http://{host}:{port}  (ctrl-c to stop)")
    console.print("[yellow]policy:[/yellow] this server stays on the local machine; no cloud share.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("stopped")
        server.server_close()


@data_app.command("list")
def data_list(root: str = _ROOT) -> None:
    store = _store(root)
    try:
        _table("datasets", store.list("datasets"), ["id", "n_rows", "origin", "target_hint", "description"])
    except StudioError as e:
        _die(e)


@data_app.command("import")
def data_import(
    path: Path = typer.Argument(..., exists=True, readable=True),
    dataset_id: str = typer.Option(..., "--id", help="Dataset id"),
    root: str = _ROOT,
    description: str = typer.Option("", "--description"),
) -> None:
    from evalloop.studio.data import import_dataset

    store = _store(root)
    try:
        meta = import_dataset(store, dataset_id, source=path, origin=f"file:{path}", description=description)
    except StudioError as e:
        _die(e)
        return
    _print_json(meta)


@data_app.command("from-task")
def data_from_task(
    task: str = typer.Argument(...),
    dataset_id: str = typer.Option(None, "--id"),
    root: str = _ROOT,
) -> None:
    from evalloop.studio.data import import_from_task

    store = _store(root)
    try:
        meta = import_from_task(store, task, dataset_id=dataset_id)
    except (StudioError, ValueError, RuntimeError) as e:
        _die(e)
        return
    _print_json(meta)


@data_app.command("show")
def data_show(
    dataset_id: str = typer.Argument(...),
    root: str = _ROOT,
    limit: int = typer.Option(5, "--limit"),
) -> None:
    from evalloop.studio.data import load_dataset, load_profile

    store = _store(root)
    try:
        meta, rows = load_dataset(store, dataset_id)
        profile = load_profile(store, dataset_id)
    except StudioError as e:
        _die(e)
        return
    _print_json({"meta": meta, "profile": profile, "sample": rows[:limit]})


@knowledge_app.command("list")
def knowledge_list(root: str = _ROOT) -> None:
    store = _store(root)
    try:
        _table("knowledge", store.list("knowledge"), ["id", "n_docs", "description"])
    except StudioError as e:
        _die(e)


@knowledge_app.command("import")
def knowledge_import(
    paths: list[Path] = typer.Argument(..., exists=True, readable=True),
    knowledge_id: str = typer.Option(..., "--id"),
    root: str = _ROOT,
    description: str = typer.Option("", "--description"),
    append: bool = typer.Option(False, "--append"),
) -> None:
    from evalloop.studio.knowledge import import_knowledge_from_files

    store = _store(root)
    try:
        meta = import_knowledge_from_files(store, knowledge_id, paths, description=description, append=append)
    except (StudioError, ValueError) as e:
        _die(e)
        return
    _print_json(meta)


@knowledge_app.command("search")
def knowledge_search(
    knowledge_id: str = typer.Argument(...),
    query: str = typer.Argument(...),
    k: int = typer.Option(3, "--k"),
    root: str = _ROOT,
) -> None:
    from evalloop.studio.knowledge import search

    store = _store(root)
    try:
        hits = search(store, knowledge_id, query, k=k)
    except StudioError as e:
        _die(e)
        return
    _print_json(hits)


@model_app.command("list")
def model_list(
    root: str = _ROOT,
    kind: str = typer.Option(None, "--kind", help="llm-provider / trained / prompt-variant"),
) -> None:
    from evalloop.studio.models import list_models

    store = _store(root)
    try:
        rows = list_models(store, kind=kind)
    except StudioError as e:
        _die(e)
        return
    _table("models", rows, ["id", "kind", "algorithm", "problem", "deployed", "description"])


@model_app.command("from-registry")
def model_from_registry(root: str = _ROOT) -> None:
    from evalloop.studio.models import import_from_registry

    store = _store(root)
    try:
        imported = import_from_registry(store)
    except (StudioError, ValueError, RuntimeError) as e:
        _die(e)
        return
    console.print(f"imported {len(imported)} llm providers from config.yaml")


@model_app.command("show")
def model_show(model_id: str = typer.Argument(...), root: str = _ROOT) -> None:
    from evalloop.studio.models import show_model

    store = _store(root)
    try:
        _print_json(show_model(store, model_id))
    except StudioError as e:
        _die(e)


@model_app.command("predict")
def model_predict(
    model_id: str = typer.Argument(...),
    root: str = _ROOT,
    json_row: str = typer.Option(..., "--json", help='Row as JSON, e.g. {"input":"..."}'),
) -> None:
    from evalloop.studio.automl import predict_row

    store = _store(root)
    try:
        row = json.loads(json_row)
        _print_json(predict_row(store, model_id, row))
    except (StudioError, json.JSONDecodeError) as e:
        _die(e)


@process_app.command("list")
def process_list(root: str = _ROOT) -> None:
    store = _store(root)
    try:
        _table("processes", store.list("processes"), ["id", "name", "n_steps", "description"])
    except StudioError as e:
        _die(e)


@process_app.command("show")
def process_show(process_id: str = typer.Argument(...), root: str = _ROOT) -> None:
    from evalloop.studio.processes import load_process

    store = _store(root)
    try:
        _print_json(load_process(store, process_id))
    except StudioError as e:
        _die(e)


@process_app.command("graph")
def process_graph(process_id: str = typer.Argument(...), root: str = _ROOT) -> None:
    from evalloop.studio.processes import load_process, process_mermaid

    store = _store(root)
    try:
        spec = load_process(store, process_id)
        console.print(process_mermaid(spec))
    except StudioError as e:
        _die(e)


@process_app.command("run")
def process_run(
    process_id: str = typer.Argument(...),
    root: str = _ROOT,
    json_inputs: str = typer.Option(None, "--json", help="Inputs as JSON object"),
    text: str = typer.Option(None, "--input", help="Shorthand for {\"input\": TEXT}"),
    dataset_id: str = typer.Option(None, "--dataset", help="Batch-run over a dataset"),
    limit: int = typer.Option(None, "--limit"),
) -> None:
    from evalloop.studio.processes import run_process, run_process_on_dataset

    store = _store(root)
    try:
        if dataset_id:
            result = run_process_on_dataset(store, process_id, dataset_id, limit=limit)
        else:
            inputs: dict[str, Any] = json.loads(json_inputs) if json_inputs else {}
            if text is not None:
                inputs.setdefault("input", text)
            result = run_process(store, process_id, inputs)
    except (StudioError, json.JSONDecodeError) as e:
        _die(e)
        return
    _print_json(result if dataset_id else {"run_id": result.get("run_id"), "outputs": result.get("outputs"), "trace": result.get("trace")})


@apps_app.command("list")
def app_list(root: str = _ROOT) -> None:
    store = _store(root)
    try:
        _table("apps", store.list("apps"), ["id", "name", "process", "description"])
    except StudioError as e:
        _die(e)


@apps_app.command("run")
def app_run(
    app_id: str = typer.Argument(...),
    root: str = _ROOT,
    json_inputs: str = typer.Option(None, "--json"),
    text: str = typer.Option(None, "--input"),
) -> None:
    from evalloop.studio.apps import run_app

    store = _store(root)
    try:
        inputs: dict[str, Any] = json.loads(json_inputs) if json_inputs else {}
        if text is not None:
            inputs.setdefault("input", text)
        result = run_app(store, app_id, inputs)
    except (StudioError, json.JSONDecodeError) as e:
        _die(e)
        return
    _print_json({"run_id": result.get("run_id"), "app_id": result.get("app_id"), "outputs": result.get("outputs")})


@apps_app.command("chat")
def app_chat(
    app_id: str = typer.Argument(...),
    text: str = typer.Argument(...),
    root: str = _ROOT,
    session_id: str = typer.Option(None, "--session"),
) -> None:
    from evalloop.studio.apps import chat_app

    store = _store(root)
    try:
        result = chat_app(store, app_id, text, session_id=session_id)
    except StudioError as e:
        _die(e)
        return
    _print_json({"session_id": result["session_id"], "reply": result["reply"], "outputs": result.get("outputs")})


@train_app.command("start")
def train_start(
    dataset_id: str = typer.Argument(...),
    target: str = typer.Option(..., "--target"),
    root: str = _ROOT,
    seed: int = typer.Option(0, "--seed"),
    job_id: str = typer.Option(None, "--job-id"),
) -> None:
    from evalloop.studio.automl import train_job

    store = _store(root)
    try:
        job = train_job(store, dataset_id, target, job_id=job_id, seed=seed)
    except StudioError as e:
        _die(e)
        return
    _print_json(job)


@train_app.command("leaderboard")
def train_leaderboard(job_id: str = typer.Argument(...), root: str = _ROOT) -> None:
    from evalloop.studio.store import read_json

    store = _store(root)
    try:
        store.get("jobs", job_id)
        board = read_json(store.paths.job_dir(job_id) / "leaderboard.json")
    except StudioError as e:
        _die(e)
        return
    rows = []
    for item in board:
        hold = item.get("holdout") or {}
        rows.append(
            {
                "model": item.get("model"),
                "score": item.get("score"),
                "accuracy": hold.get("accuracy", ""),
                "macro_f1": hold.get("macro_f1", ""),
                "mae": hold.get("mae", ""),
                "rmse": hold.get("rmse", ""),
            }
        )
    _table("leaderboard", rows, ["model", "score", "accuracy", "macro_f1", "mae", "rmse"])


@train_app.command("compare")
def train_compare(jobs: str = typer.Argument(..., help="Comma-separated job ids"), root: str = _ROOT) -> None:
    from evalloop.studio.automl import compare_jobs

    store = _store(root)
    job_ids = [part.strip() for part in jobs.split(",") if part.strip()]
    try:
        _print_json(compare_jobs(store, job_ids))
    except StudioError as e:
        _die(e)


@train_app.command("score")
def train_score(
    model_id: str = typer.Argument(...),
    dataset_id: str = typer.Option(..., "--dataset"),
    root: str = _ROOT,
    limit: int = typer.Option(None, "--limit"),
) -> None:
    from evalloop.studio.automl import batch_score

    store = _store(root)
    try:
        result = batch_score(store, model_id, dataset_id, limit=limit)
    except StudioError as e:
        _die(e)
        return
    _print_json({k: v for k, v in result.items() if k != "predictions"} | {"n_predictions": len(result.get("predictions") or [])})
