"""evalloop CLI entry point (`uv run evalloop ...`). See README.md.

Every task-scoped command takes `--task NAME` (falling back to the
EVALLOOP_TASK env var, then config.yaml's default_task). Tasks live under
tasks/<name>/ -- see issue #47.
"""

from __future__ import annotations

import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from evalloop import build as build_mod
from evalloop import paths as paths_mod
from evalloop import report as report_mod
from evalloop import run as run_mod
from evalloop.schemas import Config, SchemaError, load_task, parse_promptfoo_output, restrict_models

app = typer.Typer(
    add_completion=False,
    help=(
        "llm-eval-loop: promptfoo runs+grades multi-model evals; evalloop owns dataset "
        "safety, judge calibration, failure analysis, GEPA optimization, blog export. "
        "Tasks are self-contained workspaces under tasks/<name>/ (select with --task).\n\n"
        "NOTE: `promptfoo share` (cloud upload) is never used by this project - use "
        "`evalloop view` to browse local results instead."
    ),
)
task_app = typer.Typer(help="Manage task workspaces under tasks/")
app.add_typer(task_app, name="task")
console = Console()

_TASK_OPTION = typer.Option(
    None, "--task", help="Task workspace under tasks/ (default: $EVALLOOP_TASK, then config.yaml default_task)"
)

_REQUIRED_ENV_BY_PREFIX = {
    "anthropic:": "ANTHROPIC_API_KEY",
    "openai:": "OPENAI_API_KEY",
    "gemini:": "GEMINI_API_KEY",
    "google:": "GEMINI_API_KEY",
}


def _env_key_for_provider(provider: str) -> str | None:
    for prefix, env_key in _REQUIRED_ENV_BY_PREFIX.items():
        if provider.startswith(prefix):
            return env_key
    return None


def _check_ollama_reachable(timeout: float = 3.0) -> bool:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _node_version_ok(version_str: str) -> bool | None:
    """promptfoo's own runtime check (confirmed by actually running it) requires
    `^20.20.0 || >=22.22.0` — i.e. 20.20.x-20.x, OR >=22.22.0. Versions in
    between (21.x, or 22.0.0-22.21.x) are rejected by promptfoo itself with a
    hard error, not just an npm engine warning.
    """
    match = re.search(r"v?(\d+)\.(\d+)", version_str)
    if not match:
        return None
    major, minor = int(match.group(1)), int(match.group(2))
    if major == 20:
        return minor >= 20
    if major == 22:
        return minor >= 22
    if major > 22:
        return True
    return False


def _load_task_or_exit(task: str | None, models: str | None = None):
    """Resolve --task into (Config, TaskPaths), optionally narrowing models."""
    try:
        cfg, paths = load_task(task)
        if models:
            cfg = restrict_models(cfg, [a.strip() for a in models.split(",") if a.strip()])
        return cfg, paths
    except (SchemaError, paths_mod.TaskNotFoundError) as e:
        console.print(f"[bold red]config error:[/bold red] {e}")
        raise typer.Exit(1) from e


@task_app.command("init")
def task_init(
    name: str = typer.Argument(..., help="New task name (lowercase alphanumerics and hyphens)"),
    answer_type: str = typer.Option("label", "--answer-type", help="label / json / text"),
) -> None:
    """Scaffold tasks/<name>/ (task.yaml + prompts/ + PROVENANCE.md). golden.jsonl is up to you -- it stays out of git."""
    try:
        paths = paths_mod.init_task_workspace(name, answer_type=answer_type)
    except (paths_mod.TaskExistsError, paths_mod.TaskNotFoundError, ValueError) as e:
        console.print(f"[bold red]task init failed:[/bold red] {e}")
        raise typer.Exit(1) from e
    console.print(f"created {paths.task_dir}")
    console.print("next steps:")
    console.print(f"  1. edit {paths.task_config} (labels, models, judge)")
    console.print(f"  2. edit {paths.prompt_file}" + (" and prompts/judge_rubric.txt" if answer_type == "text" else ""))
    console.print(f"  3. put your dataset at {paths.golden} (gitignored -- document it in PROVENANCE.md)")
    console.print(f"  4. uv run evalloop build --task {name}")


@task_app.command("list")
def task_list() -> None:
    """List task workspaces under tasks/ (with dataset presence and default marker)."""
    try:
        from evalloop.schemas import load_global_config

        global_config = load_global_config(paths_mod.REPO_ROOT / "config.yaml")
        default_task = global_config.default_task
    except SchemaError:
        default_task = None

    names = paths_mod.list_tasks()
    if not names:
        console.print("no tasks found under tasks/")
        return
    table = Table(title="tasks")
    table.add_column("task", overflow="fold")
    table.add_column("dataset", overflow="fold")
    table.add_column("default", overflow="fold")
    for name in names:
        tp = paths_mod.TaskPaths(root=paths_mod.REPO_ROOT, task=name)
        if tp.golden.exists():
            dataset = "[green]present[/green]"
        else:
            dataset = "[yellow]missing (see PROVENANCE.md)[/yellow]"
        table.add_row(name, dataset, "*" if name == default_task else "")
    console.print(table)


@app.command()
def diagnose(
    answers: str = typer.Option(
        None,
        "--answers",
        help="Non-interactive mode for tests: comma-separated Q1,Q2,Q3 (1=yes/2=no for Q1/Q3; Q2=1-5)",
    ),
) -> None:
    """Interactive checklist: symptom → APO granularity → recommended optimize.method (docs/APO_GUIDE.md)."""
    from evalloop import diagnose as diagnose_mod

    parsed: list[int] | None = None
    if answers is not None:
        try:
            parsed = diagnose_mod.parse_answers(answers)
        except ValueError as e:
            console.print(f"[bold red]diagnose failed:[/bold red] {e}")
            raise typer.Exit(1) from e
    try:
        diagnose_mod.run_diagnose(answers=parsed)
    except ValueError as e:
        console.print(f"[bold red]diagnose failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def doctor(task: str = _TASK_OPTION) -> None:
    """Check Node/promptfoo/Ollama/API-key connectivity; run one tiny eval per provider."""
    console.print(
        "[bold yellow]policy reminder:[/bold yellow] this project never runs `promptfoo share` "
        "(no cloud upload). Use `evalloop view` for local results.\n"
    )
    cfg, _paths = _load_task_or_exit(task)

    node_version = run_mod.get_node_version()
    ok = _node_version_ok(node_version)
    node_flag = (
        "[green]ok[/green]"
        if ok
        else ("[red]unsupported, need ^20.20.0 or >=22.22.0[/red]" if ok is False else "[yellow]unknown[/yellow]")
    )
    console.print(f"node --version: {node_version}  {node_flag}")

    pf_version = run_mod.get_promptfoo_version()
    console.print(f"promptfoo version (pinned: {run_mod.PROMPTFOO_VERSION}): {pf_version}")

    table = Table(title="providers")
    table.add_column("alias", overflow="fold")
    table.add_column("provider", overflow="fold")
    table.add_column("env key", overflow="fold")
    table.add_column("env status", overflow="fold")
    table.add_column("note", overflow="fold")

    for m in cfg.models:
        env_key = _env_key_for_provider(m.provider)
        if env_key is None:
            env_status = "n/a"
        elif os.environ.get(env_key):
            env_status = "[green]present[/green]"
        else:
            env_status = "[red]MISSING[/red]"

        note = ""
        if m.provider.startswith("ollama:"):
            note = "[green]ollama reachable[/green]" if _check_ollama_reachable() else "[red]ollama NOT reachable[/red]"

        table.add_row(m.alias, m.provider, env_key or "-", env_status, note)
    console.print(table)

    _smoke_test_providers(cfg)


def _smoke_test_providers(cfg: Config) -> None:
    """One trivial eval covering every configured provider at once (no assertions,
    so `success` just reflects whether the API call itself worked).
    """
    with tempfile.TemporaryDirectory(prefix="evalloop-doctor-") as tmp:
        tmp_path = Path(tmp)
        providers = []
        for m in cfg.models:
            # mirror build.py: models with supports_sampling_params=false reject
            # temperature with HTTP 400, so sending it here makes the smoke test
            # report a false connectivity failure for exactly those models
            provider_config: dict = {}
            if m.supports_sampling_params:
                provider_config["temperature"] = 0.0
            provider_config["max_tokens"] = 16
            providers.append({"id": m.provider, "label": m.alias, "config": provider_config})
        smoke_config = {
            "description": "evalloop doctor smoke test",
            "providers": providers,
            "prompts": ["Reply with a single word: OK. Input: {{input}}"],
            "tests": [
                {
                    "description": "smoke",
                    "vars": {
                        "case_id": "doctor-smoke-0001",
                        "input": "connectivity check",
                    },
                }
            ],
        }
        config_path = tmp_path / "smoke.yaml"
        output_path = tmp_path / "smoke_output.json"
        config_path.write_text(yaml.safe_dump(smoke_config, allow_unicode=True), encoding="utf-8")

        console.print("\nrunning 1-case smoke eval against every provider (this calls real APIs)...")
        try:
            proc = run_mod.run_promptfoo_eval(config_path, output_path, repeat=1, timeout_s=120)
        except Exception as e:  # noqa: BLE001 - doctor must never crash, only report
            console.print(f"[red]smoke test could not run npx promptfoo: {e}[/red]")
            return

        if not output_path.exists():
            console.print(f"[red]smoke test produced no output.[/red]\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            return

        parsed = parse_promptfoo_output(output_path)
        table = Table(title="smoke test results")
        table.add_column("alias", overflow="fold")
        table.add_column("status", overflow="fold")
        table.add_column("detail", overflow="fold")
        for r in parsed.results:
            if r.error:
                table.add_row(r.alias or "?", "[red]error[/red]", str(r.error)[:120])
            elif r.passed is False:
                table.add_row(r.alias or "?", "[yellow]ran, flagged fail[/yellow]", (r.reason or "")[:120])
            else:
                table.add_row(r.alias or "?", "[green]ok[/green]", (r.output or "")[:60])
        console.print(table)
        for w in parsed.warnings:
            console.print(f"[yellow]parser warning: {w}[/yellow]")


@app.command()
def build(
    task: str = _TASK_OPTION,
    models: str = typer.Option(
        None, help="Comma-separated alias subset of this task's models (e.g. CI smoke: --models gptoss20b)"
    ),
    allow_same_judge: bool = typer.Option(
        False, help="Iron rule #2 override: allow llm-rubric judge == an evaluated model"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost-estimate confirmation prompt"),
    shuffle_demos: int = typer.Option(
        None,
        "--shuffle-demos",
        help=(
            "After a normal build, write N demoshuffle promptfoo variants "
            "(seeds 0..N-1) for demo-order sensitivity checks (requires {{demos}})"
        ),
    ),
) -> None:
    """tasks/<name>/golden.jsonl + task.yaml -> data/build/<name>/ + promptfoo/<name>/promptfooconfig.yaml."""
    cfg, paths = _load_task_or_exit(task, models)
    try:
        build_mod.build(
            cfg,
            paths,
            allow_same_judge=allow_same_judge,
            yes=yes,
            confirm_fn=lambda msg: typer.confirm(msg),
            shuffle_demos=shuffle_demos,
        )
    except (SchemaError, build_mod.BuildError) as e:
        console.print(f"[bold red]build failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def run(
    task: str = _TASK_OPTION,
    variant: str = typer.Option(
        None, help="Name of a promptfoo/<task>/variants/{name}.yaml to run instead of the base config"
    ),
    repeat: int = typer.Option(None, help="Override run.repeat from the config"),
    limit: int = typer.Option(None, help="Only run the first N test cases (maps to promptfoo --filter-first-n)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable promptfoo's disk cache for this run"),
    split: str = typer.Option(
        "test",
        help=(
            "Which holdout to evaluate: 'test' (default) or 'dev'. dev requires split=='dev' "
            "cases in golden.jsonl and a rebuild; it feeds the optimize shipping gate without "
            "consuming the test split."
        ),
    ),
    timeout: int = typer.Option(
        None,
        help=(
            "Kill promptfoo if it runs longer than this many seconds. Default: wait "
            "indefinitely -- large batches on slow local models can legitimately take hours, "
            "and promptfoo doesn't write output.json incrementally, so a timeout discards "
            "all progress."
        ),
    ),
) -> None:
    """Run `npx promptfoo eval` against the built config and record results/<task>/runs/{run_id}/."""
    cfg, paths = _load_task_or_exit(task)
    try:
        run_mod.run(
            cfg, paths, variant=variant, repeat=repeat, limit=limit, no_cache=no_cache, timeout_s=timeout, split=split
        )
    except (SchemaError, run_mod.RunError) as e:
        console.print(f"[bold red]run failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def view(
    directory: str = typer.Argument(None, help="promptfoo output/config directory to view"),
    port: int = typer.Option(None, help="Port for the local viewer server"),
) -> None:
    """Launch `promptfoo view` (local-only browser; never uploads anything)."""
    run_mod.view(directory=Path(directory) if directory else None, port=port)


@app.command()
def report(
    run_id: str = typer.Argument(..., help="run_id under results/<task>/runs/"),
    task: str = _TASK_OPTION,
) -> None:
    """results/<task>/runs/{run_id}/output.json -> results/<task>/reports/{run_id}.md"""
    _cfg, paths = _load_task_or_exit(task)
    try:
        report_mod.report(run_id, paths)
    except report_mod.ReportError as e:
        console.print(f"[bold red]report failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def calibrate(
    task: str = _TASK_OPTION,
    run_id: str = typer.Option(None, help="If set, cross-check this run's gradingResults instead of re-grading"),
) -> None:
    """Compare the LLM judge against the task's human_labels.jsonl; warn below judge.agreement_threshold."""
    from evalloop import calibrate as calibrate_mod

    cfg, paths = _load_task_or_exit(task)
    try:
        calibrate_mod.calibrate(cfg, paths, run_id=run_id)
    except (SchemaError, calibrate_mod.CalibrateError) as e:
        console.print(f"[bold red]calibrate failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def failures(
    run_id: str = typer.Argument(..., help="run_id under results/<task>/runs/"),
    task: str = _TASK_OPTION,
) -> None:
    """Extract failing cases from a run into failures.jsonl + the task's notes.csv template."""
    from evalloop import analyze as analyze_mod

    _cfg, paths = _load_task_or_exit(task)
    try:
        analyze_mod.failures(run_id, paths)
    except analyze_mod.AnalyzeError as e:
        console.print(f"[bold red]failures failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def cluster(
    task: str = _TASK_OPTION,
    notes: str = typer.Option(None, help="Path to notes.csv (default: the task's notes.csv)"),
) -> None:
    """LLM-draft a failure taxonomy from notes.csv into the task's taxonomy.draft.yaml (never overwrites taxonomy.yaml)."""
    from evalloop import analyze as analyze_mod

    cfg, paths = _load_task_or_exit(task)
    try:
        analyze_mod.cluster(cfg, paths, notes_path=notes)
    except analyze_mod.AnalyzeError as e:
        console.print(f"[bold red]cluster failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def pivot(
    run_id: str = typer.Argument(..., help="run_id under results/<task>/runs/"),
    task: str = _TASK_OPTION,
) -> None:
    """Failure category x model cross-tab -> results/<task>/reports/pivot_{run_id}.md."""
    from evalloop import analyze as analyze_mod

    _cfg, paths = _load_task_or_exit(task)
    try:
        analyze_mod.pivot(run_id, paths)
    except analyze_mod.AnalyzeError as e:
        console.print(f"[bold red]pivot failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def optimize(
    task: str = _TASK_OPTION,
    force: bool = typer.Option(
        False,
        "--force",
        help="Demote preflight errors (small train set, label coverage) to warnings and continue. "
        "Use only when you accept the overfitting risk.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost-estimate confirmation prompt"),
) -> None:
    """Run the configured optimizer (optimize.method) on split=='train' only; then run/report/compare on the variant."""
    from evalloop import optimize as optimize_mod

    cfg, paths = _load_task_or_exit(task)
    try:
        optimize_mod.optimize(cfg, paths, force=force, yes=yes, confirm_fn=lambda msg: typer.confirm(msg))
    except optimize_mod.OptimizeError as e:
        console.print(f"[bold red]optimize failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def compare(
    runs: str = typer.Option(..., help="Two or more run_ids, comma-separated: A,B or A,B,C,..."),
    task: str = _TASK_OPTION,
) -> None:
    """Compare 2+ runs -> results/<task>/reports/compare_*.md (2-run delta or multi-run matrix)."""
    from evalloop import optimize as optimize_mod

    _cfg, paths = _load_task_or_exit(task)
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    if len(run_ids) < 2:
        console.print("[bold red]compare failed:[/bold red] --runs must be at least 'A,B'")
        raise typer.Exit(1)
    try:
        optimize_mod.compare(run_ids, paths)
    except optimize_mod.OptimizeError as e:
        console.print(f"[bold red]compare failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def blog(
    runs: str = typer.Option(..., help="One or more run_ids, comma-separated: A or A,B or A,B,C"),
    slug: str = typer.Option(None, help="Override blog.slug_prefix from the task config"),
    task: str = _TASK_OPTION,
) -> None:
    """Publish-guarded export of figures/tables/conditions/article draft -> blog/<task>/{date}_{slug}/."""
    from evalloop import blog as blog_mod

    cfg, paths = _load_task_or_exit(task)
    run_ids = [r.strip() for r in runs.split(",")]
    try:
        blog_mod.blog(cfg, paths, run_ids=run_ids, slug=slug)
    except (SchemaError, blog_mod.BlogGuardError) as e:
        console.print(f"[bold red]blog failed:[/bold red] {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
