"""evalloop CLI entry point (`uv run evalloop ...`). See README.md section 7."""

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
from evalloop import report as report_mod
from evalloop import run as run_mod
from evalloop.schemas import Config, SchemaError, load_config, parse_promptfoo_output

app = typer.Typer(
    add_completion=False,
    help=(
        "llm-eval-loop: promptfoo runs+grades multi-model evals; evalloop owns dataset "
        "safety, judge calibration, failure analysis, GEPA optimization, blog export.\n\n"
        "NOTE: `promptfoo share` (cloud upload) is never used by this project - use "
        "`evalloop view` to browse local results instead."
    ),
)
console = Console()

DEFAULT_CONFIG_PATH = "config.yaml"

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


def _load_config_or_exit(config_path: str) -> Config:
    try:
        return load_config(config_path)
    except SchemaError as e:
        console.print(f"[bold red]config error:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def doctor(config: str = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config.yaml")) -> None:
    """Check Node/promptfoo/Ollama/API-key connectivity; run one tiny eval per provider."""
    console.print(
        "[bold yellow]policy reminder:[/bold yellow] this project never runs `promptfoo share` "
        "(no cloud upload). Use `evalloop view` for local results.\n"
    )
    cfg = _load_config_or_exit(config)

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
            "tests": [{"description": "smoke", "vars": {"input": "connectivity check"}}],
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
    config: str = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config.yaml"),
    allow_same_judge: bool = typer.Option(
        False, help="Iron rule #2 override: allow llm-rubric judge == an evaluated model"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost-estimate confirmation prompt"),
) -> None:
    """golden.jsonl + config.yaml -> data/build/tests_*.yaml + promptfoo/promptfooconfig.yaml."""
    try:
        build_mod.build(
            config_path=config,
            allow_same_judge=allow_same_judge,
            yes=yes,
            confirm_fn=lambda msg: typer.confirm(msg),
        )
    except (SchemaError, build_mod.BuildError) as e:
        console.print(f"[bold red]build failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def run(
    variant: str = typer.Option(None, help="Name of a promptfoo/variants/{name}.yaml to run instead of the base config"),
    repeat: int = typer.Option(None, help="Override run.repeat from config.yaml"),
    limit: int = typer.Option(None, help="Only run the first N test cases (maps to promptfoo --filter-first-n)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable promptfoo's disk cache for this run"),
    config: str = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config.yaml"),
) -> None:
    """Run `npx promptfoo eval` against the built config and record results/runs/{run_id}/."""
    try:
        run_mod.run(variant=variant, repeat=repeat, limit=limit, no_cache=no_cache, config_path=config)
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
def report(run_id: str = typer.Argument(..., help="run_id under results/runs/")) -> None:
    """results/runs/{run_id}/output.json -> results/reports/{run_id}.md"""
    try:
        report_mod.report(run_id)
    except report_mod.ReportError as e:
        console.print(f"[bold red]report failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def calibrate(
    config: str = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config.yaml"),
    run_id: str = typer.Option(None, help="If set, cross-check this run's gradingResults instead of re-grading"),
) -> None:
    """Compare the LLM judge against data/human_labels.jsonl; warn below judge.agreement_threshold."""
    from evalloop import calibrate as calibrate_mod

    try:
        calibrate_mod.calibrate(config_path=config, run_id=run_id)
    except (SchemaError, calibrate_mod.CalibrateError) as e:
        console.print(f"[bold red]calibrate failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def failures(run_id: str = typer.Argument(..., help="run_id under results/runs/")) -> None:
    """Extract failing cases from a run into failures.jsonl + a notes.csv template."""
    from evalloop import analyze as analyze_mod

    try:
        analyze_mod.failures(run_id)
    except analyze_mod.AnalyzeError as e:
        console.print(f"[bold red]failures failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def cluster(
    notes: str = typer.Option("data/notes.csv", help="Path to notes.csv"),
    config: str = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config.yaml"),
) -> None:
    """LLM-draft a failure taxonomy from notes.csv into data/taxonomy.draft.yaml (never overwrites taxonomy.yaml)."""
    from evalloop import analyze as analyze_mod

    try:
        analyze_mod.cluster(notes_path=notes, config_path=config)
    except analyze_mod.AnalyzeError as e:
        console.print(f"[bold red]cluster failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def pivot(run_id: str = typer.Argument(..., help="run_id under results/runs/")) -> None:
    """Failure category x model cross-tab -> reports/pivot_{run_id}.md."""
    from evalloop import analyze as analyze_mod

    try:
        analyze_mod.pivot(run_id)
    except analyze_mod.AnalyzeError as e:
        console.print(f"[bold red]pivot failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def optimize(config: str = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config.yaml")) -> None:
    """Run dspy GEPA on split=='train' only; then run/report/compare on the optimized variant."""
    from evalloop import optimize as optimize_mod

    try:
        optimize_mod.optimize(config_path=config)
    except optimize_mod.OptimizeError as e:
        console.print(f"[bold red]optimize failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def compare(runs: str = typer.Option(..., help="Two run_ids, comma-separated: A,B")) -> None:
    """before/after comparison of two runs -> reports/compare_A_B.md."""
    from evalloop import optimize as optimize_mod

    run_ids = [r.strip() for r in runs.split(",")]
    if len(run_ids) != 2:
        console.print("[bold red]compare failed:[/bold red] --runs must be exactly 'A,B'")
        raise typer.Exit(1)
    try:
        optimize_mod.compare(run_ids[0], run_ids[1])
    except optimize_mod.OptimizeError as e:
        console.print(f"[bold red]compare failed:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command()
def blog(
    runs: str = typer.Option(..., help="One or two run_ids, comma-separated: A or A,B"),
    slug: str = typer.Option(None, help="Override blog.slug_prefix from config.yaml"),
    config: str = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config.yaml"),
) -> None:
    """Publish-guarded export of figures/tables/conditions/article draft -> blog/{date}_{slug}/."""
    from evalloop import blog as blog_mod

    run_ids = [r.strip() for r in runs.split(",")]
    try:
        blog_mod.blog(run_ids=run_ids, slug=slug, config_path=config)
    except (SchemaError, blog_mod.BlogGuardError) as e:
        console.print(f"[bold red]blog failed:[/bold red] {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
