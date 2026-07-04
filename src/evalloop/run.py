"""Wraps `npx promptfoo@<pinned version> eval` as a subprocess and records results.

Iron rules enforced here (README.md section 11):
    3. results are append-only: every run gets a fresh run_id directory and a
       new line in results/index.jsonl. Nothing here ever deletes or rewrites
       a prior run.
    4. promptfoo's disk cache stays on by default; --no-cache must be passed
       explicitly by the caller.
    5. the actual cost (summed from output.json) is always recorded in
       meta.json and index.jsonl, mirroring the pre-run estimate from build.py.
    7. `--share`/`--no-share`: we always pass --no-share so a misconfigured
       global promptfoo config can't accidentally upload results.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from evalloop.schemas import Config, load_config, parse_promptfoo_output

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTFOO_CONFIG_PATH = REPO_ROOT / "promptfoo" / "promptfooconfig.yaml"
VARIANTS_DIR = REPO_ROOT / "promptfoo" / "variants"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
INDEX_PATH = RESULTS_DIR / "index.jsonl"


class RunError(RuntimeError):
    pass


# promptfooのバージョンはここで一元的に固定する（@latestは使わない）。
# @latestだと (1) npmのその時点の最新版が毎回実行されるサプライチェーン露出、
# (2) 連載期間中に採点・出力仕様が変わる再現性ドリフト、の2つの問題がある。
# 更新手順: この値を上げる → `evalloop doctor` + `run --limit` スモーク → コミット
# （README「必要環境」参照）。実行時の実バージョンはmeta.jsonにも事後記録される。
PROMPTFOO_VERSION = "0.121.17"


def _npx_base_cmd() -> list[str]:
    """`subprocess.run(["npx", ...])` raises FileNotFoundError on Windows because
    npx is installed as `npx.cmd`, which bare CreateProcess (no shell) won't
    resolve via PATHEXT the way cmd.exe does. shutil.which() applies PATHEXT
    resolution and works cross-platform, so always go through it.
    """
    npx_path = shutil.which("npx")
    if npx_path is None:
        raise RunError("`npx` not found on PATH. Install Node.js 20.20+ (see README.md section 13).")
    return [npx_path, f"promptfoo@{PROMPTFOO_VERSION}"]


def new_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(2)}"


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_promptfoo_version() -> str:
    """Best-effort: promptfoo's CLI docs (as of this writing) don't document a
    `--version` flag explicitly, but Commander-based CLIs register one by
    default. TODO: confirm the definitive way to fetch this once `doctor` has
    been run once against a real npx install; fall back gracefully either way.
    """
    try:
        npx_cmd = _npx_base_cmd()
    except RunError:
        return "unknown"
    for args in (["--version"], ["-V"]):
        try:
            proc = subprocess.run(
                [*npx_cmd, *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                cwd=REPO_ROOT,
            )
            out = proc.stdout.strip() or proc.stderr.strip()
            if proc.returncode == 0 and out:
                return out.splitlines()[-1].strip()
        except (OSError, subprocess.TimeoutExpired):
            continue
    return "unknown"


def get_node_version() -> str:
    try:
        proc = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10
        )
        return proc.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def resolve_config_path(variant: str | None) -> Path:
    if variant is None:
        return PROMPTFOO_CONFIG_PATH
    variant_path = VARIANTS_DIR / f"{variant}.yaml"
    if not variant_path.exists():
        raise RunError(f"variant config not found: {variant_path} (run `evalloop optimize` first?)")
    return variant_path


@dataclass
class RunOutcome:
    run_id: str
    output_path: Path
    meta_path: Path
    meta: dict = field(default_factory=dict)


def run_promptfoo_eval(
    promptfoo_config_path: Path,
    output_path: Path,
    repeat: int,
    limit: int | None = None,
    no_cache: bool = False,
    max_concurrency: int | None = None,
    timeout_s: int = 1800,
) -> subprocess.CompletedProcess:
    """Shared by `evalloop run` and `evalloop doctor` (doctor uses limit=1 on a
    throwaway config). Flags confirmed against promptfoo.dev CLI docs:
    -c/--config, -o/--output, --repeat, --filter-first-n, --no-cache, --no-share.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_npx_base_cmd(),
        "eval",
        "-c",
        str(promptfoo_config_path),
        "-o",
        str(output_path),
        "--repeat",
        str(repeat),
        "--no-share",  # iron rule #7: never upload, even if global config defaults to share
    ]
    if limit is not None:
        cmd += ["--filter-first-n", str(limit)]
    if no_cache:
        cmd += ["--no-cache"]
    if max_concurrency is not None:
        cmd += ["-j", str(max_concurrency)]

    # promptfoo's own stdout/stderr, and anything it echoes from prompts/data,
    # is UTF-8 -- capturing with text=True but no explicit encoding falls back
    # to the OS locale codepage (cp932 on ja-JP Windows), which raises
    # UnicodeDecodeError deep in subprocess's background reader thread the
    # moment any non-cp932 byte shows up (e.g. curly quotes, em dashes,
    # certain CJK punctuation). Decode as UTF-8 explicitly and never raise on
    # a stray bad byte.
    return subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s
    )


def _actual_cost_from_output(output_path: Path) -> float:
    parsed = parse_promptfoo_output(output_path)
    return sum(r.cost or 0.0 for r in parsed.results)


def run(
    variant: str | None = None,
    repeat: int | None = None,
    limit: int | None = None,
    no_cache: bool = False,
    config_path: str | Path = REPO_ROOT / "config.yaml",
) -> RunOutcome:
    config: Config = load_config(config_path)
    promptfoo_config_path = resolve_config_path(variant)
    if not promptfoo_config_path.exists():
        raise RunError(f"{promptfoo_config_path} does not exist; run `evalloop build` first")

    effective_repeat = repeat if repeat is not None else config.run.repeat

    run_id = new_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "output.json"
    meta_path = run_dir / "meta.json"

    proc = run_promptfoo_eval(
        promptfoo_config_path,
        output_path,
        repeat=effective_repeat,
        limit=limit,
        no_cache=no_cache,
    )

    # Iron rule #3 (append-only ledger): even a total failure gets recorded in
    # meta.json/index.jsonl before we raise, so `results/` stays a complete
    # audit trail and no orphaned empty run_id directory is left behind.
    output_missing = not output_path.exists()
    actual_cost = _actual_cost_from_output(output_path) if output_path.exists() else 0.0
    prompt_path = REPO_ROOT / config.task.prompt_file

    resolved_config_path = Path(config_path).resolve()
    try:
        config_path_display = str(resolved_config_path.relative_to(REPO_ROOT))
    except ValueError:
        config_path_display = str(resolved_config_path)
    try:
        promptfoo_config_display = str(promptfoo_config_path.relative_to(REPO_ROOT))
    except ValueError:
        promptfoo_config_display = str(promptfoo_config_path)

    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_name": config.task.name,
        "answer_type": config.task.answer_type,
        "variant": variant,
        "config_path": config_path_display,  # so blog.py can generate accurate `--config` repro commands
        "promptfoo_config_path": promptfoo_config_display,
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
        "prompt_sha256": sha256_of_file(prompt_path),
        "repeat": effective_repeat,
        "limit": limit,
        "no_cache": no_cache,
        "models": [{"alias": m.alias, "provider": m.provider, "tier": m.tier} for m in config.models],
        "actual_cost_usd": actual_cost,
        "judge": {
            "provider": config.judge.provider,
            "calibration_status": "uncalibrated",  # updated in place by `evalloop calibrate`
            "agreement_rate": None,
        },
        "promptfoo_version": get_promptfoo_version(),
        "node_version": get_node_version(),
        "evalloop_command": (
            f"evalloop run{f' --config {config_path_display}' if config_path_display != 'config.yaml' else ''}"
            f"{f' --variant {variant}' if variant else ''}"
            f" --repeat {effective_repeat}{f' --limit {limit}' if limit else ''}{' --no-cache' if no_cache else ''}"
        ),
        "promptfoo_exit_code": proc.returncode,
        "promptfoo_stderr_tail": proc.stderr[-2000:] if proc.returncode != 0 else "",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    index_entry = {
        "run_id": run_id,
        "created_at": meta["created_at"],
        "task_name": config.task.name,
        "variant": variant,
        "actual_cost_usd": actual_cost,
        "promptfoo_exit_code": proc.returncode,
    }
    with INDEX_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

    print(f"[run] run_id={run_id}")
    print(f"[run] output -> {output_path}")
    print(f"[run] meta   -> {meta_path}")
    print(f"[run] actual cost: ${actual_cost:.4f}")
    if proc.returncode != 0:
        print(f"[run] WARNING: promptfoo exited with code {proc.returncode}; see meta.json stderr tail")

    if output_missing:
        raise RunError(
            f"promptfoo eval failed (exit {proc.returncode}) and produced no output.json "
            f"(recorded as run_id={run_id} in results/index.jsonl for the audit trail).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    return RunOutcome(run_id=run_id, output_path=output_path, meta_path=meta_path, meta=meta)


def view(directory: Path | None = None, port: int | None = None) -> subprocess.CompletedProcess:
    """Pass-through to `promptfoo view`. This only ever reads local results —
    it is not the same as `promptfoo share`, which is banned in this project.
    """
    cmd = [*_npx_base_cmd(), "view"]
    if directory is not None:
        cmd.append(str(directory))
    if port is not None:
        cmd += ["-p", str(port)]
    return subprocess.run(cmd, cwd=REPO_ROOT)
