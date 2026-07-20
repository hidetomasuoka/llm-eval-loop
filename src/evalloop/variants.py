"""Optimized-variant artifacts: promptfoo variant config generation, the
auto-generated slug/summary identity, and the optimized/index.jsonl appender.

Extracted from evalloop.optimize (kept there as re-exports for backward
compatibility).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import yaml

from evalloop.optimizers.base import OptimizeError
from evalloop.paths import TaskPaths

# ---------------------------------------------------------------------------
# variant config generation (reroots every file:// reference one level
# deeper, since promptfoo/variants/{name}.yaml lives one directory below
# promptfoo/promptfooconfig.yaml)
# ---------------------------------------------------------------------------


def _reroot_file_refs(obj, prefix: str):
    if isinstance(obj, dict):
        return {k: _reroot_file_refs(v, prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_reroot_file_refs(v, prefix) for v in obj]
    if isinstance(obj, str) and obj.startswith("file://"):
        return "file://" + prefix + obj[len("file://") :]
    return obj


def to_variant_relpath(target: Path, variants_dir: Path) -> str:
    rel = os.path.relpath(target, start=variants_dir)
    return rel.replace(os.sep, "/")


def build_variant_config(target_alias: str, task_path: Path, paths: TaskPaths, split: str = "test") -> dict:
    """split='dev' derives from promptfooconfig.dev.yaml (tests -> tests_dev),
    producing the variant config the optimize shipping gate runs on."""
    source = paths.promptfoo_config_dev if split == "dev" else paths.promptfoo_config
    if not source.exists():
        hint = (
            f"add split=='dev' cases to {paths.golden} and run `evalloop build --task {paths.task}`"
            if split == "dev"
            else f"run `evalloop build --task {paths.task}` first"
        )
        raise OptimizeError(f"{source} not found; {hint}")
    base_config = yaml.safe_load(source.read_text(encoding="utf-8"))
    variant_config = _reroot_file_refs(base_config, prefix="../")
    variant_config["prompts"] = [f"file://{to_variant_relpath(task_path, paths.variants_dir)}"]
    variant_config["description"] = f"{base_config.get('description', '')} (optimized: {target_alias})"
    return variant_config


# ---------------------------------------------------------------------------
# variant slug / summary (auto-generated identity for optimized artifacts)
# ---------------------------------------------------------------------------

_SLUG_MAX_LEN = 40
_PARAM_KEY_SHORT = {
    "val_ratio": "val",
    "seed": "seed",
    "breadth": "br",
    "depth": "d",
    "init_temperature": "temp",
    "population_size": "pop",
    "generations": "gen",
}
# {method}-{YYYYMMDD-HHMMSS} or {method}-{YYYYMMDD-HHMMSS}-{slug}
_OPTIMIZED_DIR_RE = re.compile(r"^[^-]+-\d{8}-\d{6}(?:-(.+))?$")


def _slug_from_dir_name(name: str) -> str | None:
    """Extract the auto slug from an optimized dir name, if present."""
    m = _OPTIMIZED_DIR_RE.match(name)
    if not m:
        return None
    return m.group(1)


def _occupied_slugs(alias_dir: Path) -> set[str]:
    if not alias_dir.is_dir():
        return set()
    found: set[str] = set()
    for child in alias_dir.iterdir():
        if not child.is_dir():
            continue
        slug = _slug_from_dir_name(child.name)
        if slug:
            found.add(slug)
    return found


def _sanitize_slug_part(value: str) -> str:
    # allow '.' so float params stay readable (e.g. val0.2)
    s = re.sub(r"[^a-z0-9.]+", "-", str(value).lower())
    return s.strip("-.")


def _short_param_key(key: str) -> str:
    if key in _PARAM_KEY_SHORT:
        return _PARAM_KEY_SHORT[key]
    cleaned = re.sub(r"[^a-z0-9]+", "", str(key).lower())
    return cleaned[:6] if cleaned else "p"


def _format_param_token(key: str, value) -> str | None:
    """Turn a scalar param into a compact slug token; skip nested/long values."""
    short = _short_param_key(key)
    if isinstance(value, bool):
        return f"{short}{int(value)}"
    if isinstance(value, int):
        return f"{short}{value}"
    if isinstance(value, float):
        return f"{short}{value:g}"
    if isinstance(value, str) and len(value) <= 16 and not re.search(r"[\s/]", value):
        part = _sanitize_slug_part(value)
        return f"{short}{part}" if part else None
    return None


def _instructions_hash(base_instructions: str, optimized_instructions: str) -> str:
    payload = f"{base_instructions}\0{optimized_instructions}".encode()
    return hashlib.sha256(payload).hexdigest()[:4]


def _make_variant_slug(
    *,
    auto: str,
    params: dict,
    train_case_count: int,
    base_instructions: str = "",
    optimized_instructions: str = "",
    occupied: set[str] | None = None,
) -> str:
    """Build a short deterministic slug: auto + scalar params + n{train}.

    On collision with `occupied`, append a 4-hex hash of the instructions diff.
    """
    parts = [_sanitize_slug_part(auto) or "auto"]
    for key in sorted(params):
        if key == "auto":
            continue
        token = _format_param_token(key, params[key])
        if token:
            parts.append(token)
    train_token = f"n{train_case_count}"
    parts.append(train_token)
    slug = "-".join(p for p in parts if p)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    # truncate earlier segments first so the trailing n{train} identity stays
    if len(slug) > _SLUG_MAX_LEN:
        max_prefix_len = _SLUG_MAX_LEN - len(train_token) - 1
        prefix = "-".join(parts[:-1])[:max_prefix_len].rstrip("-.")
        slug = f"{prefix}-{train_token}" if prefix else train_token

    occupied = occupied or set()
    if slug not in occupied:
        return slug
    suffix = _instructions_hash(base_instructions, optimized_instructions)
    # keep n{train} at the end after the collision hash when possible
    max_prefix_len = _SLUG_MAX_LEN - len(train_token) - 5  # -{4hex}-nN
    if max_prefix_len > 0:
        prefix = "-".join(parts[:-1])[:max_prefix_len].rstrip("-.")
        if prefix:
            return f"{prefix}-{suffix}-{train_token}"
    return f"{train_token}-{suffix}"[:_SLUG_MAX_LEN]


def _make_variant_summary(
    *,
    method: str,
    auto: str,
    params: dict,
    train_case_count: int,
    base_instructions: str,
    optimized_instructions: str,
) -> str:
    """One-line auto summary: settings + instruction char-length delta."""
    extras: list[str] = []
    for key in sorted(params):
        if key == "auto":
            continue
        value = params[key]
        if isinstance(value, (int, float, bool)):
            extras.append(f"{key}={value}")
        elif isinstance(value, str) and len(value) <= 32:
            one_line = re.sub(r"\s+", " ", value).strip()
            if one_line:
                extras.append(f"{key}={one_line}")
    extra_s = (" " + " ".join(extras)) if extras else ""
    return (
        f"{method} auto={auto}{extra_s} train={train_case_count}; "
        f"instructions {len(base_instructions)}→{len(optimized_instructions)} chars"
    )


def _append_optimized_index(paths: TaskPaths, entry: dict) -> None:
    paths.optimized_dir.mkdir(parents=True, exist_ok=True)
    with paths.optimized_index.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
