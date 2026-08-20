"""Dify-like apps: a process + optional knowledge/model bindings, runnable via API."""

from __future__ import annotations

from typing import Any

import yaml

from evalloop.studio.errors import StudioError
from evalloop.studio.ids import validate_id
from evalloop.studio.processes import run_process
from evalloop.studio.store import StudioStore, atomic_write_yaml


def save_app(store: StudioStore, spec: dict[str, Any]) -> dict[str, Any]:
    if "id" not in spec or "process" not in spec:
        raise StudioError("app spec requires id and process")
    app_id = validate_id(str(spec["id"]), "app")
    process_id = validate_id(str(spec["process"]), "process")
    store.get("processes", process_id)
    payload = dict(spec)
    payload["id"] = app_id
    payload["process"] = process_id
    payload.setdefault("name", app_id)
    payload.setdefault("description", "")
    payload.setdefault("inputs", [])
    path = store.paths.app_file(app_id)
    atomic_write_yaml(path, payload)
    return store.put(
        "apps",
        app_id,
        {
            "kind": "app",
            "name": payload.get("name"),
            "description": payload.get("description", ""),
            "process": process_id,
            "knowledge": payload.get("knowledge"),
            "models": payload.get("models") or {},
        },
    )


def load_app(store: StudioStore, app_id: str) -> dict[str, Any]:
    store.get("apps", app_id)
    path = store.paths.app_file(app_id)
    if not path.exists():
        raise StudioError(f"app {app_id!r} is missing {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def run_app(store: StudioStore, app_id: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = load_app(store, app_id)
    merged = dict(inputs or {})
    defaults = spec.get("defaults") or {}
    for key, value in defaults.items():
        merged.setdefault(key, value)
    if spec.get("knowledge") and "knowledge" not in merged:
        merged["knowledge"] = spec["knowledge"]
    result = run_process(store, spec["process"], merged, record=True)
    result["app_id"] = app_id
    return result
