"""File-backed studio catalog. Runtime files live under a single root.

Layout (all gitignored at the repo default ``studio/``)::

    <root>/catalog.json
    <root>/datasets/<id>/{meta.yaml,data.jsonl,profile.json}
    <root>/knowledge/<id>/{meta.yaml,docs.jsonl}
    <root>/models/<id>/{meta.yaml,artifact.json}
    <root>/processes/<id>.yaml
    <root>/apps/<id>.yaml
    <root>/jobs/<id>/{meta.yaml,leaderboard.json}
    <root>/runs/<id>.json
    <root>/sessions/<id>.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evalloop.paths import REPO_ROOT
from evalloop.studio.errors import StudioError
from evalloop.studio.ids import utcnow, validate_id

CATALOG_VERSION = 1
ENTITY_KINDS = ("datasets", "knowledge", "models", "processes", "apps", "jobs", "runs", "sessions")
SINGULAR = {
    "datasets": "dataset",
    "knowledge": "knowledge",
    "models": "model",
    "processes": "process",
    "apps": "app",
    "jobs": "job",
    "runs": "run",
    "sessions": "session",
}


def default_studio_root() -> Path:
    return REPO_ROOT / "studio"


def resolve_root(root: str | Path | None) -> Path:
    return Path(root) if root is not None else default_studio_root()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_write_yaml(path: Path, payload: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def empty_catalog() -> dict[str, Any]:
    catalog: dict[str, Any] = {"version": CATALOG_VERSION, "created_at": utcnow()}
    for kind in ENTITY_KINDS:
        catalog[kind] = {}
    return catalog


@dataclass(frozen=True)
class StudioPaths:
    root: Path

    @property
    def catalog(self) -> Path:
        return self.root / "catalog.json"

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.root / "datasets" / dataset_id

    def knowledge_dir(self, knowledge_id: str) -> Path:
        return self.root / "knowledge" / knowledge_id

    def model_dir(self, model_id: str) -> Path:
        return self.root / "models" / model_id

    def process_file(self, process_id: str) -> Path:
        return self.root / "processes" / f"{process_id}.yaml"

    def app_file(self, app_id: str) -> Path:
        return self.root / "apps" / f"{app_id}.yaml"

    def job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def run_file(self, run_id: str) -> Path:
        return self.root / "runs" / f"{run_id}.json"

    def session_file(self, session_id: str) -> Path:
        return self.root / "sessions" / f"{session_id}.json"


class StudioStore:
    """Create-or-load a studio workspace and keep catalog.json in sync."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.paths = StudioPaths(root=resolve_root(root).resolve())

    @property
    def root(self) -> Path:
        return self.paths.root

    def init(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.paths.catalog.exists():
            catalog = empty_catalog()
            atomic_write_json(self.paths.catalog, catalog)
            return catalog
        return self.load_catalog()

    def load_catalog(self) -> dict[str, Any]:
        if not self.paths.catalog.exists():
            raise StudioError(f"studio not initialized: {self.paths.catalog} (run `evalloop studio init`)")
        catalog = read_json(self.paths.catalog)
        if not isinstance(catalog, dict):
            raise StudioError(f"corrupt catalog: {self.paths.catalog}")
        for kind in ENTITY_KINDS:
            catalog.setdefault(kind, {})
        return catalog

    def save_catalog(self, catalog: dict[str, Any]) -> None:
        atomic_write_json(self.paths.catalog, catalog)

    def put(self, kind: str, entity_id: str, meta: dict[str, Any]) -> dict[str, Any]:
        if kind not in ENTITY_KINDS:
            raise StudioError(f"unknown catalog kind {kind!r}")
        validate_id(entity_id, SINGULAR.get(kind, kind))
        catalog = self.init()
        record = dict(meta)
        record["id"] = entity_id
        record.setdefault("created_at", utcnow())
        record["updated_at"] = utcnow()
        catalog[kind][entity_id] = record
        self.save_catalog(catalog)
        return record

    def get(self, kind: str, entity_id: str) -> dict[str, Any]:
        catalog = self.load_catalog()
        bucket = catalog.get(kind) or {}
        if entity_id not in bucket:
            raise StudioError(f"{SINGULAR.get(kind, kind)} {entity_id!r} not found")
        return dict(bucket[entity_id])

    def list(self, kind: str) -> list[dict[str, Any]]:
        catalog = self.load_catalog()
        bucket = catalog.get(kind) or {}
        return [dict(v) for _, v in sorted(bucket.items())]

    def delete(self, kind: str, entity_id: str) -> None:
        catalog = self.load_catalog()
        bucket = catalog.get(kind) or {}
        if entity_id not in bucket:
            raise StudioError(f"{SINGULAR.get(kind, kind)} {entity_id!r} not found")
        del bucket[entity_id]
        catalog[kind] = bucket
        self.save_catalog(catalog)
