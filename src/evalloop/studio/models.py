"""Model catalog: LLM providers (from config.yaml), trained AutoML, prompt variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalloop.paths import REPO_ROOT
from evalloop.studio.errors import StudioError
from evalloop.studio.ids import validate_id
from evalloop.studio.store import StudioStore, atomic_write_yaml


def register_llm_provider(
    store: StudioStore,
    alias: str,
    *,
    provider: str,
    tier: str = "unknown",
    price_in_per_mtok: float = 0.0,
    price_out_per_mtok: float = 0.0,
    supports_sampling_params: bool = True,
    description: str = "",
) -> dict[str, Any]:
    model_id = validate_id(alias, "model")
    meta = {
        "kind": "llm-provider",
        "provider": provider,
        "tier": tier,
        "price_in_per_mtok": price_in_per_mtok,
        "price_out_per_mtok": price_out_per_mtok,
        "supports_sampling_params": supports_sampling_params,
        "description": description or f"LLM provider from config.yaml ({provider})",
        "callable": False,
        "note": "Python never calls this provider directly. Use evalloop build/run (promptfoo).",
    }
    model_dir = store.paths.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_yaml(model_dir / "meta.yaml", meta)
    return store.put("models", model_id, meta)


def import_from_registry(store: StudioStore, repo_root: Path | None = None) -> list[dict[str, Any]]:
    from evalloop.schemas import load_global_config

    root = repo_root or REPO_ROOT
    cfg = load_global_config(root / "config.yaml")
    imported = []
    for model in cfg.models:
        imported.append(
            register_llm_provider(
                store,
                model.alias,
                provider=model.provider,
                tier=model.tier,
                price_in_per_mtok=model.price_in_per_mtok,
                price_out_per_mtok=model.price_out_per_mtok,
                supports_sampling_params=model.supports_sampling_params,
            )
        )
    return imported


def register_prompt_variant(
    store: StudioStore,
    model_id: str,
    *,
    task: str,
    path: str,
    description: str = "",
) -> dict[str, Any]:
    validate_id(model_id, "model")
    meta = {
        "kind": "prompt-variant",
        "task": task,
        "path": path,
        "description": description or f"Optimized prompt for task {task}",
        "callable": False,
    }
    model_dir = store.paths.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_yaml(model_dir / "meta.yaml", meta)
    return store.put("models", model_id, meta)


def list_models(store: StudioStore, kind: str | None = None) -> list[dict[str, Any]]:
    models = store.list("models")
    if kind:
        models = [m for m in models if m.get("kind") == kind]
    return models


def show_model(store: StudioStore, model_id: str) -> dict[str, Any]:
    meta = store.get("models", model_id)
    artifact = store.paths.model_dir(model_id) / "artifact.json"
    if artifact.exists():
        from evalloop.studio.automl import json_read

        payload = json_read(artifact)
        meta = dict(meta)
        meta["has_artifact"] = True
        meta["algorithm"] = payload.get("model", {}).get("algorithm") or meta.get("algorithm")
        meta["problem"] = payload.get("model", {}).get("problem") or meta.get("problem")
    return meta


def require_trained(store: StudioStore, model_id: str) -> dict[str, Any]:
    meta = store.get("models", model_id)
    if meta.get("kind") != "trained":
        raise StudioError(
            f"model {model_id!r} is kind={meta.get('kind')!r}; classify/predict needs a trained AutoML model"
        )
    return meta
