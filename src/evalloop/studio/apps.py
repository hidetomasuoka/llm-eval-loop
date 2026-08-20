"""Dify-like apps: a process + optional knowledge/model bindings, runnable via API."""

from __future__ import annotations

from typing import Any

import yaml

from evalloop.studio.errors import StudioError
from evalloop.studio.ids import new_entity_id, utcnow, validate_id
from evalloop.studio.processes import _format_history, run_process
from evalloop.studio.store import StudioStore, atomic_write_json, atomic_write_yaml, read_json


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
    if merged.get("history") and "history_text" not in merged:
        merged["history_text"] = _format_history(merged.get("history"))
    result = run_process(store, spec["process"], merged, record=True)
    result["app_id"] = app_id
    return result


def _reply_from_outputs(outputs: dict[str, Any]) -> str:
    for key in ("answer", "reply", "prediction", "label"):
        if key in outputs and outputs[key] not in (None, ""):
            return str(outputs[key])
    return yaml.safe_dump(outputs, allow_unicode=True, sort_keys=False).strip()


def load_session(store: StudioStore, session_id: str) -> dict[str, Any]:
    store.get("sessions", session_id)
    path = store.paths.session_file(session_id)
    if not path.exists():
        raise StudioError(f"session {session_id!r} is missing {path}")
    return read_json(path)


def chat_app(
    store: StudioStore,
    app_id: str,
    message: str,
    *,
    session_id: str | None = None,
    extra_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dify-like conversational turn. History is injected as history / history_text."""
    store.get("apps", app_id)
    if session_id:
        validate_id(session_id, "session")
        try:
            session = load_session(store, session_id)
            if session.get("app_id") != app_id:
                raise StudioError(f"session {session_id!r} belongs to app {session.get('app_id')!r}")
        except StudioError as exc:
            if "not found" not in str(exc) and "missing" not in str(exc):
                raise
            session = {"id": session_id, "app_id": app_id, "messages": [], "created_at": utcnow()}
    else:
        session_id = new_entity_id("sess")
        session = {"id": session_id, "app_id": app_id, "messages": [], "created_at": utcnow()}
    history = list(session.get("messages") or [])
    inputs = dict(extra_inputs or {})
    inputs["input"] = message
    inputs["history"] = history
    inputs["history_text"] = _format_history(history)
    result = run_app(store, app_id, inputs)
    reply = _reply_from_outputs(result.get("outputs") or {})
    history.append({"role": "user", "content": message, "at": utcnow()})
    history.append({"role": "assistant", "content": reply, "at": utcnow(), "outputs": result.get("outputs")})
    session["messages"] = history
    session["updated_at"] = utcnow()
    session["last_run_id"] = result.get("run_id")
    atomic_write_json(store.paths.session_file(session_id), session)
    store.put(
        "sessions",
        session_id,
        {"kind": "session", "app_id": app_id, "n_messages": len(history), "last_run_id": result.get("run_id")},
    )
    return {
        "session_id": session_id,
        "app_id": app_id,
        "reply": reply,
        "outputs": result.get("outputs"),
        "run_id": result.get("run_id"),
        "messages": history,
    }


def openai_chat_completion(store: StudioStore, payload: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-compatible /v1/chat/completions shim (Dify-style app serving)."""
    app_id = str(payload.get("model") or "")
    if not app_id:
        raise StudioError("chat.completions requires model (app id)")
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise StudioError("chat.completions requires messages")
    user_text = ""
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "user":
            user_text = str(item.get("content") or "")
            break
    if not user_text:
        raise StudioError("chat.completions needs a user message")
    session_id = payload.get("user") or payload.get("session_id")
    turn = chat_app(store, app_id, user_text, session_id=session_id if session_id else None)
    return {
        "id": turn["session_id"],
        "object": "chat.completion",
        "model": app_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": turn["reply"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
