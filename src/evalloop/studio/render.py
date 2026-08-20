"""Mustache-lite ``{{path.to.value}}`` renderer used by process steps."""

from __future__ import annotations

import re
from typing import Any

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][\w.]*)\s*\}\}")


def lookup(state: dict[str, Any], path: str, default: Any = "") -> Any:
    current: Any = state
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            idx = int(part)
            if idx < 0 or idx >= len(current):
                return default
            current = current[idx]
        else:
            return default
    return current


def render(template: str, state: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        value = lookup(state, match.group(1), default="")
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return _stringify(value)
        return str(value)

    return _PLACEHOLDER.sub(repl, template)


def _stringify(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("title") or str(item)
                parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)
