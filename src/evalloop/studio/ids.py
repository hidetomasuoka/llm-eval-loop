"""Stable identifiers and timestamps for studio entities."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from evalloop.studio.errors import StudioError

STUDIO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def validate_id(name: str, kind: str = "id") -> str:
    if not isinstance(name, str) or not STUDIO_ID_RE.match(name):
        raise StudioError(
            f"invalid {kind} {name!r}: must match {STUDIO_ID_RE.pattern} "
            "(lowercase alphanumerics and hyphens)"
        )
    return name
