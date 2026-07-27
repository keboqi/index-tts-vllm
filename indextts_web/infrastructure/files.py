"""Path validation and artifact persistence primitives."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def safe_component(value: str, *, fallback: str = "artifact") -> str:
    cleaned = _SAFE_COMPONENT.sub("_", str(value or "")).strip("._")
    return cleaned or fallback


def contained_path(root: Path, *parts: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*(safe_component(part) for part in parts)).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError("path escapes artifact root")
    return candidate


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

