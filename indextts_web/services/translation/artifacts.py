"""Translation artifact storage with containment checks."""

from __future__ import annotations

from pathlib import Path

from ...infrastructure.files import contained_path, safe_component


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str, kind: str, suffix: str) -> Path:
        extension = safe_component(suffix.lstrip("."), fallback="bin")
        filename = f"{safe_component(session_id)}_{safe_component(kind)}.{extension}"
        return contained_path(self.root, filename)

    def write(self, session_id: str, kind: str, suffix: str, content: bytes) -> Path:
        self.ensure()
        target = self.path(session_id, kind, suffix)
        target.write_bytes(content)
        return target

