"""TTS backend selection and lifecycle."""

from __future__ import annotations

from collections.abc import Iterable

from .base import TTSBackend


class BackendRegistry:
    def __init__(self, backends: Iterable[TTSBackend], *, default: str = "index") -> None:
        self._backends = {backend.name: backend for backend in backends}
        if default not in self._backends:
            raise ValueError(f"unknown default backend: {default}")
        self.default = default

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))

    def get(self, name: str | None = None) -> TTSBackend:
        normalized = (name or self.default).strip().lower()
        try:
            return self._backends[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown TTS backend {normalized!r}; expected one of {self.names}") from exc

    async def shutdown(self) -> None:
        for backend in self._backends.values():
            await backend.shutdown()

