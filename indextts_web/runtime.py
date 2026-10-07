"""Runtime state shared by app assembly and the production endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

from .services.tts.factory import build_backend_registry
from .services.tts.registry import BackendRegistry


@dataclass(slots=True)
class RuntimeContainer:
    settings: Any
    backends: BackendRegistry
    concurrency: Any
    legacy: ModuleType

    @classmethod
    def from_legacy(cls, legacy: ModuleType) -> RuntimeContainer:
        backends = build_backend_registry(legacy)
        legacy.TTS_BACKEND_REGISTRY = backends
        return cls(
            settings=legacy.SETTINGS,
            backends=backends,
            concurrency=legacy.CONCURRENCY,
            legacy=legacy,
        )
