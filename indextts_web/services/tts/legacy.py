"""Compatibility adapter around the production implementation during extraction."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from .base import BackendCapabilities, SynthesisRequest


class LegacyBackend:
    name = ""
    capabilities = BackendCapabilities(False, False, False)

    def __init__(self, legacy: ModuleType) -> None:
        self.legacy = legacy

    async def synthesize(self, request: SynthesisRequest) -> Path:
        raise NotImplementedError

    async def stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        if False:
            yield b""
        raise NotImplementedError(f"{self.name} streaming remains transport-owned")

    async def status(self) -> Mapping[str, Any]:
        if self.name == "index":
            manager = self.legacy.tts_manager
            return {"ready": manager.is_ready(), **manager.vllm_status()}
        manager = getattr(self.legacy, f"{self.name}_backend_manager")
        return await manager.status()

    async def shutdown(self) -> None:
        if self.name == "index":
            return
        await getattr(self.legacy, f"{self.name}_backend_manager").shutdown()
