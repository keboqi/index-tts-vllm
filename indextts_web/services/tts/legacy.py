"""Compatibility adapter around the production implementation during extraction."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from .base import BackendCapabilities, SynthesisRequest


class LegacyBackend:
    name = ""
    manager_attribute = ""
    capabilities = BackendCapabilities(False, False, False)

    def __init__(self, legacy: ModuleType) -> None:
        self.legacy = legacy

    @property
    def manager(self) -> Any:
        return getattr(self.legacy, self.manager_attribute)

    async def synthesize(self, request: SynthesisRequest) -> Path:
        raise NotImplementedError

    async def stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        if False:
            yield b""
        raise NotImplementedError(f"{self.name} streaming remains transport-owned")

    async def status(self) -> Mapping[str, Any]:
        if self.name == "index":
            manager = self.manager
            return {"ready": manager.is_ready(), **manager.vllm_status()}
        return await self.manager.status()

    async def shutdown(self) -> None:
        if self.name == "index":
            return
        await self.manager.shutdown()
