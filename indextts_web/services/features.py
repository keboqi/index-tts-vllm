"""Facades for non-TTS feature managers."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(slots=True)
class ModelService:
    legacy: ModuleType

    def inventory(self) -> list[dict[str, Any]]:
        return self.legacy._loaded_model_inventory()

    def cuda_memory(self) -> dict[str, Any]:
        return self.legacy._cuda_memory_summary()

    async def unload(self, model_key: str) -> list[str]:
        async with self.legacy._model_manager_lock:
            return await self.legacy._run_blocking(self.legacy._unload_optional_model_sync, model_key)


@dataclass(slots=True)
class StableAudioService:
    legacy: ModuleType

    @property
    def manager(self) -> Any:
        return self.legacy.stable_audio3_manager

    def status(self) -> dict[str, Any]:
        return self.manager.status()

    def variants(self) -> list[dict[str, Any]]:
        return self.manager.list_models()


@dataclass(slots=True)
class SpeakerService:
    legacy: ModuleType

    def ready(self) -> bool:
        return self.legacy.speaker_api is not None

    async def list(self) -> Any:
        if not self.ready():
            return {}
        return await self.legacy.speaker_api.list_speakers()


@dataclass(slots=True)
class VideoService:
    legacy: ModuleType

    def downloaded(self) -> list[dict[str, Any]]:
        return self.legacy._list_downloaded_video_entries()

    def translated(self) -> list[dict[str, Any]]:
        return self.legacy._list_translated_video_entries()
