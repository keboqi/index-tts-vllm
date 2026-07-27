"""Application service container and compatibility wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .schemas import SchemaInventory
from .schemas import from_legacy as schemas_from_legacy
from .services.features import (
    ModelService,
    SpeakerService,
    StableAudioService,
    VideoService,
)
from .services.translation import ArtifactStore, InMemorySessionRepository, TranslationOrchestrator
from .services.tts.factory import build_backend_registry
from .services.tts.registry import BackendRegistry


@dataclass(slots=True)
class RuntimeContainer:
    settings: Any
    backends: BackendRegistry
    translation: TranslationOrchestrator
    models: ModelService
    stable_audio: StableAudioService
    speakers: SpeakerService
    video: VideoService
    schemas: SchemaInventory
    concurrency: Any
    legacy: ModuleType

    @classmethod
    def from_legacy(cls, legacy: ModuleType) -> RuntimeContainer:
        backends = build_backend_registry(legacy)
        legacy.TTS_BACKEND_REGISTRY = backends
        sessions = InMemorySessionRepository(
            ttl_seconds=getattr(legacy, "ADVANCED_TRANSLATE_SESSION_TTL_SECONDS", 3600)
        )
        artifact_root = Path(legacy.TRANSLATE_SESSION_MEDIA_DIR)
        translation = TranslationOrchestrator(legacy, sessions, ArtifactStore(artifact_root))
        return cls(
            settings=legacy.SETTINGS,
            backends=backends,
            translation=translation,
            models=ModelService(legacy),
            stable_audio=StableAudioService(legacy),
            speakers=SpeakerService(legacy),
            video=VideoService(legacy),
            schemas=schemas_from_legacy(legacy),
            concurrency=legacy.CONCURRENCY,
            legacy=legacy,
        )
