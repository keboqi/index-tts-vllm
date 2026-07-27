"""Stage-oriented facade over the production translation pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any

from .artifacts import ArtifactStore
from .sessions import SessionRepository


class TranslationStage(str, Enum):
    INGEST = "ingest"
    PREPROCESS = "preprocess"
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    SEGMENT = "segment"
    SYNTHESIZE = "synthesize"
    DURATION_MATCH = "duration_match"
    MIX = "mix"
    EXPORT = "export"


@dataclass(frozen=True, slots=True)
class TranslationProgress:
    stage: TranslationStage
    message: str = ""
    progress: float | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def as_event(self) -> dict[str, Any]:
        return {
            "type": "progress",
            "stage": self.stage.value,
            "message": self.message,
            "progress": self.progress,
            **self.data,
        }


ProgressCallback = Callable[[TranslationProgress], Awaitable[None]]


class TranslationOrchestrator:
    def __init__(
        self,
        legacy: ModuleType,
        sessions: SessionRepository,
        artifacts: ArtifactStore,
    ) -> None:
        self.legacy = legacy
        self.sessions = sessions
        self.artifacts = artifacts

    async def synthesize(self, *args: Any, **kwargs: Any) -> Any:
        return await self.legacy._synthesize_translated_audio(*args, **kwargs)

    async def transcribe_with_gemini(self, *args: Any, **kwargs: Any) -> Any:
        return await self.legacy._gemini_transcribe_translate(*args, **kwargs)

    async def emit(
        self,
        callback: ProgressCallback | None,
        stage: TranslationStage,
        *,
        message: str = "",
        progress: float | None = None,
        **payload: Any,
    ) -> None:
        if callback is not None:
            await callback(
                TranslationProgress(
                    stage=stage,
                    message=message,
                    progress=progress,
                    data=payload,
                )
            )
