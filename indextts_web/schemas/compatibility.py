from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True, slots=True)
class SchemaInventory:
    tts: dict[str, type[Any]]
    translation: dict[str, type[Any]]
    video: dict[str, type[Any]]
    speakers: dict[str, type[Any]]
    utilities: dict[str, type[Any]]


def _models(module: ModuleType, names: tuple[str, ...]) -> dict[str, type[Any]]:
    return {name: getattr(module, name) for name in names}


def from_legacy(module: ModuleType) -> SchemaInventory:
    return SchemaInventory(
        tts=_models(module, ("SpeakRequest", "CloneRequest")),
        translation=_models(
            module,
            (
                "TranslateRequest",
                "SpeakerOverrideInput",
                "TranslateSegmentInput",
                "SegmentPreviewRequest",
                "TranslateGenerateRequest",
                "MergeChunksRequest",
                "ChunkBatchGenerateRequest",
            ),
        ),
        video=_models(module, ("VideoInfoRequest", "VideoDownloadRequest", "VideoReplaceAudioRequest")),
        speakers=_models(module, ("VoiceDesignRequest", "SaveDesignedVoiceRequest")),
        utilities=_models(module, ("CookieImportCurlRequest",)),
    )
