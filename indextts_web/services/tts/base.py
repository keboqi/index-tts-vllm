"""Backend-neutral TTS contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    native_streaming: bool
    native_duration: bool
    emotion_text: bool
    reference_audio: bool = True


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    text: str
    output_path: Path
    speaker_preset: str | None = None
    prompt_audio: str | None = None
    reference_text: str | None = None
    language: str | None = None
    interval_silence_ms: int = 0
    target_duration_ms: int = 0
    duration_control: str = "original"
    max_text_tokens: int = 120
    diffusion_steps: int = 10
    verbose: bool = False
    emotion_audio: str | None = None
    emotion_text: str | None = None
    emotion_weight: float = 0.6
    cache_prompt_audio: bool = True
    seed: int | None = None
    sampling: Mapping[str, Any] = field(default_factory=dict)


class TTSBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    async def synthesize(self, request: SynthesisRequest) -> Path: ...

    async def stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]: ...

    async def status(self) -> Mapping[str, Any]: ...

    async def shutdown(self) -> None: ...
