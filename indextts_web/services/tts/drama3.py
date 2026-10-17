from pathlib import Path

from .base import BackendCapabilities, SynthesisRequest
from .legacy import LegacyBackend


class Drama3Backend(LegacyBackend):
    name = "drama3"
    manager_attribute = "drama3_backend_manager"
    capabilities = BackendCapabilities(native_streaming=False, native_duration=False, emotion_text=True)

    async def synthesize(self, request: SynthesisRequest) -> Path:
        prompt = await self.legacy._resolve_external_prompt_audio(
            spk_audio_prompt=request.prompt_audio, speaker_preset=request.speaker_preset,
            backend_label="Drama 3 SGLang-Omni",
        )
        result = await self.manager.synthesize_to_file(
            text=request.text, output_path=str(request.output_path), prompt_wav=prompt,
            reference_text=request.reference_text, emotion_text=request.emotion_text,
            sampling=request.sampling, max_text_tokens=request.max_text_tokens,
            interval_silence_ms=request.interval_silence_ms,
        )
        if request.target_duration_ms > 0:
            await self.legacy._postprocess_ffmpeg_duration(result, request.target_duration_ms)
        return Path(result)
