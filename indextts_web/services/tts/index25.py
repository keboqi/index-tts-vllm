from pathlib import Path

from .base import BackendCapabilities, SynthesisRequest
from .legacy import LegacyBackend


class IndexTTS25Backend(LegacyBackend):
    name = "index25"
    capabilities = BackendCapabilities(
        native_streaming=False,
        native_duration=True,
        emotion_text=True,
    )

    async def synthesize(self, request: SynthesisRequest) -> Path:
        legacy = self.legacy
        target_duration = max(0, int(request.target_duration_ms))
        native_duration = 0 if request.duration_control == "ffmpeg" and target_duration else target_duration
        prompt = await legacy._resolve_indextts25_prompt_audio(
            spk_audio_prompt=request.prompt_audio,
            speaker_preset=request.speaker_preset,
        )
        result = await legacy.indextts25_backend_manager.synthesize_to_file(
            text=request.text,
            output_path=str(request.output_path),
            language=request.language,
            prompt_wav=prompt,
            reference_text=request.reference_text,
            speech_length=native_duration,
            interval_silence=request.interval_silence_ms,
            diffusion_steps=request.diffusion_steps,
            max_text_tokens_per_sentence=request.max_text_tokens,
            emotion_audio=request.emotion_audio,
            emotion_text=request.emotion_text,
            emotion_weight=request.emotion_weight,
            cache_prompt_audio=request.cache_prompt_audio,
            seed=request.seed,
            sampling=request.sampling,
        )
        if request.duration_control == "ffmpeg" and target_duration:
            await legacy._postprocess_ffmpeg_duration(result, target_duration)
        return Path(result)
