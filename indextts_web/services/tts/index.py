from pathlib import Path

from .base import BackendCapabilities, SynthesisRequest
from .legacy import LegacyBackend


class IndexBackend(LegacyBackend):
    name = "index"
    capabilities = BackendCapabilities(
        native_streaming=True,
        native_duration=True,
        emotion_text=True,
    )

    async def synthesize(self, request: SynthesisRequest) -> Path:
        legacy = self.legacy
        target_duration = max(0, int(request.target_duration_ms))
        native_duration = 0 if request.duration_control == "ffmpeg" and target_duration else target_duration
        async with legacy.INDEXTTS_GPU_WORK_SLOTS:
            await legacy.tts_manager.ensure_awake()
            tts = legacy.tts_manager.get_tts()
            result = await tts.infer(
                spk_audio_prompt=request.prompt_audio or "",
                text=request.text,
                output_path=str(request.output_path),
                interval_silence=request.interval_silence_ms,
                speech_length=native_duration,
                diffusion_steps=request.diffusion_steps,
                verbose=request.verbose,
                speaker_preset=request.speaker_preset,
                emo_audio_prompt=request.emotion_audio,
                emo_alpha=request.emotion_weight,
                use_emo_text=bool(request.emotion_text),
                emo_text=request.emotion_text,
                max_text_tokens_per_sentence=request.max_text_tokens,
            )
        if request.duration_control == "ffmpeg" and target_duration:
            await legacy._postprocess_ffmpeg_duration(result, target_duration)
        return Path(result)
