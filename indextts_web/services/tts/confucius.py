from pathlib import Path

from .base import BackendCapabilities, SynthesisRequest
from .legacy import LegacyBackend


class ConfuciusBackend(LegacyBackend):
    name = "confucius"
    capabilities = BackendCapabilities(
        native_streaming=True,
        native_duration=True,
        emotion_text=False,
    )

    async def synthesize(self, request: SynthesisRequest) -> Path:
        legacy = self.legacy
        omni_manager = getattr(legacy, "indextts25_backend_manager", None)
        coordinated = getattr(getattr(legacy.confucius_backend_manager, "gpu_coordinator", None), "enabled", False)
        if not coordinated and omni_manager is not None and omni_manager.process_running():
            await omni_manager.stop("switching to Confucius4-TTS")
        if not coordinated and getattr(legacy.confucius_backend_manager, "_vllm_sleeping", False):
            await legacy.confucius_backend_manager.wake_vllm()
        if request.emotion_text or request.emotion_audio:
            print("[Confucius4-TTS] Ignoring IndexTTS emotion controls for Confucius backend.")
        target_duration = max(0, int(request.target_duration_ms))
        native_duration = 0 if request.duration_control == "ffmpeg" and target_duration else target_duration
        prompt = await legacy._resolve_confucius_prompt_audio(
            spk_audio_prompt=request.prompt_audio,
            speaker_preset=request.speaker_preset,
        )
        result = await legacy.confucius_backend_manager.synthesize_to_file(
            text=request.text,
            output_path=str(request.output_path),
            language=request.language,
            prompt_wav=prompt,
            speech_length=native_duration,
            diffusion_steps=request.diffusion_steps,
            max_text_tokens_per_sentence=request.max_text_tokens,
            verbose=request.verbose,
            cache_prompt_audio=request.cache_prompt_audio,
        )
        if request.duration_control == "ffmpeg" and target_duration:
            await legacy._postprocess_ffmpeg_duration(result, target_duration)
        return Path(result)
