from pathlib import Path

from .base import BackendCapabilities, SynthesisRequest
from .legacy import LegacyBackend


class HiggsBackend(LegacyBackend):
    name = "higgs"
    capabilities = BackendCapabilities(
        native_streaming=True,
        native_duration=False,
        emotion_text=False,
    )

    async def synthesize(self, request: SynthesisRequest) -> Path:
        legacy = self.legacy
        if request.emotion_text or request.emotion_audio:
            print("[Higgs SGLang] Ignoring IndexTTS emotion controls for Higgs backend.")
        prompt = await legacy._resolve_external_prompt_audio(
            spk_audio_prompt=request.prompt_audio,
            speaker_preset=request.speaker_preset,
            backend_label="Higgs SGLang",
        )
        sampling = dict(request.sampling)
        result = await legacy.higgs_backend_manager.synthesize_to_file(
            text=request.text,
            output_path=str(request.output_path),
            prompt_wav=prompt,
            reference_text=request.reference_text,
            speech_length=max(0, int(request.target_duration_ms)),
            max_text_tokens_per_sentence=request.max_text_tokens,
            temperature=sampling.get("temperature"),
            top_k=sampling.get("top_k"),
            top_p=sampling.get("top_p"),
            max_new_tokens=sampling.get("max_new_tokens"),
            seed=request.seed,
        )
        return Path(result)
