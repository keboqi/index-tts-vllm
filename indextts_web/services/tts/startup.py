"""Startup and snapshot hooks for the selected TTS backend."""

from pathlib import Path
from uuid import uuid4

from .base import SynthesisRequest


def startup_backend(settings) -> str:
    return settings.startup_tts_backend or "none"


def _use_reference_presets(legacy) -> None:
    # Reference presets remain available without loading IndexTTS embeddings.
    legacy.tts_manager.speaker_manager = legacy.SpeakerPresetManager()
    legacy.speaker_api = legacy.SpeakerAPIWrapper(legacy.tts_manager.speaker_manager)


async def initialize_backend(legacy) -> None:
    backend = startup_backend(legacy.SETTINGS)
    if backend == "none":
        _use_reference_presets(legacy)
        return
    if backend == "index":
        legacy.GPU_PROFILE.check_startup_memory(
            non_vllm_gib=float(legacy.os.getenv("INDEXTTS_NON_VLLM_RESERVE_GIB", "8")),
        )
        if not await legacy.tts_manager.initialize():
            raise RuntimeError("IndexTTS initialization failed; see the model error and resolved GPU profile above")
        if legacy.SETTINGS.use_torch_compile:
            await legacy.warmup_model()
        return
    _use_reference_presets(legacy)
    manager = legacy.TTS_BACKEND_REGISTRY.get(backend).manager
    await manager.ensure_ready()


async def warmup_backend(legacy) -> None:
    backend = startup_backend(legacy.SETTINGS)
    if backend == "none":
        return
    if backend == "index":
        await legacy.warmup_model(strict=True)
        legacy.tts_manager.refresh_post_snapshot_state()
        return
    reference = Path(legacy.current_dir) / "examples" / "voice_01.wav"
    if not reference.is_file():
        raise FileNotFoundError(f"Warmup audio missing: {reference}")
    output = Path(legacy.ROOT_OUTPUT_DIR) / f"snapshot_{backend}_{uuid4().hex}.wav"
    try:
        await legacy.TTS_BACKEND_REGISTRY.get(backend).synthesize(SynthesisRequest(
            text="Hello, this voice is ready to speak.", prompt_audio=str(reference), output_path=output,
        ))
        if not output.is_file() or output.stat().st_size <= 44:
            raise RuntimeError(f"{backend} snapshot warmup returned no audio")
    finally:
        output.unlink(missing_ok=True)


async def restore_backend(legacy) -> None:
    backend = startup_backend(legacy.SETTINGS)
    if backend == "none":
        return
    if backend == "index":
        await legacy.tts_manager.wake_from_snapshot()
    else:
        manager = legacy.TTS_BACKEND_REGISTRY.get(backend).manager
        if backend == "drama3":
            # Modal restores the running GPU pipeline directly. The separate
            # NVIDIA driver job wrapper is reserved for later ordinary launches.
            manager._server.checkpoint_launch_enabled = True
            if not manager.process_running() or not (await manager.status())["ready"]:
                raise RuntimeError("Drama 3 snapshot did not restore a ready pipeline; rebuild the snapshot")
        else:
            await manager.ensure_ready()
        legacy.tts_manager.refresh_post_snapshot_state()
