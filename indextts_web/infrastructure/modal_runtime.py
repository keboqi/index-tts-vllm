"""Container-local application code with persistent model/user data."""

from __future__ import annotations

import shutil
from pathlib import Path

PERSISTENT_DIRECTORIES = (
    "checkpoints", "outputs", "speaker_presets", "emotion_cache",
    "Confucius4-TTS", "index-tts-2.5-vllm-omni-experiment",
)


def prepare_runtime_code(source: Path, persistent: Path, destination: Path) -> Path:
    """Use the source shipped with the deployment, never stale volume code.

    Only destination (a fresh container-local directory) is modified. The
    persistent application checkout, weights, and user files are not replaced.
    """
    if destination.exists():
        raise FileExistsError(f"Runtime code directory already exists: {destination}")
    shutil.copytree(source, destination)
    for name in PERSISTENT_DIRECTORIES:
        target = persistent / name
        if name in {"outputs", "speaker_presets", "emotion_cache"}:
            target.mkdir(parents=True, exist_ok=True)
        if not target.is_dir():
            raise FileNotFoundError(f"Persistent runtime data missing: {target}; run prepare_model first")
        link = destination / name
        if link.exists():
            raise ValueError(f"Deployment source must not contain runtime data: {link}")
        link.symlink_to(target, target_is_directory=True)
    # Preserve user-provided assets if present; bundled example assets remain
    # available on a fresh volume. Both locations contain data, not Python code.
    if (persistent / "assets").is_dir():
        if (destination / "assets").is_dir():
            shutil.rmtree(destination / "assets")
        (destination / "assets").symlink_to(persistent / "assets", target_is_directory=True)
    return destination
