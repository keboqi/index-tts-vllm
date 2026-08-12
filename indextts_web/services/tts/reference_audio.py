"""Backend-neutral reference-audio normalization."""

from __future__ import annotations

import hashlib
import io
import os
import uuid
import wave
from pathlib import Path

MIN_REFERENCE_AUDIO_MS = 1000


def ensure_minimum_reference_duration(
    value: str | None,
    *,
    cache_dir: Path,
    minimum_ms: int = MIN_REFERENCE_AUDIO_MS,
) -> str | None:
    """Return a cached WAV padded with trailing silence when it is too short."""
    if not value or value.startswith(("http://", "https://", "data:", "file://")):
        return value
    source_path = Path(value).expanduser().resolve()
    if not source_path.is_file():
        return value
    raw = source_path.read_bytes()
    try:
        with wave.open(io.BytesIO(raw), "rb") as source:
            params = source.getparams()
            frames = source.readframes(source.getnframes())
    except (EOFError, wave.Error):
        return value

    minimum_frames = round(params.framerate * max(0, int(minimum_ms)) / 1000)
    if params.nframes >= minimum_frames:
        return value

    stat = source_path.stat()
    cache_key = hashlib.sha256(
        f"{source_path}\0{stat.st_mtime_ns}\0{stat.st_size}\0{minimum_frames}".encode()
    ).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{source_path.stem}_{cache_key}_min{minimum_ms}ms.wav"
    if output_path.is_file():
        return str(output_path)

    frame_width = params.nchannels * params.sampwidth
    padded = frames.ljust(minimum_frames * frame_width, b"\x00")
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams(params)
        target.writeframes(padded)

    temporary_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_bytes(output.getvalue())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(output_path)
