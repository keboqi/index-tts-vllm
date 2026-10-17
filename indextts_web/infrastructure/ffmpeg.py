"""FFmpeg/ffprobe process helpers and filter-argument escaping."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess


@functools.lru_cache(maxsize=1)
def ffmpeg_available() -> bool:
    """Check once if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def ffmpeg_seconds_from_ms(ms: int) -> str:
    return f"{max(0, int(ms)) / 1000:.3f}"


def escape_filter_path(path: str) -> str:
    normalized = os.path.abspath(path).replace("\\", "/")
    return (
        normalized
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )


def escape_filter_quoted_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def run_ffmpeg_command(cmd: list[str]) -> None:
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        cmd_str = " ".join(cmd)
        raise RuntimeError(
            f"ffmpeg failed (code {process.returncode})\ncmd: {cmd_str}\n"
            f"stderr: {stderr or '<empty>'}\nstdout: {stdout or '<empty>'}"
        )


def ffprobe_duration_seconds(path: str) -> float | None:
    if shutil.which("ffprobe") is None or not os.path.exists(path):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if process.returncode != 0:
            return None
        value = (process.stdout or "").strip()
        return float(value) if value else None
    except Exception:
        return None


def ffprobe_video_dimensions(path: str) -> tuple[int, int] | None:
    if shutil.which("ffprobe") is None or not os.path.exists(path):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        path,
    ]
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if process.returncode != 0:
            return None
        value = (process.stdout or "").strip().splitlines()[0]
        width_text, height_text = value.split("x", 1)
        width = int(width_text)
        height = int(height_text)
        if width <= 0 or height <= 0:
            return None
        return width, height
    except Exception:
        return None


def thread_args(threads: int) -> list[str]:
    return ["-threads", str(threads)]


def video_thread_args(threads: int) -> list[str]:
    return ["-threads:v", str(threads)]


def filter_thread_args(threads: int) -> list[str]:
    thread_count = threads if threads > 0 else max(1, os.cpu_count() or 1)
    return ["-filter_threads", str(thread_count), "-filter_complex_threads", str(thread_count)]


def codec_args_for_format(fmt: str | None, bitrate: str | None, *, threads: int) -> list[str]:
    fmt_normalized = (fmt or "").lower()
    codec_map = {
        "mp3": ["-c:a", "libmp3lame"],
        "ogg": ["-c:a", "libvorbis"],
        "opus": ["-c:a", "libopus"],
        "aac": ["-c:a", "aac"],
        "webm": ["-c:a", "libopus"],
        "wav": ["-c:a", "pcm_s16le"],
        "flac": ["-c:a", "flac"],
    }
    codec_args = codec_map.get(fmt_normalized, ["-c:a", "libmp3lame"])
    if bitrate and fmt_normalized in {"mp3", "ogg", "opus", "aac", "webm"}:
        codec_args = codec_args + ["-b:a", bitrate]
    return codec_args + thread_args(threads)
