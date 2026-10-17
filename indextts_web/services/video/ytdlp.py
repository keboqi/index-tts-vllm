"""yt-dlp options, format selection, and direct-download fallbacks."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...infrastructure.ffmpeg import ffmpeg_available


def clean_ytdl_error(exc: Exception) -> str:
    msg = str(exc)
    msg = re.sub(r'\x1b\[[0-9;]*m', '', msg)  # Remove ANSI codes
    msg = msg.replace('ERROR:', '').strip()
    return msg


def is_format_unavailable_error(exc: Exception) -> bool:
    return "requested format is not available" in clean_ytdl_error(exc).lower()


VIDEO_DOWNLOAD_EXTENSIONS = {"mp4", "mkv", "webm", "mov", "m4v"}


VIDEO_DOWNLOAD_FORMATS = {
    "best": "bestvideo*+bestaudio/bestvideo+bestaudio/best",
    "2160p": "bestvideo*[height<=2160]+bestaudio/bestvideo[height<=2160]+bestaudio/best[height<=2160]/best",
    "1440p": "bestvideo*[height<=1440]+bestaudio/bestvideo[height<=1440]+bestaudio/best[height<=1440]/best",
    "1080p": "bestvideo*[height<=1080]+bestaudio/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
    "720p": "bestvideo*[height<=720]+bestaudio/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
    "480p": "bestvideo*[height<=480]+bestaudio/bestvideo[height<=480]+bestaudio/best[height<=480]/best",
    "360p": "bestvideo*[height<=360]+bestaudio/bestvideo[height<=360]+bestaudio/best[height<=360]/best",
    "worst": "worstvideo*+worstaudio/worst",
}


def dedupe_preserving_order(values: list[str | None]) -> list[str | None]:
    seen: set[str | None] = set()
    result: list[str | None] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def download_format_fallbacks(format_selector: str, quality_key: str) -> list[str | None]:
    """Return format selectors from most-specific to safest."""
    candidates: list[str | None] = [format_selector]
    if quality_key in VIDEO_DOWNLOAD_FORMATS:
        height_match = re.fullmatch(r"(\d+)p", quality_key)
        if height_match:
            height = height_match.group(1)
            candidates.extend(
                [
                    f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/best",
                    f"bestvideo*[height<={height}]/best[height<={height}]/best",
                ]
            )
        elif quality_key == "best":
            candidates.extend(["bestvideo+bestaudio/best", "bestvideo*/best"])
        elif quality_key == "worst":
            candidates.extend(["worstvideo+worstaudio/worst", "worstvideo*/worst"])
    else:
        candidates.append("bestvideo+bestaudio/best")
    candidates.extend(["best", None])
    return dedupe_preserving_order(candidates)


def setup_hint() -> str:
    return (
        "YouTube currently requires yt-dlp's external JavaScript challenge solver setup. "
        "Upgrade with `pip install -U \"yt-dlp[default]\"`, install a supported JS runtime "
        "such as Deno or Node, enable that runtime for the yt-dlp Python API, and use a "
        "running PO Token provider for affected YouTube videos."
    )


def auth_hint() -> str:
    return (
        "YouTube is asking this server to sign in. Import fresh youtube.com cookies exported from "
        "a private/incognito YouTube session, then retry. For CLI verification, pass the same file "
        "with `--cookies /path/to/youtube_com_cookies.txt`."
    )


def common_opts() -> dict[str, Any]:
    opts: dict[str, Any] = {}
    js_runtimes: dict[str, dict[str, str]] = {}
    configured_runtimes = (os.environ.get("YTDLP_JS_RUNTIMES") or "").strip()
    configured_node = (os.environ.get("YTDLP_NODE_PATH") or os.environ.get("NODE_BINARY") or "").strip()

    def find_executable(name: str) -> str | None:
        found = shutil.which(name)
        if found:
            return found
        # Uvicorn services often have a thinner PATH than an interactive shell.
        # Ask a login shell as a last resort so it can see nvm/homebrew setup.
        try:
            process = subprocess.run(
                ["bash", "-lc", f"command -v {name}"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            candidate = (process.stdout or "").strip().splitlines()[0] if process.stdout.strip() else ""
            if candidate and os.path.exists(candidate):
                return candidate
        except Exception:
            pass
        if name == "node":
            try:
                process = subprocess.run(
                    ["bash", "-lc", "node -p 'process.execPath'"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                candidate = (process.stdout or "").strip().splitlines()[0] if process.stdout.strip() else ""
                if candidate and os.path.exists(candidate):
                    return candidate
            except Exception:
                pass
        if name == "node":
            node_candidates: list[str] = []
            home = Path.home()
            node_candidates.extend(str(path) for path in (home / ".nvm" / "versions" / "node").glob("*/bin/node"))
            node_candidates.extend(str(path) for path in Path("/home").glob("*/.nvm/versions/node/*/bin/node"))
            node_candidates.extend(
                [
                    "/usr/local/bin/node",
                    "/usr/bin/node",
                    "/snap/bin/node",
                    "/opt/node/bin/node",
                    "/home/linuxbrew/.linuxbrew/bin/node",
                ]
            )
            existing = [path for path in node_candidates if os.path.exists(path)]
            if existing:
                existing.sort(reverse=True)
                return existing[0]
        return None

    runtime_paths = {
        "node": configured_node if configured_node and os.path.exists(configured_node) else find_executable("node"),
        "deno": find_executable("deno"),
        "bun": find_executable("bun"),
        "quickjs": find_executable("qjs") or find_executable("quickjs"),
    }
    if configured_runtimes:
        runtime_items: list[tuple[str, str | None]] = []
        for item in configured_runtimes.split(","):
            runtime_spec = item.strip()
            if not runtime_spec:
                continue
            name, _, explicit_path = runtime_spec.partition(":")
            runtime_items.append((name.strip(), explicit_path.strip() or runtime_paths.get(name.strip())))
    elif runtime_paths.get("node"):
        # Mirror the known-good CLI path: --js-runtimes node.
        runtime_items = [("node", runtime_paths["node"])]
    else:
        runtime_items = [(name, path) for name, path in runtime_paths.items()]

    for runtime, executable in runtime_items:
        if runtime and executable:
            js_runtimes[runtime] = {"path": executable}
    if js_runtimes:
        opts["js_runtimes"] = js_runtimes

    remote_components = [
        item.strip()
        for item in (os.environ.get("YTDLP_REMOTE_COMPONENTS") or "").split(",")
        if item.strip()
    ]
    if remote_components:
        opts["remote_components"] = set(remote_components)
    return opts


def raw_format_is_direct(fmt: dict[str, Any]) -> bool:
    protocol = str(fmt.get("protocol") or "").lower()
    ext = str(fmt.get("ext") or "").lower()
    if not fmt.get("url") or fmt.get("has_drm"):
        return False
    if protocol in {"mhtml", "storyboard"} or ext in {"mhtml", "json"}:
        return False
    return True


def raw_format_score(fmt: dict[str, Any]) -> tuple[int, float, float, int]:
    height = int(fmt.get("height") or 0)
    tbr = float(fmt.get("tbr") or 0)
    fps = float(fmt.get("fps") or 0)
    size = int(fmt.get("filesize") or fmt.get("filesize_approx") or 0)
    return (height, tbr, fps, size)


def extract_format_ids_from_selector(selector: str) -> list[str]:
    tokens = re.split(r"[+/]", selector or "")
    ids: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token or token.startswith("best") or token.startswith("worst"):
            continue
        token = re.sub(r"\[.*$", "", token).strip()
        if token:
            ids.append(token)
    return ids


def select_raw_video_formats(
    raw_info: dict[str, Any],
    quality_key: str,
    format_selector: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw_formats = [fmt for fmt in (raw_info.get("formats") or []) if isinstance(fmt, dict) and raw_format_is_direct(fmt)]
    by_id = {str(fmt.get("format_id")): fmt for fmt in raw_formats if fmt.get("format_id") is not None}
    audio_formats = [
        fmt for fmt in raw_formats
        if (fmt.get("acodec") or "none") != "none" and (fmt.get("vcodec") or "none") == "none"
    ]
    audio_formats.sort(key=raw_format_score, reverse=True)

    for format_id in extract_format_ids_from_selector(format_selector):
        fmt = by_id.get(format_id)
        if not fmt or (fmt.get("vcodec") or "none") == "none":
            continue
        if (fmt.get("acodec") or "none") != "none":
            return fmt, None
        return fmt, audio_formats[0] if audio_formats else None

    height_limit: int | None = None
    height_match = re.fullmatch(r"(\d+)p", quality_key)
    if height_match:
        height_limit = int(height_match.group(1))

    video_formats = [fmt for fmt in raw_formats if (fmt.get("vcodec") or "none") != "none"]
    if height_limit:
        limited = [fmt for fmt in video_formats if int(fmt.get("height") or 0) <= height_limit]
        if limited:
            video_formats = limited
    reverse = quality_key != "worst"
    combined_formats = [fmt for fmt in video_formats if (fmt.get("acodec") or "none") != "none"]
    video_only_formats = [fmt for fmt in video_formats if (fmt.get("acodec") or "none") == "none"]
    combined_formats.sort(key=raw_format_score, reverse=reverse)
    video_only_formats.sort(key=raw_format_score, reverse=reverse)

    if video_only_formats and audio_formats and ffmpeg_available():
        return video_only_formats[0], audio_formats[0]
    if combined_formats:
        return combined_formats[0], None
    if video_only_formats:
        return video_only_formats[0], None
    return None, None


def download_direct_format(
    fmt: dict[str, Any],
    output_path: str,
    emit: Callable[..., None],
    message: str,
) -> None:
    headers = dict(fmt.get("http_headers") or {})
    headers.setdefault("User-Agent", "Mozilla/5.0")
    request = urllib.request.Request(str(fmt["url"]), headers=headers)
    downloaded = 0
    with urllib.request.urlopen(request, timeout=60) as response, open(output_path, "wb") as fh:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            percent = round((downloaded / total) * 100, 2) if total else None
            emit(
                "progress",
                message=message,
                percent=percent,
                downloaded_bytes=downloaded,
                total_bytes=total,
                speed=None,
                eta=None,
            )


def format_video_duration(seconds: float | None) -> str:
    if not seconds or seconds <= 0:
        return ""
    whole = int(round(seconds))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_size_mb(size_bytes: int) -> float:
    return round(max(0, int(size_bytes or 0)) / (1024 * 1024), 2)
