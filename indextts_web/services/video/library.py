"""Downloaded and translated video files: safe paths, listings, and derived caches."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import time
import uuid
from typing import Any
from urllib.parse import quote

from ...infrastructure.ffmpeg import ffprobe_duration_seconds
from ...infrastructure.filenames import sanitize_base_filename
from .ytdlp import VIDEO_DOWNLOAD_EXTENSIONS, format_size_mb, format_video_duration

DOWNLOADED_VIDEO_ID_PREFIX = "v_"


def safe_video_download_filename(filename: str) -> str:
    safe_name = os.path.basename(filename or "")
    if not safe_name or safe_name != filename:
        raise ValueError("Invalid video filename.")
    return safe_name


def encode_downloaded_video_id(filename: str) -> str:
    safe_name = safe_video_download_filename(filename)
    encoded = base64.urlsafe_b64encode(safe_name.encode("utf-8")).decode("ascii")
    return DOWNLOADED_VIDEO_ID_PREFIX + encoded.rstrip("=")


def decode_downloaded_video_id(video_id_or_filename: str) -> str:
    value = (video_id_or_filename or "").strip()
    if value.startswith(DOWNLOADED_VIDEO_ID_PREFIX):
        payload = value[len(DOWNLOADED_VIDEO_ID_PREFIX):]
        padding = "=" * (-len(payload) % 4)
        try:
            return base64.urlsafe_b64decode((payload + padding).encode("ascii")).decode("utf-8")
        except Exception as exc:
            raise ValueError("Invalid downloaded video id.") from exc
    return value


def _is_video_file(name: str) -> bool:
    return os.path.splitext(name)[1].lstrip(".").lower() in VIDEO_DOWNLOAD_EXTENSIONS


def _content_cache_path(cache_dir: str, video_path: str, fallback_stem: str, suffix: str) -> str:
    stat = os.stat(video_path)
    cache_key = hashlib.md5(
        f"{os.path.abspath(video_path)}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    ).hexdigest()[:16]
    stem = sanitize_base_filename(os.path.basename(video_path)) or fallback_stem
    return os.path.join(cache_dir, f"{stem}_{cache_key}{suffix}")


class VideoLibrary:
    def __init__(
        self,
        download_dir: str,
        translated_dir: str,
        audio_cache_dir: str,
        snapshot_cache_dir: str,
    ) -> None:
        self.download_dir = download_dir
        self.translated_dir = translated_dir
        self.audio_cache_dir = audio_cache_dir
        self.snapshot_cache_dir = snapshot_cache_dir

    def downloaded_path(self, video_id_or_filename: str) -> str:
        safe_name = safe_video_download_filename(decode_downloaded_video_id(video_id_or_filename))
        path = os.path.abspath(os.path.join(self.download_dir, safe_name))
        root = os.path.abspath(self.download_dir)
        if os.path.commonpath([root, path]) != root:
            raise ValueError("Invalid video path.")
        return path

    def translated_path(self, filename: str) -> str:
        safe_name = os.path.basename(filename or "")
        if not safe_name or safe_name != filename:
            raise ValueError("Invalid translated video filename.")
        if not _is_video_file(safe_name):
            raise ValueError("Only translated video outputs can be deleted.")
        path = os.path.abspath(os.path.join(self.translated_dir, safe_name))
        root = os.path.abspath(self.translated_dir)
        if os.path.commonpath([root, path]) != root:
            raise ValueError("Invalid translated video path.")
        return path

    def output_path_for(self, raw_info: dict[str, Any], ext: str) -> str:
        """A non-colliding download path named after the video's title and id."""
        title = sanitize_base_filename(str(raw_info.get("title") or raw_info.get("id") or "downloaded_video"))
        video_id = sanitize_base_filename(str(raw_info.get("id") or "")) or uuid.uuid4().hex[:8]
        base = (title or "downloaded_video")[:160].strip()
        safe_ext = re.sub(r"[^A-Za-z0-9]", "", ext or "").lower() or "mp4"
        if safe_ext not in VIDEO_DOWNLOAD_EXTENSIONS:
            safe_ext = "mp4"
        path = os.path.join(self.download_dir, f"{base} [{video_id}].{safe_ext}")
        if not os.path.exists(path):
            return path
        for index in range(2, 1000):
            candidate = os.path.join(self.download_dir, f"{base} [{video_id}] ({index}).{safe_ext}")
            if not os.path.exists(candidate):
                return candidate
        return os.path.join(self.download_dir, f"{base} [{video_id}] {uuid.uuid4().hex[:6]}.{safe_ext}")

    @staticmethod
    def downloaded_entry(path: str) -> dict[str, Any]:
        safe_name = os.path.basename(path)
        video_id = encode_downloaded_video_id(safe_name)
        stat = os.stat(path)
        duration_seconds = ffprobe_duration_seconds(path)
        return {
            "id": video_id,
            "filename": safe_name,
            "title": os.path.splitext(safe_name)[0],
            "extension": os.path.splitext(safe_name)[1].lstrip(".").lower(),
            "size_bytes": int(stat.st_size),
            "size_mb": format_size_mb(stat.st_size),
            "mtime": stat.st_mtime,
            "duration_seconds": duration_seconds,
            "duration_label": format_video_duration(duration_seconds),
            "url": f"/api/downloaded_videos/{video_id}",
            "poster_url": f"/api/downloaded_videos/{video_id}/snapshot",
        }

    @staticmethod
    def translated_entry(path: str) -> dict[str, Any]:
        safe_name = os.path.basename(path)
        stat = os.stat(path)
        duration_seconds = ffprobe_duration_seconds(path)
        encoded_name = quote(safe_name, safe="")
        return {
            "id": safe_name,
            "filename": safe_name,
            "title": os.path.splitext(safe_name)[0],
            "extension": os.path.splitext(safe_name)[1].lstrip(".").lower(),
            "size_bytes": int(stat.st_size),
            "size_mb": format_size_mb(stat.st_size),
            "mtime": stat.st_mtime,
            "mtime_label": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            "duration_seconds": duration_seconds,
            "duration_label": format_video_duration(duration_seconds),
            "url": f"/api/translate_outputs/{encoded_name}",
            "poster_url": f"/api/translate_outputs/{encoded_name}/snapshot",
        }

    def list_downloaded(self) -> list[dict[str, Any]]:
        return self._list_entries(self.download_dir, self.downloaded_entry, "downloaded")

    def list_translated(self) -> list[dict[str, Any]]:
        return self._list_entries(self.translated_dir, self.translated_entry, "translated")

    @staticmethod
    def _list_entries(directory: str, build_entry: Any, label: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not os.path.exists(directory):
            return entries
        for entry in os.scandir(directory):
            if not entry.is_file() or not _is_video_file(entry.name):
                continue
            try:
                entries.append(build_entry(entry.path))
            except Exception as exc:
                print(f"⚠️ Failed to inspect {label} video '{entry.name}': {exc}")
        entries.sort(key=lambda item: item.get("mtime") or 0, reverse=True)
        return entries

    def find_recent_download(self, start_time: float, info: dict[str, Any] | None = None) -> str | None:
        """The newest video written since ``start_time``, matching the yt-dlp id when known."""
        candidates: list[str] = []
        video_id = str((info or {}).get("id") or "").strip()
        for entry in os.scandir(self.download_dir):
            if not entry.is_file() or not _is_video_file(entry.name):
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            if video_id and video_id not in entry.name:
                continue
            if stat.st_mtime >= start_time - 5:
                candidates.append(entry.path)
        if not candidates:
            return None
        candidates.sort(key=lambda item: os.path.getmtime(item), reverse=True)
        return candidates[0]

    def audio_cache_path(self, video_path: str) -> str:
        return _content_cache_path(self.audio_cache_dir, video_path, "downloaded_video", ".mp3")

    def snapshot_cache_path(self, video_path: str) -> str:
        return _content_cache_path(self.snapshot_cache_dir, video_path, "video", ".jpg")
