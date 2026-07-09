"""
MOSS-Transcribe-Diarize pipeline backed by SGLang-Omni.

This module talks to an OpenAI-compatible SGLang-Omni server exposing
``/v1/audio/transcriptions`` and adapts MOSS diarized output into the same
``(segments, speaker_profiles, raw_text, cache_info)`` tuple used by the other
local transcription pipelines.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from whisperx_pipeline import (
        _can_attempt_translation_backend,
        _make_translation_llm,
        _translate_texts_adaptively,
    )
except Exception:
    _can_attempt_translation_backend = None  # type: ignore[assignment]
    _make_translation_llm = None  # type: ignore[assignment]
    _translate_texts_adaptively = None  # type: ignore[assignment]


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(
    name: str,
    default: int,
    *,
    min_value: int = 1,
    max_value: Optional[int] = None,
) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float = 0.0,
    max_value: Optional[float] = None,
) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


MOSS_TRANSCRIBE_MODEL = (
    os.getenv("MOSS_TRANSCRIBE_MODEL", "OpenMOSS-Team/MOSS-Transcribe-Diarize").strip()
    or "OpenMOSS-Team/MOSS-Transcribe-Diarize"
)
MOSS_TRANSCRIBE_SGLANG_URL = (
    os.getenv("MOSS_TRANSCRIBE_SGLANG_URL", "http://127.0.0.1:8003").strip()
    or "http://127.0.0.1:8003"
).rstrip("/")
MOSS_TRANSCRIBE_MANAGE_BACKEND = _env_bool("MOSS_TRANSCRIBE_MANAGE_BACKEND", True)
MOSS_TRANSCRIBE_MANAGER_SCRIPT = (
    os.getenv("MOSS_TRANSCRIBE_MANAGER_SCRIPT", "sglang_omni_moss_transcribe.sh").strip()
    or "sglang_omni_moss_transcribe.sh"
)
MOSS_TRANSCRIBE_START_TIMEOUT = _env_float(
    "MOSS_TRANSCRIBE_START_TIMEOUT",
    3600.0,
    min_value=1.0,
)
MOSS_TRANSCRIBE_REQUEST_TIMEOUT = _env_float(
    "MOSS_TRANSCRIBE_REQUEST_TIMEOUT",
    1800.0,
    min_value=1.0,
)
MOSS_TRANSCRIBE_RESPONSE_FORMAT = (
    os.getenv("MOSS_TRANSCRIBE_RESPONSE_FORMAT", "verbose_json").strip()
    or "verbose_json"
)
MOSS_TRANSCRIBE_TEMPERATURE = (
    os.getenv("MOSS_TRANSCRIBE_TEMPERATURE", "0").strip()
    or "0"
)
MOSS_TRANSCRIBE_MAX_NEW_TOKENS = os.getenv(
    "MOSS_TRANSCRIBE_MAX_NEW_TOKENS",
    "65536",
).strip()
MOSS_TRANSCRIBE_PROMPT = os.getenv("MOSS_TRANSCRIBE_PROMPT", "").strip()
MOSS_TRANSCRIBE_LANGUAGE = os.getenv("MOSS_TRANSCRIBE_LANGUAGE", "").strip()
MOSS_TRANSCRIBE_API_KEY = os.getenv("MOSS_TRANSCRIBE_API_KEY", "").strip()
MOSS_TRANSCRIBE_CACHE_DIR = os.getenv(
    "MOSS_TRANSCRIBE_CACHE_DIR",
    os.path.join(_SCRIPT_DIR, "moss_transcribe_cache"),
)
MOSS_TRANSCRIBE_CACHE_VERSION = 1
MOSS_TRANSCRIBE_TRANSLATION_LLM = os.getenv(
    "MOSS_TRANSCRIBE_TRANSLATION_LLM",
    os.getenv("WHISPERX_TRANSLATION_LLM", "tencent/Hy-MT2-1.8B"),
)
MOSS_TRANSCRIBE_TRANSLATION_BATCH_SIZE = _env_int(
    "MOSS_TRANSCRIBE_TRANSLATION_BATCH_SIZE",
    30,
)
MOSS_TRANSCRIBE_TRANSLATION_MAX_WORKERS = _env_int(
    "MOSS_TRANSCRIBE_TRANSLATION_MAX_WORKERS",
    10,
    max_value=10,
)


_BACKEND_START_LOCK = threading.Lock()
_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*\[(?P<speaker>S\d+|speaker\s*\d+|spk[_\s-]*\d+)\]\s*",
    re.IGNORECASE,
)
_COMPACT_START_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\]",
    re.IGNORECASE,
)
_TRAILING_TIMESTAMP_RE = re.compile(r"\[(?P<end>\d+(?:\.\d+)?)\]\s*$")
_BRACKET_TIMESTAMP_RE = re.compile(r"\[\d+(?:\.\d+)?\]")


def _server_url(path: str) -> str:
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{MOSS_TRANSCRIBE_SGLANG_URL}{suffix}"


def _auth_headers() -> Dict[str, str]:
    if not MOSS_TRANSCRIBE_API_KEY:
        return {}
    return {"Authorization": f"Bearer {MOSS_TRANSCRIBE_API_KEY}"}


def _check_health(timeout: float = 1.0) -> bool:
    request = urllib.request.Request(
        _server_url("/v1/models"),
        headers=_auth_headers(),
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(getattr(response, "status", 200)) < 300
    except Exception:
        return False


def _resolve_manager_script() -> Optional[str]:
    candidate = os.path.expanduser(MOSS_TRANSCRIBE_MANAGER_SCRIPT)
    if not os.path.isabs(candidate):
        candidate = os.path.join(_SCRIPT_DIR, candidate)
    candidate = os.path.abspath(candidate)
    return candidate if os.path.isfile(candidate) else None


def _resolve_bash() -> Optional[str]:
    requested = os.getenv("MOSS_TRANSCRIBE_BASH", "bash").strip() or "bash"
    if os.path.isabs(requested) and os.path.isfile(requested):
        return requested
    return shutil.which(requested)


def is_moss_transcribe_available() -> bool:
    if _check_health(timeout=0.5):
        return True
    if not MOSS_TRANSCRIBE_MANAGE_BACKEND:
        return False
    return _resolve_manager_script() is not None and _resolve_bash() is not None


def _managed_backend_env() -> Dict[str, str]:
    env = os.environ.copy()
    parsed_url = urllib.parse.urlparse(MOSS_TRANSCRIBE_SGLANG_URL)
    port = parsed_url.port or 8003
    prefixed_to_script = {
        "MODEL": MOSS_TRANSCRIBE_MODEL,
        "PORT": str(port),
        "MEM_FRACTION_STATIC": os.getenv("MOSS_TRANSCRIBE_MEM_FRACTION_STATIC", "0.20"),
        "MAX_RUNNING_REQUESTS": os.getenv("MOSS_TRANSCRIBE_MAX_RUNNING_REQUESTS", "4"),
        "CUDA_GRAPH_MAX_BS": os.getenv("MOSS_TRANSCRIBE_CUDA_GRAPH_MAX_BS", "4"),
    }
    for key, value in prefixed_to_script.items():
        env[key] = str(value)

    optional_mappings = {
        "MOSS_TRANSCRIBE_NAME": "NAME",
        "MOSS_TRANSCRIBE_IMAGE": "IMAGE",
        "MOSS_TRANSCRIBE_PROJECT_ROOT": "PROJECT_ROOT",
        "MOSS_TRANSCRIBE_HF_CACHE": "HF_CACHE",
        "MOSS_TRANSCRIBE_EXTRA_ARGS": "EXTRA_ARGS",
    }
    for source, target in optional_mappings.items():
        value = os.getenv(source)
        if value is not None and value.strip():
            env[target] = value.strip()
    return env


def _start_managed_backend() -> None:
    script_path = _resolve_manager_script()
    if script_path is None:
        raise RuntimeError(
            "MOSS transcription backend is not healthy and the manager script "
            f"was not found: {MOSS_TRANSCRIBE_MANAGER_SCRIPT}"
        )
    bash_path = _resolve_bash()
    if bash_path is None:
        raise RuntimeError(
            "MOSS transcription backend is not healthy and bash was not found. "
            "Install Git Bash/WSL bash or set MOSS_TRANSCRIBE_BASH."
        )

    print(f"[MOSS] Starting managed SGLang-Omni backend with {script_path}")
    proc = subprocess.run(
        [bash_path, script_path, "start"],
        cwd=_SCRIPT_DIR,
        env=_managed_backend_env(),
        capture_output=True,
        text=True,
        timeout=MOSS_TRANSCRIBE_START_TIMEOUT,
        check=False,
    )
    if proc.returncode != 0:
        output = "\n".join(
            part for part in [proc.stdout.strip(), proc.stderr.strip()] if part
        )
        if len(output) > 4000:
            output = output[-4000:]
        raise RuntimeError(
            "Managed MOSS transcription backend failed to start. "
            f"Exit code {proc.returncode}.\n{output}"
        )


def _ensure_backend_ready() -> None:
    if _check_health(timeout=2.0):
        return
    if not MOSS_TRANSCRIBE_MANAGE_BACKEND:
        raise RuntimeError(
            "MOSS transcription SGLang server is not responding at "
            f"{MOSS_TRANSCRIBE_SGLANG_URL}. Start it with "
            "`bash sglang_omni_moss_transcribe.sh start` or set "
            "MOSS_TRANSCRIBE_MANAGE_BACKEND=1."
        )

    with _BACKEND_START_LOCK:
        if _check_health(timeout=2.0):
            return
        _start_managed_backend()
        deadline = time.monotonic() + MOSS_TRANSCRIBE_START_TIMEOUT
        while time.monotonic() < deadline:
            if _check_health(timeout=5.0):
                print("[MOSS] SGLang-Omni backend is healthy.")
                return
            time.sleep(2.0)

    raise RuntimeError(
        "Timed out waiting for the MOSS transcription SGLang backend to become "
        f"healthy at {MOSS_TRANSCRIBE_SGLANG_URL}."
    )


def _cache_key(
    audio_hash: str,
    *,
    dest_language: str,
    enable_translation: bool,
    translation_llm_model: Optional[str],
    input_mime_type: Optional[str],
) -> str:
    raw = "|".join(
        [
            f"v{MOSS_TRANSCRIBE_CACHE_VERSION}",
            audio_hash,
            f"url={MOSS_TRANSCRIBE_SGLANG_URL}",
            f"model={MOSS_TRANSCRIBE_MODEL}",
            f"mime={input_mime_type or ''}",
            f"dst={dest_language}",
            f"translate={enable_translation}",
            f"llm={translation_llm_model or 'default'}",
            f"format={MOSS_TRANSCRIBE_RESPONSE_FORMAT}",
            f"temperature={MOSS_TRANSCRIBE_TEMPERATURE}",
            f"max_new_tokens={MOSS_TRANSCRIBE_MAX_NEW_TOKENS}",
            f"language={MOSS_TRANSCRIBE_LANGUAGE}",
            f"prompt={MOSS_TRANSCRIBE_PROMPT}",
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_path(cache_key: str) -> str:
    os.makedirs(MOSS_TRANSCRIBE_CACHE_DIR, exist_ok=True)
    return os.path.join(MOSS_TRANSCRIBE_CACHE_DIR, f"moss_transcribe_{cache_key}.json")


def _load_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(cache_key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _write_cache(cache_key: str, record: Dict[str, Any]) -> Optional[str]:
    path = _cache_path(cache_key)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        return path
    except Exception as exc:
        print(f"Warning: MOSS transcription cache write failed: {exc}")
        return None


def _mime_extension(mime_type: Optional[str]) -> str:
    if not mime_type:
        return "wav"
    clean_mime = mime_type.split(";", 1)[0].strip().lower()
    mapping = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/flac": "flac",
        "audio/ogg": "ogg",
        "audio/webm": "webm",
        "audio/mp4": "mp4",
        "audio/m4a": "m4a",
        "audio/x-m4a": "m4a",
    }
    if clean_mime in mapping:
        return mapping[clean_mime]
    guessed = mimetypes.guess_extension(clean_mime)
    return guessed.lstrip(".") if guessed else "wav"


def _multipart_body(
    *,
    fields: Dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    file_mime_type: str,
) -> Tuple[bytes, str]:
    boundary = f"----index-tts-moss-{uuid.uuid4().hex}"
    chunks: List[bytes] = []
    for key, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
        )
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {file_mime_type}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_bytes)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def _transcribe_with_sglang(
    audio_bytes: bytes,
    *,
    input_mime_type: Optional[str],
) -> Dict[str, Any]:
    fields: Dict[str, str] = {
        "model": MOSS_TRANSCRIBE_MODEL,
        "response_format": MOSS_TRANSCRIBE_RESPONSE_FORMAT,
        "temperature": MOSS_TRANSCRIBE_TEMPERATURE,
    }
    if MOSS_TRANSCRIBE_MAX_NEW_TOKENS:
        fields["max_new_tokens"] = MOSS_TRANSCRIBE_MAX_NEW_TOKENS
    if MOSS_TRANSCRIBE_PROMPT:
        fields["prompt"] = MOSS_TRANSCRIBE_PROMPT
    if MOSS_TRANSCRIBE_LANGUAGE:
        fields["language"] = MOSS_TRANSCRIBE_LANGUAGE

    clean_mime = (input_mime_type or "audio/wav").split(";", 1)[0].strip() or "audio/wav"
    filename = f"audio.{_mime_extension(clean_mime)}"
    body, boundary = _multipart_body(
        fields=fields,
        file_field="file",
        filename=filename,
        file_bytes=audio_bytes,
        file_mime_type=clean_mime,
    )
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
        **_auth_headers(),
    }
    request = urllib.request.Request(
        _server_url("/v1/audio/transcriptions"),
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=MOSS_TRANSCRIBE_REQUEST_TIMEOUT,
        ) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"MOSS transcription request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"MOSS transcription request failed: {exc}") from exc

    response_text = response_bytes.decode("utf-8", errors="replace")
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        payload = {"text": response_text}
    if not isinstance(payload, dict):
        payload = {"response": payload}
    return payload


def _as_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if math.isfinite(seconds):
            return max(0.0, seconds)
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", ".")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return max(0.0, float(text))
    parts = [part.strip() for part in text.split(":") if part.strip()]
    if not parts:
        return None
    try:
        total = 0.0
        multiplier = 1.0
        for part in reversed(parts):
            total += float(part) * multiplier
            multiplier *= 60.0
        return max(0.0, total)
    except ValueError:
        return None


def _timestamp_pair(entry: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    start = _as_seconds(
        entry.get("start")
        if "start" in entry
        else entry.get("start_time", entry.get("begin", entry.get("from")))
    )
    end = _as_seconds(
        entry.get("end")
        if "end" in entry
        else entry.get("end_time", entry.get("stop", entry.get("to")))
    )
    timestamp = entry.get("timestamp") or entry.get("timestamps")
    if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        if start is None:
            start = _as_seconds(timestamp[0])
        if end is None:
            end = _as_seconds(timestamp[1])
    return start, end


def _speaker_key(raw_speaker: Any) -> str:
    text = str(raw_speaker or "").strip()
    if not text:
        return "S01"
    text = text.strip("[]").strip()
    digits = re.findall(r"\d+", text)
    if digits:
        return f"S{int(digits[0]):02d}"
    return text.upper()


def _speaker_mapper() -> Tuple[Dict[str, str], Any]:
    mapping: Dict[str, str] = {}

    def resolve(raw_speaker: Any) -> str:
        key = _speaker_key(raw_speaker)
        if key not in mapping:
            mapping[key] = f"speaker{len(mapping) + 1}"
        return mapping[key]

    return mapping, resolve


def _strip_speaker_prefix(text: str) -> Tuple[str, Optional[str]]:
    value = (text or "").strip()
    value = re.sub(r"^\[\d+(?:\.\d+)?\]\s*", "", value)
    match = _SPEAKER_PREFIX_RE.match(value)
    speaker = None
    if match:
        speaker = match.group("speaker")
        value = value[match.end() :].strip()
    value = _TRAILING_TIMESTAMP_RE.sub("", value).strip()
    return value, speaker


def _payload_text(payload: Dict[str, Any]) -> str:
    for key in ("text", "transcript", "transcription"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _segments_from_verbose_payload(
    payload: Dict[str, Any],
    resolve_speaker: Any,
) -> List[Dict[str, Any]]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return []

    segments: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_segments):
        if not isinstance(entry, dict):
            continue
        text_value = (
            entry.get("text")
            or entry.get("source_text")
            or entry.get("transcript")
            or entry.get("transcription")
            or ""
        )
        source_text, speaker_from_text = _strip_speaker_prefix(str(text_value))
        if not source_text:
            continue
        start, end = _timestamp_pair(entry)
        if start is None:
            start = segments[-1]["end"] if segments else 0.0
        if end is None:
            duration = max(0.6, len(source_text.split()) * 0.35)
            end = start + duration
        if end <= start:
            end = start + 0.05
        speaker_raw = (
            entry.get("speaker")
            or entry.get("speaker_id")
            or entry.get("speaker_label")
            or speaker_from_text
            or "S01"
        )
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": resolve_speaker(speaker_raw),
                "source_text": source_text,
                "translated_text": "",
                "moss_segment_index": idx,
            }
        )
    return segments


def _segments_from_compact_text(
    text: str,
    resolve_speaker: Any,
) -> List[Dict[str, Any]]:
    matches = list(_COMPACT_START_RE.finditer(text or ""))
    segments: List[Dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = _as_seconds(match.group("start"))
        if start is None:
            continue
        speaker = match.group("speaker")
        chunk_start = match.end()
        chunk_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[chunk_start:chunk_end]
        end_match = _TRAILING_TIMESTAMP_RE.search(chunk)
        if end_match:
            end = _as_seconds(end_match.group("end"))
            source_text = chunk[: end_match.start()]
        elif idx + 1 < len(matches):
            end = _as_seconds(matches[idx + 1].group("start"))
            source_text = chunk
        else:
            source_text = chunk
            end = start + max(0.6, len(source_text.split()) * 0.35)
        source_text = _BRACKET_TIMESTAMP_RE.sub("", source_text).strip()
        source_text, speaker_from_text = _strip_speaker_prefix(source_text)
        if not source_text:
            continue
        if end is None or end <= start:
            end = start + 0.05
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": resolve_speaker(speaker_from_text or speaker),
                "source_text": source_text,
                "translated_text": "",
                "moss_segment_index": idx,
            }
        )
    return segments


def _speaker_profiles(speaker_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    if not speaker_mapping:
        speaker_mapping["S01"] = "speaker1"
    return [
        {
            "id": speaker_id,
            "label": speaker_id.title(),
            "description": f"Detected by MOSS-Transcribe-Diarize as {raw_id}.",
            "moss_speaker": raw_id,
        }
        for raw_id, speaker_id in speaker_mapping.items()
    ]


def _payload_to_segments(
    payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    speaker_mapping, resolve_speaker = _speaker_mapper()
    segments = _segments_from_verbose_payload(payload, resolve_speaker)
    if not segments:
        segments = _segments_from_compact_text(_payload_text(payload), resolve_speaker)
    if not segments:
        raise RuntimeError("MOSS returned no usable diarized transcription segments.")
    return segments, _speaker_profiles(speaker_mapping)


def _translation_jobs(
    segments: List[Dict[str, Any]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for start in range(0, len(segments), batch_size):
        chunk = segments[start : start + batch_size]
        non_empty = [segment for segment in chunk if (segment.get("source_text") or "").strip()]
        if not non_empty:
            continue
        jobs.append(
            {
                "label": f"Batch {len(jobs) + 1}",
                "start": start + 1,
                "end": min(start + batch_size, len(segments)),
                "segments": non_empty,
                "source_texts": [segment["source_text"] for segment in non_empty],
            }
        )
    return jobs


def _translate_segments(
    segments: List[Dict[str, Any]],
    *,
    dest_language: str,
    source_language: Optional[str],
    llm_model: str,
    translation_batch_size: int,
    translation_max_workers: int,
) -> None:
    if (
        _make_translation_llm is None
        or _translate_texts_adaptively is None
        or _can_attempt_translation_backend is None
        or not _can_attempt_translation_backend()
    ):
        print("Warning: translation helpers are unavailable; leaving MOSS translations empty.")
        return

    jobs = _translation_jobs(segments, translation_batch_size)
    if not jobs:
        return
    worker_count = min(max(1, translation_max_workers), len(jobs))
    print(
        f"[MOSS] Translating {len(jobs)} batch(es) to {dest_language} "
        f"with up to {worker_count} worker(s)."
    )

    def run_job(job: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        llm = _make_translation_llm(
            llm_model=llm_model,
            dest_language=dest_language,
            source_language=source_language,
            batch_label=str(job["label"]),
        )
        translated = _translate_texts_adaptively(
            llm=llm,
            source_texts=job["source_texts"],
            dest_language=dest_language,
            batch_label=str(job["label"]),
            source_language=source_language,
        )
        return job, translated

    def apply_result(job: Dict[str, Any], translated_texts: Sequence[str]) -> None:
        if len(translated_texts) != len(job["segments"]):
            print(f"[MOSS] {job['label']} returned an unexpected translation count.")
            return
        for segment, translated_text in zip(job["segments"], translated_texts):
            segment["translated_text"] = translated_text

    if worker_count <= 1:
        for job in jobs:
            result_job, translated = run_job(job)
            apply_result(result_job, translated)
        return

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="moss_translate",
    ) as pool:
        pending: Dict[Any, Dict[str, Any]] = {}
        next_index = 0

        def submit_next() -> None:
            nonlocal next_index
            if next_index >= len(jobs):
                return
            job = jobs[next_index]
            pending[pool.submit(run_job, job)] = job
            next_index += 1

        for _ in range(worker_count):
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    result_job, translated = future.result()
                except Exception as exc:
                    print(f"[MOSS] {job['label']} translation failed: {exc}")
                else:
                    apply_result(result_job, translated)
                submit_next()


def _release_memory() -> None:
    try:
        import gc

        gc.collect()
    except Exception:
        pass


def translate_audio(
    audio_bytes: bytes,
    *,
    input_mime_type: Optional[str] = None,
    dest_language: str = "English",
    enable_translation: bool = True,
    translation_llm_model: Optional[str] = None,
    translation_batch_size: int = MOSS_TRANSCRIBE_TRANSLATION_BATCH_SIZE,
    translation_max_workers: int = MOSS_TRANSCRIBE_TRANSLATION_MAX_WORKERS,
    force_refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, Dict[str, Any]]:
    if not audio_bytes:
        raise RuntimeError("MOSS transcription received empty audio bytes.")

    translation_batch_size = max(1, int(translation_batch_size or 1))
    try:
        translation_max_workers = int(translation_max_workers or 1)
    except (TypeError, ValueError):
        translation_max_workers = MOSS_TRANSCRIBE_TRANSLATION_MAX_WORKERS
    translation_max_workers = max(1, min(10, translation_max_workers))
    llm_model = translation_llm_model or MOSS_TRANSCRIBE_TRANSLATION_LLM

    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    cache_key = _cache_key(
        audio_hash,
        dest_language=dest_language,
        enable_translation=enable_translation,
        translation_llm_model=llm_model,
        input_mime_type=input_mime_type,
    )
    cache_info: Dict[str, Any] = {
        "audio_md5": audio_hash,
        "hit": False,
        "force_refresh": bool(force_refresh),
        "pipeline": "moss_transcribe",
        "moss_model": MOSS_TRANSCRIBE_MODEL,
        "sglang_url": MOSS_TRANSCRIBE_SGLANG_URL,
        "response_format": MOSS_TRANSCRIBE_RESPONSE_FORMAT,
        "max_new_tokens": MOSS_TRANSCRIBE_MAX_NEW_TOKENS,
        "translation_batch_size": translation_batch_size,
        "translation_max_workers": translation_max_workers,
    }

    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached and isinstance(cached.get("segments"), list):
            cache_info["hit"] = True
            cache_info["cache_file"] = os.path.basename(_cache_path(cache_key))
            cache_info["created_at"] = cached.get("created_at")
            cache_info["translation_llm_model"] = cached.get("translation_llm_model")
            cache_info["source_language"] = cached.get("source_language")
            return (
                cached["segments"],
                cached.get("speaker_profiles") or [],
                cached.get("raw_text", ""),
                cache_info,
            )

    try:
        _ensure_backend_ready()
        print("[MOSS] Sending audio to SGLang-Omni transcription endpoint...")
        payload = _transcribe_with_sglang(
            audio_bytes,
            input_mime_type=input_mime_type,
        )
        segments, speaker_profiles = _payload_to_segments(payload)
        source_language = str(payload.get("language") or payload.get("source_language") or "auto")
        cache_info["source_language"] = source_language
        cache_info["segment_count"] = len(segments)
        cache_info["speaker_count"] = len(speaker_profiles)

        if enable_translation and dest_language:
            _translate_segments(
                segments,
                dest_language=dest_language,
                source_language=source_language,
                llm_model=llm_model,
                translation_batch_size=translation_batch_size,
                translation_max_workers=translation_max_workers,
            )
            cache_info["translation_llm_model"] = llm_model
        elif not enable_translation:
            for segment in segments:
                segment["translated_text"] = segment.get("source_text", "")

        raw_output = {
            "pipeline": "moss_transcribe",
            "moss_model": MOSS_TRANSCRIBE_MODEL,
            "sglang_url": MOSS_TRANSCRIBE_SGLANG_URL,
            "max_new_tokens": MOSS_TRANSCRIBE_MAX_NEW_TOKENS,
            "source_language": source_language,
            "response": payload,
            "speakers": speaker_profiles,
            "segments": segments,
        }
        raw_text = json.dumps(raw_output, ensure_ascii=False, indent=2)
        cache_path = _write_cache(
            cache_key,
            {
                "created_at": time.time(),
                "segments": segments,
                "speaker_profiles": speaker_profiles,
                "raw_text": raw_text,
                "source_language": source_language,
                "translation_llm_model": llm_model if enable_translation else None,
            },
        )
        if cache_path:
            cache_info["cache_file"] = os.path.basename(cache_path)
        return segments, speaker_profiles, raw_text, cache_info
    finally:
        _release_memory()


_run_moss_transcribe_pipeline_sync = translate_audio
