"""Managed IndexTTS 2.5 vLLM-Omni process and HTTP client."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import mimetypes
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Awaitable, Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from indextts_web.config import AppSettings

INDEXTTS25_BACKEND = "index25"
INDEXTTS25_LANGUAGES = ("zh", "en", "ja", "es", "ar")
_TEXT_BOUNDARY = re.compile(r"(?<=[.!?。！？；;])\s*")
_LANGUAGE_ALIASES = {
    "arabic": "ar",
    "ar": "ar",
    "chinese": "zh",
    "mandarin": "zh",
    "zh": "zh",
    "zh-cn": "zh",
    "zh_cn": "zh",
    "english": "en",
    "en": "en",
    "japanese": "ja",
    "jp": "ja",
    "ja": "ja",
    "spanish": "es",
    "es": "es",
}


@lru_cache(maxsize=64)
def _cached_audio_reference(path_value: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    path = Path(path_value)
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def audio_reference(value: str, *, use_cache: bool = True) -> str:
    if value.startswith(("http://", "https://", "data:", "file://")):
        return value
    path = Path(value).expanduser().resolve()
    stat = path.stat()
    if use_cache:
        return _cached_audio_reference(str(path), stat.st_mtime_ns, stat.st_size)
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def split_text(text: str, max_tokens: int) -> list[str]:
    value = text.strip()
    if not value:
        return []
    pieces = [piece.strip() for piece in _TEXT_BOUNDARY.split(value) if piece.strip()]
    result: list[str] = []
    for piece in pieces:
        words = piece.split()
        if len(words) == 1 and len(piece) > max_tokens:
            result.extend(piece[index : index + max_tokens] for index in range(0, len(piece), max_tokens))
        elif len(words) <= max_tokens:
            result.append(piece)
        else:
            result.extend(
                " ".join(words[index : index + max_tokens])
                for index in range(0, len(words), max_tokens)
            )
    return result


def allocate_durations(texts: list[str], total_ms: int, silence_ms: int) -> list[int]:
    if not texts:
        return []
    if total_ms <= 0:
        return [0] * len(texts)
    available = total_ms - max(0, len(texts) - 1) * silence_ms
    if available < len(texts):
        raise ValueError("target duration is shorter than requested inter-sentence silence")
    weights = [max(1, len(text)) for text in texts]
    weight_sum = sum(weights)
    durations = [max(1, round(available * weight / weight_sum)) for weight in weights]
    durations[-1] += available - sum(durations)
    return durations


def join_wav(chunks: list[bytes], silence_ms: int = 0) -> bytes:
    if not chunks:
        raise ValueError("IndexTTS 2.5 returned no WAV chunks")
    output = io.BytesIO()
    params = None
    frames: list[bytes] = []
    for index, chunk in enumerate(chunks):
        with wave.open(io.BytesIO(chunk), "rb") as source:
            current = source.getparams()
            signature = (current.nchannels, current.sampwidth, current.framerate, current.comptype)
            if params is None:
                params = current
            elif signature != (params.nchannels, params.sampwidth, params.framerate, params.comptype):
                raise ValueError("IndexTTS 2.5 WAV chunks use incompatible audio formats")
            frames.append(source.readframes(source.getnframes()))
            if index + 1 < len(chunks) and silence_ms:
                silence_frames = round(current.framerate * silence_ms / 1000)
                frames.append(b"\x00" * silence_frames * current.nchannels * current.sampwidth)
    assert params is not None
    with wave.open(output, "wb") as target:
        target.setparams(params)
        target.writeframes(b"".join(frames))
    return output.getvalue()


def fit_wav_duration(wav_bytes: bytes, target_ms: int) -> bytes:
    source_io = io.BytesIO(wav_bytes)
    output = io.BytesIO()
    with wave.open(source_io, "rb") as source:
        params = source.getparams()
        raw = source.readframes(source.getnframes())
    frame_width = params.nchannels * params.sampwidth
    target_frames = round(params.framerate * target_ms / 1000)
    target_bytes = target_frames * frame_width
    fitted = raw[:target_bytes].ljust(target_bytes, b"\x00")
    with wave.open(output, "wb") as target:
        target.setparams(params)
        target.writeframes(fitted)
    return output.getvalue()


class ManagedIndexTTS25Backend:
    """Lazily launch the isolated vLLM-Omni server and serve WAV requests."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        app_dir: Path,
        output_root: Path,
        prepare_gpu: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings
        self.app_dir = app_dir.resolve()
        self.output_root = output_root.resolve()
        self.prepare_gpu = prepare_gpu
        self._process: subprocess.Popen[bytes] | None = None
        self._started_process = False
        self._want_running = False
        self._started_at: float | None = None
        self._last_ready_at: float | None = None
        self._last_health: dict[str, Any] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._log_file_handle: Any | None = None
        self._log_path: Path | None = None
        self._last_exit_code: int | None = None
        self._last_exit_at: float | None = None
        self._last_start_command: list[str] = []
        self._active_requests = 0
        self._lock = asyncio.Lock()
        self._segment_slots = asyncio.Semaphore(max(1, settings.indextts25_max_parallel_segments))

    @property
    def base_url(self) -> str:
        host = (self.settings.indextts25_host or "127.0.0.1").strip()
        return f"http://{host}:{int(self.settings.indextts25_port)}"

    @property
    def repo_dir(self) -> Path:
        path = Path(self.settings.indextts25_repo_dir).expanduser()
        if not path.is_absolute():
            path = self.app_dir / path
        return path.resolve()

    @property
    def model_dir(self) -> Path:
        configured = self.settings.indextts25_model_dir.strip()
        path = Path(configured).expanduser() if configured else self.repo_dir / "models" / "IndexTTS-2.5"
        if not path.is_absolute():
            path = self.app_dir / path
        return path.resolve()

    @property
    def data_dir(self) -> Path:
        configured = self.settings.indextts25_data_dir.strip()
        path = Path(configured).expanduser() if configured else self.repo_dir / "runtime" / "indextts25"
        if not path.is_absolute():
            path = self.app_dir / path
        return path.resolve()

    def _log_dir(self) -> Path:
        configured = self.settings.indextts25_log_dir.strip()
        path = Path(configured).expanduser() if configured else self.output_root / "indextts25_backend_logs"
        if not path.is_absolute():
            path = self.app_dir / path
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _poll_process(self) -> int | None:
        if self._process is None:
            return None
        returncode = self._process.poll()
        if returncode is not None:
            if self._last_exit_at is None or self._last_exit_code != returncode:
                self._last_exit_code = returncode
                self._last_exit_at = time.monotonic()
        return returncode

    def process_running(self) -> bool:
        return self._process is not None and self._poll_process() is None

    def _build_start_command(self) -> tuple[list[str], dict[str, str]]:
        env = os.environ.copy()
        env.update(
            {
                "INDEXTTS25_HOST": self.settings.indextts25_host,
                "INDEXTTS25_PORT": str(self.settings.indextts25_port),
                "INDEXTTS25_MODEL_DIR": str(self.model_dir),
                "INDEXTTS25_DATA_DIR": str(self.data_dir),
                "INDEXTTS25_SERVED_MODEL_NAME": self.settings.indextts25_served_model_name,
                "PYTHONUNBUFFERED": "1",
            }
        )
        env.setdefault("PYTHONIOENCODING", "utf-8")
        custom = self.settings.indextts25_start_command.strip()
        if custom:
            if self.settings.indextts25_start_shell:
                if os.name == "nt":
                    shell = shutil.which("powershell") or shutil.which("pwsh")
                    if shell:
                        return [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", custom], env
                    return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", custom], env
                return [shutil.which("bash") or "/bin/sh", "-lc", custom], env
            return shlex.split(custom, posix=os.name != "nt"), env

        launcher = self.repo_dir / "experiments" / "indextts25_backend_compat" / "serve_api.sh"
        if os.name == "nt":
            raise RuntimeError(
                "The default IndexTTS 2.5 vLLM-Omni launcher requires Linux; "
                "set --indextts25_start_command for a remote or custom server."
            )
        if not launcher.exists():
            raise RuntimeError(
                f"IndexTTS 2.5 integration launcher not found at {launcher}. "
                "Set --indextts25_repo_dir to the experiment repository."
            )
        return ["bash", str(launcher)], env

    def _open_log_handle(self, command: list[str]) -> Any | None:
        self._close_log_handle()
        if self.settings.indextts25_attach_stdio:
            self._log_path = None
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = self._log_dir() / f"indextts25_{timestamp}_{uuid.uuid4().hex[:8]}.log"
        handle = open(self._log_path, "ab", buffering=0)
        header = (
            f"\n\n=== IndexTTS 2.5 managed backend start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"cwd: {self.repo_dir}\n"
            f"command: {' '.join(shlex.quote(str(part)) for part in command)}\n\n"
        )
        handle.write(header.encode("utf-8", errors="replace"))
        self._log_file_handle = handle
        return handle

    def _close_log_handle(self) -> None:
        handle = self._log_file_handle
        self._log_file_handle = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _popen_kwargs(self, log_handle: Any | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "close_fds": True}
        if log_handle is not None:
            kwargs.update({"stdout": log_handle, "stderr": subprocess.STDOUT})
        if self.settings.indextts25_detach_process:
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                kwargs["start_new_session"] = True
        return kwargs

    def _terminate_process_tree(self, process: subprocess.Popen[bytes], *, kill: bool = False) -> None:
        if self.settings.indextts25_detach_process and os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except Exception:
                pass
        if os.name == "nt" and self.settings.indextts25_detach_process and not kill:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                return
            except Exception:
                pass
        process.kill() if kill else process.terminate()

    def _start_sync(self) -> None:
        if self.process_running():
            return
        command, env = self._build_start_command()
        self._want_running = True
        self._last_start_command = command
        log_handle = self._open_log_handle(command)
        working_directory = self.repo_dir if self.repo_dir.exists() else self.app_dir
        self._process = subprocess.Popen(
            command,
            cwd=str(working_directory),
            env=env,
            **self._popen_kwargs(log_handle),
        )
        self._started_process = True
        self._started_at = time.monotonic()
        self._last_exit_code = None
        self._last_exit_at = None
        print(f"[IndexTTS 2.5] Lazy-started vLLM-Omni on {self.base_url}; log: {self._log_path}")

    def _stop_sync(self, reason: str, force: bool = False) -> None:
        if self._active_requests and not force:
            raise RuntimeError(
                f"IndexTTS 2.5 has {self._active_requests} active request(s); cannot switch backends yet"
            )
        process = self._process
        if process is not None and self._started_process and process.poll() is None:
            print(f"[IndexTTS 2.5] Stopping vLLM-Omni ({reason})")
            self._terminate_process_tree(process)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._terminate_process_tree(process, kill=True)
                process.wait()
        elif process is not None:
            self._poll_process()
        self._process = None
        self._started_process = False
        self._started_at = None
        self._last_health = None
        self._last_ready_at = None
        self._close_log_handle()

    def _health_sync(self, timeout: float) -> dict[str, Any] | None:
        try:
            request = urllib.request.Request(f"{self.base_url}/v1/models", headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = payload.get("data", [])
            model_ids = {
                str(item.get("id"))
                for item in models
                if isinstance(item, dict) and item.get("id") is not None
            }
            expected = self.settings.indextts25_served_model_name
            return {
                "model_loaded": expected in model_ids,
                "expected_model": expected,
                "models": models,
            }
        except Exception:
            return None

    async def health(self, timeout: float = 1.0) -> dict[str, Any] | None:
        health = await asyncio.to_thread(self._health_sync, timeout)
        self._last_health = health
        if health and health.get("model_loaded"):
            self._last_ready_at = time.monotonic()
        return health

    async def ensure_ready(self) -> dict[str, Any]:
        async with self._lock:
            self._want_running = True
            health = await self.health(1.0)
            if health and health.get("model_loaded"):
                return health
            if health is not None and not self.process_running():
                raise RuntimeError(
                    f"Port {self.settings.indextts25_port} is serving a different model; "
                    f"expected {self.settings.indextts25_served_model_name!r}, got {health.get('models')}"
                )
            if self.prepare_gpu is not None:
                await self.prepare_gpu()
            if not self.process_running():
                await asyncio.to_thread(self._start_sync)
            timeout = max(1.0, float(self.settings.indextts25_start_timeout))
            started = time.perf_counter()
            last_health = None
            while time.perf_counter() - started < timeout:
                last_health = await self.health(2.0)
                if last_health and last_health.get("model_loaded"):
                    print("[IndexTTS 2.5] vLLM-Omni backend is ready.")
                    return last_health
                if self._process is not None and self._process.poll() is not None:
                    code = self._process.returncode
                    raise RuntimeError(
                        f"IndexTTS 2.5 vLLM-Omni exited during startup with code {code}; log: {self._log_path}"
                    )
                await asyncio.sleep(2.0)
            raise RuntimeError(
                f"Timed out waiting {timeout:.0f}s for IndexTTS 2.5 vLLM-Omni; last health: {last_health}"
            )

    async def stop(self, reason: str = "backend switch") -> None:
        async with self._lock:
            self._want_running = False
            await asyncio.to_thread(self._stop_sync, reason, False)

    @staticmethod
    def resolve_language(language: str | None, text: str) -> str:
        normalized = (language or "").strip().lower()
        if " - " in normalized:
            normalized = normalized.split(" - ", 1)[0].strip()
        resolved = _LANGUAGE_ALIASES.get(normalized)
        if resolved:
            return resolved
        if normalized and normalized != "auto":
            raise ValueError(
                "unsupported IndexTTS 2.5 language; expected zh, en, ja, es, ar, or auto"
            )
        if any("\u3040" <= char <= "\u30ff" for char in text):
            return "ja"
        if any("\u0600" <= char <= "\u06ff" for char in text):
            return "ar"
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return "zh"
        return "en"

    def build_payload(
        self,
        *,
        text: str,
        language: str | None,
        prompt_wav: str,
        reference_text: str | None,
        target_duration_ms: int,
        diffusion_steps: int,
        emotion_audio: str | None,
        emotion_text: str | None,
        emotion_weight: float,
        cache_prompt_audio: bool,
        seed: int | None,
        sampling: Mapping[str, Any],
    ) -> dict[str, Any]:
        extras = {key: value for key, value in sampling.items() if value is not None}
        max_new_tokens = extras.pop("max_new_tokens", None)
        extras.update(
            {
                "lang": self.resolve_language(language, text),
                "text_normalization": True,
                "diffusion_steps": max(1, int(diffusion_steps or 10)),
                "cache_prompt_audio": bool(cache_prompt_audio),
            }
        )
        if target_duration_ms > 0:
            extras["target_duration_ms"] = int(target_duration_ms)
        if emotion_audio:
            extras["emo_audio"] = audio_reference(emotion_audio, use_cache=cache_prompt_audio)
        if emotion_text:
            extras.update({"emo_text": emotion_text, "use_emo_text": True})
        if emotion_audio or emotion_text:
            extras["emo_alpha"] = min(1.0, max(0.0, float(emotion_weight)))
        payload: dict[str, Any] = {
            "model": self.settings.indextts25_served_model_name,
            "input": text,
            "response_format": "wav",
            "stream": False,
            "ref_audio": audio_reference(prompt_wav, use_cache=cache_prompt_audio),
            "extra_params": extras,
        }
        if reference_text:
            payload["ref_text"] = reference_text
        if seed is not None:
            payload["seed"] = int(seed)
        if max_new_tokens is not None:
            payload["max_new_tokens"] = int(max_new_tokens)
        return payload

    def _post_audio_sync(self, payload: dict[str, Any], timeout: float) -> bytes:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/audio/speech",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"IndexTTS 2.5 vLLM-Omni HTTP {exc.code}: {detail}") from exc

    async def _synthesize_segment(self, payload: dict[str, Any]) -> bytes:
        async with self._segment_slots:
            timeout = max(1.0, float(self.settings.indextts25_request_timeout))
            return await asyncio.to_thread(self._post_audio_sync, payload, timeout)

    async def synthesize_to_file(
        self,
        *,
        text: str,
        output_path: str,
        language: str | None,
        prompt_wav: str | None,
        reference_text: str | None = None,
        speech_length: int = 0,
        interval_silence: int = 0,
        diffusion_steps: int = 10,
        max_text_tokens_per_sentence: int = 120,
        emotion_audio: str | None = None,
        emotion_text: str | None = None,
        emotion_weight: float = 0.6,
        cache_prompt_audio: bool = True,
        seed: int | None = None,
        sampling: Mapping[str, Any] | None = None,
    ) -> str:
        if not text.strip():
            raise ValueError("IndexTTS 2.5 synthesis text cannot be empty")
        if not prompt_wav:
            raise ValueError("IndexTTS 2.5 requires prompt audio or a speaker preset")
        max_tokens = max(1, int(max_text_tokens_per_sentence or 120))
        texts = split_text(text, max_tokens)
        durations = allocate_durations(texts, max(0, int(speech_length)), max(0, int(interval_silence)))
        self._active_requests += 1
        try:
            await self.ensure_ready()
            payloads = [
                self.build_payload(
                    text=segment,
                    language=language,
                    prompt_wav=prompt_wav,
                    reference_text=reference_text,
                    target_duration_ms=duration,
                    diffusion_steps=diffusion_steps,
                    emotion_audio=emotion_audio,
                    emotion_text=emotion_text,
                    emotion_weight=emotion_weight,
                    cache_prompt_audio=cache_prompt_audio,
                    seed=None if seed is None else seed + index,
                    sampling=sampling or {},
                )
                for index, (segment, duration) in enumerate(zip(texts, durations, strict=True))
            ]
            chunks = await asyncio.gather(*(self._synthesize_segment(payload) for payload in payloads))
        finally:
            self._active_requests = max(0, self._active_requests - 1)
        audio = join_wav(chunks, max(0, int(interval_silence)))
        if speech_length > 0:
            audio = fit_wav_duration(audio, int(speech_length))
        output = Path(output_path)
        await asyncio.to_thread(output.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(output.write_bytes, audio)
        return output_path

    async def status(self) -> dict[str, Any]:
        health = await self.health(1.0)
        return {
            "backend": INDEXTTS25_BACKEND,
            "lazy": True,
            "base_url": self.base_url,
            "repo_dir": str(self.repo_dir),
            "model_dir": str(self.model_dir),
            "data_dir": str(self.data_dir),
            "served_model_name": self.settings.indextts25_served_model_name,
            "managed_pid": self._process.pid if self.process_running() else None,
            "managed_process_running": self.process_running(),
            "healthy": bool(health and health.get("model_loaded")),
            "health": health,
            "active_requests": self._active_requests,
            "max_parallel_segments": self.settings.indextts25_max_parallel_segments,
            "keepalive_interval": self.settings.indextts25_keepalive_interval,
            "managed_log_path": str(self._log_path) if self._log_path else None,
            "last_exit_code": self._last_exit_code,
            "last_start_command": self._last_start_command,
        }

    async def _keepalive_loop(self) -> None:
        interval = max(0.0, float(self.settings.indextts25_keepalive_interval))
        while interval > 0:
            await asyncio.sleep(interval)
            try:
                if not self._want_running or self._active_requests:
                    continue
                health = await self.health(2.0)
                if health and health.get("model_loaded"):
                    continue
                if not self.process_running():
                    await self.ensure_ready()
                elif self._last_ready_at is not None:
                    unhealthy_for = time.monotonic() - self._last_ready_at
                    if unhealthy_for >= max(1.0, float(self.settings.indextts25_unhealthy_grace)):
                        async with self._lock:
                            await asyncio.to_thread(self._stop_sync, "unhealthy keepalive restart", True)
                        await self.ensure_ready()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[IndexTTS 2.5] Keepalive check failed: {exc}")

    def start_keepalive(self) -> None:
        if self.settings.indextts25_keepalive_interval <= 0:
            return
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def shutdown(self) -> None:
        self._want_running = False
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
        self._keepalive_task = None
        async with self._lock:
            await asyncio.to_thread(self._stop_sync, "application shutdown", True)
