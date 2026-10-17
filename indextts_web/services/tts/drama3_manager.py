"""Async app adapter for the isolated Drama 3 SGLang-Omni runtime."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from indextts_web.infrastructure.gpu_work import await_gpu_job, gpu_operation

from .chunking import TextMarkup, join_wav, render_segments, split_text
from .drama3_serving import environment
from .drama3_serving.server import ServerSettings, SglangOmniServer, ensure_model, persist_caches

# SGLang-Omni rejects Drama 3 requests above this; shared callers may ask for more.
MAX_NEW_TOKENS = 2048


# Drama 3 inline control tags (fishaudio/drama-3 model card, fish.audio/drama-3):
# point directions or sounds in square brackets ("[sighs]", "[horrified, staring
# at the plate]"), paired tags scoping words ("<whisper>Go on.</whisper>"), and
# speaker tags "<|speaker:0|>".."<|speaker:15|>". Text without a leading speaker
# tag is spoken by speaker 0.
_SPEAKER_TAG = re.compile(r"<\|speaker:\d+\|>")
DRAMA3_MARKUP = TextMarkup(
    atomic=re.compile(r"<\|speaker:\d+\|>|</?[A-Za-z][\w-]*>|\[[^\[\]\n]*\]"),
    speaker=_SPEAKER_TAG,
    paired_open=re.compile(r"<([A-Za-z][\w-]*)>"),
    paired_close=re.compile(r"</([A-Za-z][\w-]*)>"),
)


def emotion_prefix(text: str, emotion: str | None) -> str:
    description = (emotion or "").strip().strip("[]").strip()
    if not description:
        return text
    # Keep a leading speaker tag first so the chunk is not reassigned to speaker 0.
    lead = _SPEAKER_TAG.match(text)
    if lead:
        return f"{lead.group(0)} [{description}] {text[lead.end():].lstrip()}"
    return f"[{description}] {text}"


class ManagedDrama3Backend:
    def __init__(self, settings, *, app_dir: Path, output_root: Path, prepare_gpu=None):
        self.settings = settings
        self.prepare_gpu = prepare_gpu
        self.gpu_coordinator = None
        self._lock = asyncio.Lock()
        self._active_jobs = 0
        self._server = SglangOmniServer(ServerSettings(
            venv=Path(settings.drama3_venv).expanduser() if settings.drama3_venv else environment.default_venv(app_dir),
            data_dir=Path(settings.drama3_data_dir).expanduser() if settings.drama3_data_dir else output_root / "drama3-serving",
            model_root=Path(settings.drama3_model_dir).expanduser() if settings.drama3_model_dir else app_dir / "checkpoints" / "drama3",
            port=settings.drama3_port,
            engine_memory=settings.drama3_engine_memory,
            codec_memory=settings.drama3_codec_memory,
            max_seqs=settings.drama3_max_parallel_segments,
            start_timeout=settings.drama3_start_timeout,
            request_timeout=settings.drama3_request_timeout,
            sleep_mode=settings.drama3_sleep,
        ))
        # Chunks of every request share the server's running-request budget,
        # on threads of their own so the default executor never caps them.
        self._segment_slots = asyncio.Semaphore(settings.drama3_max_parallel_segments)
        self._speech_pool = ThreadPoolExecutor(settings.drama3_max_parallel_segments, thread_name_prefix="drama3-chunk")
        self._server.checkpoint_launch_enabled = os.getenv("DRAMA3_MODAL_SNAPSHOT_CAPTURE") != "1"

    def process_running(self) -> bool:
        return self._server.running()

    @property
    def sleeping(self) -> bool:
        return self._server.sleeping

    @gpu_operation("drama3")
    async def ensure_ready(self) -> None:
        async with self._lock:
            await self._ensure_ready_locked()

    async def _ensure_ready_locked(self) -> None:
        if self.prepare_gpu is not None:
            await self.prepare_gpu()
        if not self.process_running():
            await await_gpu_job(asyncio.to_thread(environment.prepare, self._server.settings.venv))
            await await_gpu_job(asyncio.to_thread(ensure_model, self._server.settings, token=os.getenv("HF_TOKEN")))
        await await_gpu_job(asyncio.to_thread(self._server.ensure_ready, lambda: None, print))

    async def sleep_vllm(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("Drama 3 supports wakeable sleep only at level 1")
        async with self._lock:
            if self._active_jobs:
                raise RuntimeError("Drama 3 has active requests and cannot sleep yet")
            await await_gpu_job(asyncio.to_thread(self._server.release))

    async def wake_vllm(self) -> None:
        await self.ensure_ready()

    def build_payload(self, *, text, prompt_wav=None, reference_text=None, emotion_text=None, sampling=None):
        values = {key: value for key, value in dict(sampling or {}).items() if value is not None}
        controls = {key: values[key] for key in ("temperature", "top_p", "top_k") if key in values}
        controls.setdefault("temperature", 1.0)
        controls.setdefault("top_p", 0.9)
        controls.setdefault("top_k", 30)
        payload = self._server.payload(
            text=emotion_prefix(text, emotion_text), reference=prompt_wav or ".",
            transcript=(reference_text or "").strip(),
            max_new_tokens=min(int(values.get("max_new_tokens", MAX_NEW_TOKENS)), MAX_NEW_TOKENS),
            sampling=controls,
        )
        if not prompt_wav:
            payload["references"] = []
        return payload

    def _speak_chunk(self, payload: dict) -> bytes:
        content = self._server.speech(payload)
        if not content.startswith(b"RIFF") or content[8:12] != b"WAVE":
            raise RuntimeError("Drama 3 returned an invalid WAV response")
        return content

    def _stage_reference(self, prompt_wav: str) -> str:
        source = Path(prompt_wav)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        staged = self._server.settings.media_dir / "references" / f"{digest}{source.suffix.lower()}"
        staged.parent.mkdir(parents=True, exist_ok=True)
        if not staged.is_file():
            temporary = staged.with_name(f".{staged.name}.{uuid.uuid4().hex}.tmp")
            try:
                shutil.copyfile(source, temporary)
                temporary.replace(staged)
            finally:
                temporary.unlink(missing_ok=True)
        return str(staged)

    def _write_output(self, clips: list[bytes], output_path, interval_silence_ms: int) -> str:
        content = clips[0] if len(clips) == 1 else join_wav(clips, max(0, int(interval_silence_ms)))
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        persist_caches()
        return str(target)

    async def _render(self, *, text, output_path, prompt_wav=None, max_text_tokens=120,
                      interval_silence_ms=0, **kwargs) -> str:
        chunks = split_text(text, max(1, int(max_text_tokens)), markup=DRAMA3_MARKUP)
        if not chunks:
            raise ValueError("Drama 3 speech text is required")
        if prompt_wav:
            prompt_wav = await asyncio.to_thread(self._stage_reference, prompt_wav)
        payloads = [self.build_payload(text=chunk, prompt_wav=prompt_wav, **kwargs) for chunk in chunks]
        loop = asyncio.get_running_loop()
        clips = await render_segments(
            payloads, lambda payload: loop.run_in_executor(self._speech_pool, self._speak_chunk, payload),
            slots=self._segment_slots,
        )
        return await asyncio.to_thread(self._write_output, clips, output_path, interval_silence_ms)

    @gpu_operation("drama3")
    async def synthesize_to_file(self, **kwargs) -> str:
        async with self._lock:
            await self._ensure_ready_locked()
            self._active_jobs += 1
        try:
            return await await_gpu_job(self._render(**kwargs))
        finally:
            async with self._lock:
                self._active_jobs -= 1

    async def status(self) -> dict:
        return {
            "backend": "drama3", "running": self.process_running(), "sleeping": self.sleeping,
            "ready": self.process_running() and not self.sleeping and await asyncio.to_thread(self._server.healthy),
            "base_url": self._server.base_url, "log_path": str(self._server.log_path or ""),
            "sleep_mode": self._server.settings.sleep_mode,
            "sleep_available": self._server._checkpoint_helper is not None and not self._server._sleep_disabled,
            "emotion_text": True, "native_duration": False, "native_streaming": False,
        }

    async def shutdown(self) -> None:
        async with self._lock:
            await await_gpu_job(asyncio.to_thread(self._server.stop))
