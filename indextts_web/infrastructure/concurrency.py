"""Central ownership for application concurrency limits and executors."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass(slots=True)
class ConcurrencyBudget:
    index_tts_requests: int = 100
    translation_tts_requests: int = 100
    general_workers: int = 8
    io_workers: int = 4
    audio_workers: int = 2
    higgs_workers: int = 100
    index_tts: asyncio.Semaphore = field(init=False)
    translation_tts: asyncio.Semaphore = field(init=False)
    general: ThreadPoolExecutor = field(init=False)
    io: ThreadPoolExecutor = field(init=False)
    audio: ThreadPoolExecutor = field(init=False)
    higgs: ThreadPoolExecutor = field(init=False)

    def __post_init__(self) -> None:
        self.index_tts = asyncio.BoundedSemaphore(max(1, self.index_tts_requests))
        self.translation_tts = asyncio.BoundedSemaphore(max(1, self.translation_tts_requests))
        self.general = ThreadPoolExecutor(
            max_workers=max(1, self.general_workers),
            thread_name_prefix="indextts_general",
        )
        self.io = ThreadPoolExecutor(max_workers=max(1, self.io_workers), thread_name_prefix="indextts_io")
        self.audio = ThreadPoolExecutor(max_workers=max(1, self.audio_workers), thread_name_prefix="indextts_audio")
        self.higgs = ThreadPoolExecutor(max_workers=max(1, self.higgs_workers), thread_name_prefix="indextts_higgs")

    @classmethod
    def from_environ(cls) -> ConcurrencyBudget:
        def integer(name: str, default: int, maximum: int = 256) -> int:
            try:
                return min(maximum, max(1, int(os.environ.get(name, default))))
            except (TypeError, ValueError):
                return default

        index_requests = integer("INDEXTTS_GPU_WORK_CONCURRENCY", 100)
        higgs_running = integer("HIGGS_TTS_MAX_RUNNING_REQUESTS", 100)
        higgs_work = integer("HIGGS_TTS_WORK_CONCURRENCY", higgs_running)
        return cls(
            index_tts_requests=index_requests,
            translation_tts_requests=integer("TRANSLATION_TTS_CONCURRENCY", 100, index_requests),
            general_workers=min(8, max(4, os.cpu_count() or 4)),
            io_workers=integer("INDEXTTS_IO_WORKERS", 4, 16),
            audio_workers=integer("INDEXTTS_AUDIO_WORKERS", 2, 8),
            higgs_workers=integer("HIGGS_HTTP_WORKERS", higgs_work),
        )

    def shutdown(self) -> None:
        for executor in (self.general, self.io, self.audio, self.higgs):
            executor.shutdown(wait=True, cancel_futures=True)
