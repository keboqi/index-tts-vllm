"""Keep backend transitions away from active synthesis on shared small GPUs."""

from __future__ import annotations

import asyncio
import functools
import inspect
from contextlib import aclosing, asynccontextmanager


class GpuWorkCoordinator:
    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self._condition = asyncio.Condition()
        self._backend: str | None = None
        self._owners: dict[asyncio.Task, str] = {}

    @asynccontextmanager
    async def use(self, backend: str):
        if not self.enabled:
            yield
            return
        task = asyncio.current_task()
        if task in self._owners:
            if self._owners[task] != backend:
                raise RuntimeError("Cannot switch TTS backends during an active synthesis operation")
            yield
            return
        async with self._condition:
            await self._condition.wait_for(lambda: self._backend in {None, backend})
            self._backend = backend
            self._owners[task] = backend
        try:
            yield
        finally:
            async with self._condition:
                self._owners.pop(task, None)
                if not self._owners:
                    self._backend = None
                    self._condition.notify_all()


def gpu_operation(backend: str):
    """Wrap coroutine/generator lifetimes, including cancellation and streaming."""
    def decorate(function):
        @asynccontextmanager
        async def operation(instance):
            coordinator = getattr(instance, "gpu_coordinator", None)
            if coordinator is None:
                yield
                return
            async with coordinator.use(backend):
                prepare = getattr(instance, "prepare_gpu_callback", None)
                if prepare is not None:
                    await prepare()
                yield

        if inspect.isasyncgenfunction(function):
            @functools.wraps(function)
            async def stream(self, *args, **kwargs):
                async with operation(self), aclosing(function(self, *args, **kwargs)) as chunks:
                    async for chunk in chunks:
                        yield chunk
            return stream

        @functools.wraps(function)
        async def invoke(self, *args, **kwargs):
            async with operation(self):
                return await function(self, *args, **kwargs)
        return invoke
    return decorate
