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
        self._exclusive_owner = None
        self._exclusive_waiters = 0

    @asynccontextmanager
    async def use(self, backend: str):
        task = asyncio.current_task()
        if task is self._exclusive_owner:
            yield
            return
        if task in self._owners:
            if self.enabled and self._owners[task] != backend:
                raise RuntimeError("Cannot switch TTS backends during an active synthesis operation")
            yield
            return
        async with self._condition:
            await self._condition.wait_for(lambda: self._exclusive_owner is None
                                           and not self._exclusive_waiters
                                           and (not self.enabled or self._backend in {None, backend}))
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

    @asynccontextmanager
    async def exclusive(self):
        """Manual memory changes wait for active jobs, even on large GPUs."""
        task = asyncio.current_task()
        if task is self._exclusive_owner:
            yield
            return
        if task in self._owners:
            raise RuntimeError("Cannot unload models inside active GPU work")
        async with self._condition:
            self._exclusive_waiters += 1
            try:
                await self._condition.wait_for(lambda: not self._owners and self._exclusive_owner is None)
                self._exclusive_owner = task
            finally:
                self._exclusive_waiters -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            async with self._condition:
                self._exclusive_owner = None
                self._condition.notify_all()


async def await_gpu_job(job):
    """Keep a GPU lease until a worker thread exits, including cancellation."""
    future = asyncio.ensure_future(job)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        # Cancelling run_in_executor does not stop its GPU inference thread.
        # Repeated cancellation must not let another backend reclaim its model.
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not future.cancelled():
            future.exception()
        raise


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
