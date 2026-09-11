"""CUDA worker used by both IndexTTS and Qwen Emotion (vLLM 0.10.2)."""

import torch
from vllm.device_allocator import cumem
from vllm.v1.worker.gpu_worker import Worker

from .vllm_memory import wake_allocations


class RecoverableSleepWorker(Worker):
    def wake_up(self, tags=None):
        allocator = cumem.CuMemAllocator.get_instance()
        original = allocator.wake_up

        def guarded_wake(tags=None):
            wake_allocations(
                allocator, tags=tags, free_bytes=lambda: torch.cuda.mem_get_info()[0],
                create=cumem.create_and_map, release=cumem.unmap_and_release,
                copy=cumem.libcudart.cudaMemcpy,
            )

        # Keep the upstream worker's buffer restoration and any other wake hooks.
        allocator.wake_up = guarded_wake
        try:
            return super().wake_up(tags)
        finally:
            allocator.wake_up = original
