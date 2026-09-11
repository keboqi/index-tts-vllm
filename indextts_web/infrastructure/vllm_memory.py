"""Recoverable wake for the CuMem allocator in the pinned vLLM 0.10.2."""

from __future__ import annotations


class GpuWakeError(RuntimeError):
    """Abort a synthesis batch when its engines cannot become ready."""


def wake_allocations(allocator, *, tags, free_bytes, create, release, copy, headroom=256 * 1024**2):
    """Retain CPU backups until every allocation has been restored successfully.

    vLLM 0.10.2 drops backups allocation by allocation and does not roll back
    earlier allocations if a later cuMemCreate runs out of memory. A retry can
    then remap live allocations. This function leaves the pool asleep on failure.
    """
    allocations = [(ptr, data) for ptr, data in allocator.pointer_to_data.items()
                   if tags is None or data.tag in tags]
    required = sum(data.handle[1] for _, data in allocations)
    available = free_bytes()
    if required and available < required + headroom:
        raise RuntimeError(
            f"Insufficient VRAM to wake vLLM: need {(required + headroom) / 1024**3:.2f} GiB "
            f"including headroom, have {available / 1024**3:.2f} GiB. "
            "Unload other models and retry; the engine remains safely asleep."
        )
    mapped = []
    try:
        for _, data in allocations:
            create(data.handle)
            mapped.append(data.handle)
        for ptr, data in allocations:
            if data.cpu_backup_tensor is not None:
                backup = data.cpu_backup_tensor
                copy(ptr, backup.data_ptr(), backup.numel() * backup.element_size())
    except Exception:
        for handle in reversed(mapped):
            release(handle)
        raise
    for _, data in allocations:
        data.cpu_backup_tensor = None
