"""Probe CUDA in a short-lived child, avoiding CUDA initialization before fork."""

from __future__ import annotations

import json
import subprocess
import sys

from ..gpu_profiles import GpuInfo

_PROBE = """
import json, sys, torch
device = sys.argv[1]
if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU is available')
p = torch.cuda.get_device_properties(device)
free, total = torch.cuda.mem_get_info(device)
print('INDEXTTS_GPU=' + json.dumps(dict(name=p.name, total_bytes=total,
    free_bytes=free, capability=f'{p.major}.{p.minor}', device=device)))
"""


def probe_gpu(*, device: str = "cuda:0") -> GpuInfo:
    if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError(f"vLLM requires a CUDA device, got {device!r}")
    device = "cuda:0" if device == "cuda" else device
    try:
        result = subprocess.run([sys.executable, "-c", _PROBE, device], capture_output=True,
                                text=True, timeout=60, check=True)
        line = next(line for line in reversed(result.stdout.splitlines()) if line.startswith("INDEXTTS_GPU="))
        return GpuInfo(**json.loads(line.split("=", 1)[1]))
    except (OSError, subprocess.SubprocessError, StopIteration, ValueError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"Cannot detect CUDA VRAM for automatic startup tuning: {detail.strip()}") from exc
