"""VRAM-aware startup policy. Importing this module never initializes CUDA."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

GIB = 1024**3
PROFILE_ENV = "INDEXTTS_GPU_PROFILE_JSON"
PROFILE_VERSION = 1
ENGINE_PREFIXES = {"index": "INDEXTTS_VLLM", "emotion": "QWENEMO_VLLM", "confucius": "CONFUCIUS_VLLM"}
MEMORY_ENV = {
    "index": "GPU_MEMORY_UTILIZATION",
    "emotion": "QWENEMO_GPU_MEMORY_UTILIZATION",
    "confucius": "CONFUCIUS_VLLM_GPU_MEMORY_UTILIZATION",
}


def positive_int(value: Any) -> int:
    result = int(value)
    if result < 1:
        raise ValueError("must be a positive integer")
    return result


def memory_fraction(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 < result < 1:
        raise ValueError("GPU memory utilization must be finite and between 0 and 1 (exclusive)")
    return result


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


@dataclass(frozen=True)
class GpuInfo:
    name: str
    total_bytes: int
    free_bytes: int
    capability: str
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if self.total_bytes <= 0 or not 0 <= self.free_bytes <= self.total_bytes:
            raise ValueError("Invalid GPU memory report")

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB


@dataclass(frozen=True)
class EngineProfile:
    gpu_memory_utilization: float
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None
    enforce_eager: bool | None = None

    def __post_init__(self) -> None:
        memory_fraction(self.gpu_memory_utilization)
        for value in (self.max_num_seqs, self.max_num_batched_tokens, self.max_model_len):
            if value is not None:
                positive_int(value)
        if self.enforce_eager is not None and not isinstance(self.enforce_eager, bool):
            raise ValueError("enforce_eager must be boolean")

    def kwargs(self, model_dir: str | Path | None = None) -> dict[str, Any]:
        values = {key: value for key, value in asdict(self).items() if value is not None}
        # Keep the checkpoint's context capacity. A small batched-token budget
        # must also work when chunked prefill is disabled by a custom model.
        if model_dir is not None and self.max_num_batched_tokens is not None:
            config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
            context = self.max_model_len or max(
                int(config.get("max_position_embeddings") or 0), int(config.get("n_positions") or 0)
            )
            values["max_num_batched_tokens"] = max(self.max_num_batched_tokens, context)
        return values


@dataclass(frozen=True)
class GpuProfile:
    name: str
    gpu: GpuInfo
    index: EngineProfile
    emotion: EngineProfile
    confucius: EngineProfile
    use_torch_compile: bool
    index_concurrency: int
    translation_concurrency: int
    parallel_segments: int
    conditioning_cache_size: int
    omni_ar_seqs: int
    omni_s2mel_seqs: int
    omni_ar_memory: float
    omni_s2mel_memory: float
    runtime_identity: str = ""

    def __post_init__(self) -> None:
        for field in ("index_concurrency", "translation_concurrency", "parallel_segments",
                      "conditioning_cache_size", "omni_ar_seqs", "omni_s2mel_seqs"):
            positive_int(getattr(self, field))
        memory_fraction(self.omni_ar_memory)
        memory_fraction(self.omni_s2mel_memory)

    def to_json(self) -> str:
        return json.dumps({"version": PROFILE_VERSION, **asdict(self)}, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> GpuProfile:
        values = json.loads(raw)
        if values.pop("version", None) != PROFILE_VERSION:
            raise ValueError("Incompatible GPU profile schema; redeploy the matching application code")
        values["gpu"] = GpuInfo(**values["gpu"])
        for key in ENGINE_PREFIXES:
            values[key] = EngineProfile(**values[key])
        return cls(**values)

    @property
    def cache_key(self) -> str:
        values = json.loads(self.to_json())
        # Free memory varies across cold starts/restores. It is not part of
        # engine configuration or compilation compatibility.
        values["gpu"].pop("free_bytes")
        return f"{self.name}-sm{self.gpu.capability.replace('.', '')}-" + hashlib.sha256(
            json.dumps(values, sort_keys=True).encode()
        ).hexdigest()[:12]

    def with_settings(self, settings: Any) -> GpuProfile:
        updates: dict[str, Any] = {}
        for engine, attr in (("index", "gpu_memory_utilization"), ("emotion", "qwenemo_gpu_memory_utilization"),
                             ("confucius", "confucius_vllm_gpu_memory_utilization")):
            value = getattr(settings, attr)
            if value is not None:
                updates[engine] = replace(getattr(self, engine), gpu_memory_utilization=memory_fraction(value))
        if settings.use_torch_compile is not None:
            updates["use_torch_compile"] = settings.use_torch_compile
        if settings.indextts25_max_parallel_segments is not None:
            updates["parallel_segments"] = positive_int(settings.indextts25_max_parallel_segments)
        return replace(self, **updates)

    def apply_settings(self, settings: Any) -> Any:
        return replace(settings, gpu_memory_utilization=self.index.gpu_memory_utilization,
                       qwenemo_gpu_memory_utilization=self.emotion.gpu_memory_utilization,
                       confucius_vllm_gpu_memory_utilization=self.confucius.gpu_memory_utilization,
                       use_torch_compile=self.use_torch_compile,
                       indextts25_max_parallel_segments=self.parallel_segments)

    def check_startup_memory(self, *, non_vllm_gib: float = 8.0) -> None:
        """Preflight for the initial IndexTTS load, never for an already loaded restore.

        The non-vLLM reserve is a conservative planning allowance, not a measured
        hard allocation limit. Real workload validation is still required.
        """
        if not math.isfinite(non_vllm_gib) or non_vllm_gib < 0:
            raise ValueError("INDEXTTS_NON_VLLM_RESERVE_GIB must be finite and nonnegative")
        budget = (self.index.gpu_memory_utilization + self.emotion.gpu_memory_utilization) * self.gpu.total_gib
        required = budget + non_vllm_gib + max(2.0, self.gpu.total_gib * 0.1)
        if required > self.gpu.free_bytes / GIB:
            raise RuntimeError(
                f"GPU profile {self.name} needs approximately {required:.2f} GiB for initial TTS load "
                f"(engines {budget:.2f}, non-vLLM reserve {non_vllm_gib:.2f}, plus headroom); "
                f"only {self.gpu.free_bytes / GIB:.2f}/{self.gpu.total_gib:.2f} GiB is free. "
                "Free other GPU allocations or explicitly adjust the engine budgets/reserve."
            )


def resolve_gpu_profile(gpu: GpuInfo, environ: Mapping[str, str] | None = None, *, modal: bool = False) -> GpuProfile:
    env = os.environ if environ is None else environ
    total = gpu.total_gib
    if total < 20:
        raise ValueError(f"Detected {total:.2f} GiB VRAM; automatic profiles require a 24 GB-class GPU or larger")
    if total < 32:
        profile = GpuProfile("24gb", gpu, EngineProfile(6 / total, 4, 2560, enforce_eager=True),
                             EngineProfile(3 / total, 1, 2048, 2048, True),
                             EngineProfile(6 / total, 4, enforce_eager=True), False,
                             1, 1, 1, 2, 4, 1, 4 / total, 7 / total)
    elif total < 64:
        profile = GpuProfile("48gb", gpu, EngineProfile(10 / total, 16, 4096),
                             EngineProfile(4 / total, 4, 2048, 2048),
                             EngineProfile(10 / total, 16), False,
                             4, 4, 4, 4, 16, 4, 8 / total, 14 / total)
    else:
        profile = GpuProfile("96gb", gpu, EngineProfile(0.15), EngineProfile(0.05, max_model_len=2048),
                             EngineProfile(0.20 if modal else 0.15), modal,
                             100, 100, 100, 8, 32, 16, 0.3, 0.3)
    updates: dict[str, Any] = {}
    for engine, prefix in ENGINE_PREFIXES.items():
        values = asdict(getattr(profile, engine))
        for key, convert in (("gpu_memory_utilization", memory_fraction), ("max_num_seqs", positive_int),
                             ("max_num_batched_tokens", positive_int), ("max_model_len", positive_int),
                             ("enforce_eager", boolean)):
            variable = MEMORY_ENV[engine] if key == "gpu_memory_utilization" else f"{prefix}_{key.upper()}"
            if env.get(variable, "").strip():
                try:
                    values[key] = convert(env[variable])
                except ValueError as exc:
                    raise ValueError(f"{variable}: {exc}") from exc
        updates[engine] = EngineProfile(**values)
    for field, variable in (("index_concurrency", "INDEXTTS_GPU_WORK_CONCURRENCY"),
                            ("translation_concurrency", "TRANSLATION_TTS_CONCURRENCY"),
                            ("parallel_segments", "INDEXTTS25_MAX_PARALLEL_SEGMENTS"),
                            ("conditioning_cache_size", "INDEXTTS_CONDITIONING_CACHE_SIZE")):
        if env.get(variable, "").strip():
            updates[field] = positive_int(env[variable])
    if env.get("INDEXTTS_USE_TORCH_COMPILE", "").strip():
        updates["use_torch_compile"] = boolean(env["INDEXTTS_USE_TORCH_COMPILE"])
    profile = replace(profile, runtime_identity=env.get("MODAL_IMAGE_ID", ""), **updates)
    return replace(profile, translation_concurrency=min(profile.translation_concurrency, profile.index_concurrency))


def runtime_gpu_profile(*, device: str = "cuda:0", modal: bool = False) -> GpuProfile:
    """Consume the Modal parent's resolved profile, or probe during local startup."""
    device = "cuda:0" if device == "cuda" else device
    if os.environ.get(PROFILE_ENV):
        profile = GpuProfile.from_json(os.environ[PROFILE_ENV])
        if profile.gpu.device != device:
            raise ValueError(f"GPU profile targets {profile.gpu.device}, but the model targets {device}")
        return profile
    from .infrastructure.gpu import probe_gpu

    return resolve_gpu_profile(probe_gpu(device=device), modal=modal)


def write_omni_deploy_config(base_path: Path, output_dir: Path, profile: GpuProfile) -> Path:
    """Derive stage limits without changing checkpoint/context/sampling semantics."""
    import yaml

    if profile.name == "96gb":
        return base_path
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    stages = {stage["stage_id"]: stage for stage in config["stages"]}
    if set(stages) != {0, 1}:
        raise ValueError("Unsupported IndexTTS 2.5 deployment schema; expected stages 0 and 1")
    stages[0].update(max_num_seqs=profile.omni_ar_seqs, gpu_memory_utilization=profile.omni_ar_memory)
    stages[1].update(max_num_seqs=profile.omni_s2mel_seqs, gpu_memory_utilization=profile.omni_s2mel_memory)
    stages[1].setdefault("hf_overrides", {}).update(
        s2mel_cfm_batch_size=profile.omni_s2mel_seqs,
        s2mel_dit_torch_compile=False, s2mel_vocoder_torch_compile=False,
    )
    if profile.name == "24gb":
        stages[0]["enforce_eager"] = True
    content = yaml.safe_dump(config, sort_keys=False)
    digest = hashlib.sha256(content.encode()).hexdigest()[:12]
    target = output_dir / f"indextts25-{profile.cache_key}-{digest}.yaml"
    from .infrastructure.files import atomic_write_json

    # JSON is valid YAML, and the shared helper provides an atomic replacement.
    atomic_write_json(target, config)
    return target


def main() -> None:
    """Inspect the allocated GPU and resolved options without loading weights."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modal", action="store_true", help="Use the Modal defaults for the 96 GB profile")
    args = parser.parse_args()
    profile = runtime_gpu_profile(device=args.device, modal=args.modal)
    print(json.dumps(json.loads(profile.to_json()), indent=2))


if __name__ == "__main__":
    main()
