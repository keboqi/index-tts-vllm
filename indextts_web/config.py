"""Typed configuration without importing FastAPI, Torch, or model code."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

TTS_BACKENDS = ("confucius", "higgs", "index")


def env_flag(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    try:
        value = int(environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    return min(maximum, value) if maximum is not None else value


def env_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


@dataclass(frozen=True, slots=True)
class AppSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    model_dir: str = "checkpoints"
    verbose: bool = False
    is_fp16: bool = False
    use_torch_compile: bool = False
    gpu_memory_utilization: float = 0.15
    qwenemo_gpu_memory_utilization: float = 0.05
    tts_backend: str = "index"
    confucius_repo_dir: str = "../Confucius4-TTS"
    confucius_host: str = "127.0.0.1"
    confucius_port: int = 8001
    confucius_start_command: str = ""
    confucius_start_shell: bool = True
    confucius_detach_process: bool = True
    confucius_attach_stdio: bool = False
    confucius_log_dir: str = ""
    confucius_start_timeout: float = 1800.0
    confucius_request_timeout: float = 900.0
    confucius_keepalive_interval: float = 60.0
    confucius_unhealthy_grace: float = 30.0
    confucius_vllm_gpu_memory_utilization: float = 0.15
    higgs_server_url: str = "http://127.0.0.1:8002"
    higgs_model: str = "bosonai/higgs-audio-v3-tts-4b"
    higgs_manage_backend: bool = True
    higgs_manager_script: str = "sglang_omni_higgs.sh"
    higgs_start_timeout: float = 3600.0
    higgs_request_timeout: float = 1800.0
    higgs_mem_fraction_static: float = 0.30
    higgs_max_running_requests: int = 100
    higgs_dtype: str = "bfloat16"
    higgs_initial_codec_chunk_frames: int = 1
    higgs_max_new_tokens: int = 4096
    higgs_temperature: float = 0.8
    higgs_top_k: int = 50
    higgs_top_p: float = 0.0

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> AppSettings:
        values = vars(namespace)
        supported = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in supported})


def build_parser(environ: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description="IndexTTS vLLM v2 FastAPI WebUI")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_dir", default="checkpoints")
    parser.add_argument("--is_fp16", action="store_true", default=False)
    parser.add_argument("--use_torch_compile", action="store_true", default=False)
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=env_float(env, "GPU_MEMORY_UTILIZATION", 0.15, minimum=0.0, maximum=1.0),
    )
    parser.add_argument(
        "--qwenemo_gpu_memory_utilization",
        type=float,
        default=env_float(env, "QWENEMO_GPU_MEMORY_UTILIZATION", 0.05, minimum=0.0, maximum=1.0),
    )
    parser.add_argument("--tts_backend", choices=TTS_BACKENDS, default="index")
    parser.add_argument("--confucius_repo_dir", default="../Confucius4-TTS")
    parser.add_argument("--confucius_host", default="127.0.0.1")
    parser.add_argument("--confucius_port", type=int, default=8001)
    parser.add_argument("--confucius_start_command", default="")
    parser.add_argument(
        "--confucius_start_shell",
        action=argparse.BooleanOptionalAction,
        default=env_flag(env, "CONFUCIUS_START_SHELL", True),
    )
    parser.add_argument(
        "--confucius_detach_process",
        action=argparse.BooleanOptionalAction,
        default=env_flag(env, "CONFUCIUS_DETACH_PROCESS", True),
    )
    parser.add_argument(
        "--confucius_attach_stdio",
        action=argparse.BooleanOptionalAction,
        default=env_flag(env, "CONFUCIUS_ATTACH_STDIO", False),
    )
    parser.add_argument("--confucius_log_dir", default=env.get("CONFUCIUS_LOG_DIR", ""))
    parser.add_argument(
        "--confucius_start_timeout",
        type=float,
        default=env_float(env, "CONFUCIUS_START_TIMEOUT", 1800.0, minimum=1.0),
    )
    parser.add_argument(
        "--confucius_request_timeout",
        type=float,
        default=env_float(env, "CONFUCIUS_REQUEST_TIMEOUT", 900.0, minimum=1.0),
    )
    parser.add_argument(
        "--confucius_keepalive_interval",
        type=float,
        default=env_float(env, "CONFUCIUS_KEEPALIVE_INTERVAL", 60.0, minimum=0.0),
    )
    parser.add_argument(
        "--confucius_unhealthy_grace",
        type=float,
        default=env_float(env, "CONFUCIUS_UNHEALTHY_GRACE", 30.0, minimum=1.0),
    )
    parser.add_argument(
        "--confucius_vllm_gpu_memory_utilization",
        type=float,
        default=env_float(
            env,
            "CONFUCIUS_VLLM_GPU_MEMORY_UTILIZATION",
            0.15,
            minimum=0.0,
            maximum=1.0,
        ),
    )
    parser.add_argument("--higgs_server_url", default=env.get("HIGGS_TTS_SGLANG_URL", "http://127.0.0.1:8002"))
    parser.add_argument("--higgs_model", default=env.get("HIGGS_TTS_MODEL", "bosonai/higgs-audio-v3-tts-4b"))
    parser.add_argument(
        "--higgs_manage_backend",
        action=argparse.BooleanOptionalAction,
        default=env_flag(env, "HIGGS_TTS_MANAGE_BACKEND", True),
    )
    parser.add_argument("--higgs_manager_script", default=env.get("HIGGS_TTS_MANAGER_SCRIPT", "sglang_omni_higgs.sh"))
    parser.add_argument(
        "--higgs_start_timeout",
        type=float,
        default=env_float(env, "HIGGS_TTS_START_TIMEOUT", 3600.0, minimum=1.0),
    )
    parser.add_argument(
        "--higgs_request_timeout",
        type=float,
        default=env_float(env, "HIGGS_TTS_REQUEST_TIMEOUT", 1800.0, minimum=1.0),
    )
    parser.add_argument(
        "--higgs_mem_fraction_static",
        type=float,
        default=env_float(env, "HIGGS_TTS_MEM_FRACTION_STATIC", 0.30, minimum=0.01, maximum=1.0),
    )
    parser.add_argument(
        "--higgs_max_running_requests",
        type=int,
        default=env_int(env, "HIGGS_TTS_MAX_RUNNING_REQUESTS", 100, minimum=1, maximum=256),
    )
    parser.add_argument("--higgs_dtype", default=env.get("HIGGS_TTS_DTYPE", "bfloat16"))
    parser.add_argument(
        "--higgs_initial_codec_chunk_frames",
        type=int,
        default=env_int(env, "HIGGS_TTS_INITIAL_CODEC_CHUNK_FRAMES", 1, minimum=0, maximum=75),
    )
    parser.add_argument(
        "--higgs_max_new_tokens",
        type=int,
        default=env_int(env, "HIGGS_TTS_MAX_NEW_TOKENS", 4096, minimum=1),
    )
    parser.add_argument(
        "--higgs_temperature",
        type=float,
        default=env_float(env, "HIGGS_TTS_TEMPERATURE", 0.8, minimum=0.0),
    )
    parser.add_argument(
        "--higgs_top_k",
        type=int,
        default=env_int(env, "HIGGS_TTS_TOP_K", 50, minimum=0),
    )
    parser.add_argument(
        "--higgs_top_p",
        type=float,
        default=env_float(env, "HIGGS_TTS_TOP_P", 0.0, minimum=0.0, maximum=1.0),
    )
    return parser


def load_settings(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    allow_unknown: bool = False,
) -> AppSettings:
    parser = build_parser(environ)
    if allow_unknown:
        namespace, _unknown = parser.parse_known_args(argv)
    else:
        namespace = parser.parse_args(argv)
    return AppSettings.from_namespace(namespace)
