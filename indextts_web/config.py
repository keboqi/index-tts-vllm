"""Typed configuration without importing FastAPI, Torch, or model code."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

TTS_BACKENDS = ("confucius", "index", "index25")


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
    indextts25_repo_dir: str = "../index-tts-2.5-vllm-omni-experiment"
    indextts25_model_dir: str = ""
    indextts25_data_dir: str = ""
    indextts25_host: str = "127.0.0.1"
    indextts25_port: int = 8092
    indextts25_served_model_name: str = "IndexTeam/IndexTTS-2.5"
    indextts25_start_command: str = ""
    indextts25_start_shell: bool = True
    indextts25_detach_process: bool = True
    indextts25_attach_stdio: bool = False
    indextts25_log_dir: str = ""
    indextts25_start_timeout: float = 3600.0
    indextts25_request_timeout: float = 900.0
    indextts25_keepalive_interval: float = 60.0
    indextts25_unhealthy_grace: float = 30.0
    indextts25_max_parallel_segments: int = 100

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
    parser.add_argument(
        "--indextts25_repo_dir",
        default="../index-tts-2.5-vllm-omni-experiment",
    )
    parser.add_argument("--indextts25_model_dir", default=env.get("INDEXTTS25_MODEL_DIR", ""))
    parser.add_argument("--indextts25_data_dir", default=env.get("INDEXTTS25_DATA_DIR", ""))
    parser.add_argument("--indextts25_host", default=env.get("INDEXTTS25_HOST", "127.0.0.1"))
    parser.add_argument(
        "--indextts25_port",
        type=int,
        default=env_int(env, "INDEXTTS25_PORT", 8092, maximum=65535),
    )
    parser.add_argument(
        "--indextts25_served_model_name",
        default=env.get("INDEXTTS25_SERVED_MODEL_NAME", "IndexTeam/IndexTTS-2.5"),
    )
    parser.add_argument("--indextts25_start_command", default="")
    parser.add_argument(
        "--indextts25_start_shell",
        action=argparse.BooleanOptionalAction,
        default=env_flag(env, "INDEXTTS25_START_SHELL", True),
    )
    parser.add_argument(
        "--indextts25_detach_process",
        action=argparse.BooleanOptionalAction,
        default=env_flag(env, "INDEXTTS25_DETACH_PROCESS", True),
    )
    parser.add_argument(
        "--indextts25_attach_stdio",
        action=argparse.BooleanOptionalAction,
        default=env_flag(env, "INDEXTTS25_ATTACH_STDIO", False),
    )
    parser.add_argument("--indextts25_log_dir", default=env.get("INDEXTTS25_LOG_DIR", ""))
    parser.add_argument(
        "--indextts25_start_timeout",
        type=float,
        default=env_float(env, "INDEXTTS25_START_TIMEOUT", 3600.0, minimum=1.0),
    )
    parser.add_argument(
        "--indextts25_request_timeout",
        type=float,
        default=env_float(env, "INDEXTTS25_REQUEST_TIMEOUT", 900.0, minimum=1.0),
    )
    parser.add_argument(
        "--indextts25_keepalive_interval",
        type=float,
        default=env_float(env, "INDEXTTS25_KEEPALIVE_INTERVAL", 60.0, minimum=0.0),
    )
    parser.add_argument(
        "--indextts25_unhealthy_grace",
        type=float,
        default=env_float(env, "INDEXTTS25_UNHEALTHY_GRACE", 30.0, minimum=1.0),
    )
    parser.add_argument(
        "--indextts25_max_parallel_segments",
        type=int,
        default=env_int(env, "INDEXTTS25_MAX_PARALLEL_SEGMENTS", 100, maximum=256),
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
