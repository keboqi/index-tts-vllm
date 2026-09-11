"""Run Qwen3-ASR in its own Python environment and release it after each job."""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

from ...infrastructure.files import atomic_write_json
from ...infrastructure.gpu_work import GpuWorkCoordinator

PYTHON_ENV = "QWEN_OMNIVAD_PYTHON"
TIMEOUT_ENV = "QWEN_OMNIVAD_WORKER_TIMEOUT"


class QwenOmniVadWorker:
    def __init__(self, *, app_dir: Path, coordinator: GpuWorkCoordinator | None = None,
                 prepare_gpu=None, executor=None) -> None:
        self.app_dir = app_dir.resolve()
        self.coordinator = coordinator or GpuWorkCoordinator()
        self.prepare_gpu = prepare_gpu
        self.executor = executor
        self._lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(os.environ.get(PYTHON_ENV, "").strip())

    def _python(self) -> str | None:
        value = os.environ.get(PYTHON_ENV, "").strip()
        return shutil.which(value) if value else None

    def is_available(self, *, local_available: bool = False) -> bool:
        return self._python() is not None if self.configured else local_available

    async def translate(self, audio_bytes: bytes, *, local_pipeline=None, **options: Any):
        if not self.configured:
            if local_pipeline is None:
                raise RuntimeError(f"Qwen3-ASR is unavailable; configure {PYTHON_ENV} with its Python interpreter")
            return await asyncio.get_running_loop().run_in_executor(
                self.executor, functools.partial(local_pipeline, audio_bytes, **options),
            )

        python = self._python()
        if python is None:
            raise RuntimeError(f"{PYTHON_ENV} does not point to an executable Python; rebuild/redeploy the ASR image")
        timeout = float(os.environ.get(TIMEOUT_ENV, "7200"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"{TIMEOUT_ENV} must be a positive number of seconds")

        # The process owns all ASR/diarization/translation models for this job.
        # A cancelled or timed-out process must exit before TTS can wake.
        async with self._lock, self.coordinator.use("qwen_asr"):
            if self.coordinator.enabled and self.prepare_gpu is not None:
                await self.prepare_gpu()
            with tempfile.TemporaryDirectory(prefix="qwen-omnivad-job-") as directory:
                root = Path(directory)
                request_path, response_path = root / "request.json", root / "response.json"
                audio_path = root / "input.audio"
                await asyncio.to_thread(audio_path.write_bytes, audio_bytes)
                atomic_write_json(request_path, {"audio_path": str(audio_path),
                                                "response_path": str(response_path), "options": options})
                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(self.app_dir), env.get("PYTHONPATH", "")]))
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                kwargs = {"start_new_session": True} if os.name != "nt" else {
                    "creationflags": subprocess.CREATE_NO_WINDOW,
                }
                process = await asyncio.create_subprocess_exec(
                    python, "-u", "-m", "indextts_web.services.translation.qwen_worker",
                    "--request", str(request_path), cwd=str(self.app_dir), env=env, **kwargs,
                )
                try:
                    await asyncio.wait_for(process.wait(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(f"Qwen3-ASR worker exceeded {timeout:g} seconds; increase {TIMEOUT_ENV}") from exc
                finally:
                    if process.returncode is None:
                        await self._stop(process)

                if not response_path.is_file():
                    raise RuntimeError(f"Qwen3-ASR worker exited with code {process.returncode} without a result; see worker logs")
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if process.returncode or "error" in response:
                    raise RuntimeError(f"Qwen3-ASR worker failed: {response.get('error', process.returncode)}")
                result = response.get("result")
                if (not isinstance(result, list) or len(result) != 4
                        or not isinstance(result[0], list) or not isinstance(result[1], list)
                        or not isinstance(result[2], str) or not isinstance(result[3], dict)):
                    raise RuntimeError("Qwen3-ASR worker returned an invalid pipeline result")
                return tuple(result)

    @staticmethod
    async def _stop(process) -> None:
        def send(force: bool) -> None:
            try:
                if os.name == "nt":
                    process.kill() if force else process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass

        send(False)
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            send(True)
            await process.wait()


def run_job(request_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response_path = Path(request["response_path"])
    try:
        # This import deliberately happens only inside the ASR interpreter.
        from qwen_omnivad_pipeline import translate_audio

        result = translate_audio(Path(request["audio_path"]).read_bytes(), **request["options"])
        atomic_write_json(response_path, {"result": result})
    except Exception as exc:
        traceback.print_exc()
        atomic_write_json(response_path, {"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_job(args.request))


if __name__ == "__main__":
    main()
