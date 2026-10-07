"""Run ClearVoice in a separate environment, releasing its models after each job."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import traceback
import uuid
from pathlib import Path

from ...infrastructure.files import atomic_write_json

PYTHON_ENV = "CLEARVOICE_PYTHON"
TIMEOUT_ENV = "CLEARVOICE_WORKER_TIMEOUT"
APP_DIR = Path(__file__).resolve().parents[3]


def is_configured() -> bool:
    return bool(os.environ.get(PYTHON_ENV, "").strip())


def _python() -> str | None:
    value = os.environ.get(PYTHON_ENV, "").strip()
    return shutil.which(value) if value else None


def is_available(*, local_available: bool = False) -> bool:
    return _python() is not None if is_configured() else local_available


def _outputs(input_path: str, enhancement: bool, super_resolution: bool) -> list[str]:
    paths = []
    current = Path(input_path)
    for enabled, suffix in ((enhancement, "_se"), (super_resolution, "_sr")):
        if enabled:
            current = current.with_name(f"{current.stem}{suffix}{current.suffix}")
            paths.append(str(current))
    return paths


def _cleanup(paths: list[str]) -> None:
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def process(input_path: str, apply_enhancement: bool, apply_super_resolution: bool,
            *, enhancement_model_name: str, app_dir: Path = APP_DIR):
    """Launch one bounded job; callers retain control of chunk concurrency."""
    if not (apply_enhancement or apply_super_resolution):
        return input_path, [], None
    python = _python()
    if python is None:
        raise RuntimeError(f"{PYTHON_ENV} does not point to an executable Python interpreter")
    timeout = float(os.environ.get(TIMEOUT_ENV, "7200"))
    if not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise ValueError(f"{TIMEOUT_ENV} must be between 0 and 86400 seconds")
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ClearVoice input is missing: {source}")
    # Each job owns its outputs even when several jobs process the same input.
    output_base = source.with_name(f"{source.stem[:100]}_cv_{uuid.uuid4().hex}{source.suffix}")
    generated = _outputs(str(output_base), apply_enhancement, apply_super_resolution)
    enhancement_path = generated[0] if apply_enhancement else None
    try:
        with tempfile.TemporaryDirectory(prefix="clearvoice-job-") as directory:
            request_path = Path(directory) / "request.json"
            response_path = Path(directory) / "response.json"
            atomic_write_json(request_path, {
                "input_path": str(source), "output_base": str(output_base),
                "response_path": str(response_path),
                "apply_enhancement": apply_enhancement,
                "apply_super_resolution": apply_super_resolution,
                "enhancement_model_name": enhancement_model_name,
            })
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(app_dir.resolve()), env.get("PYTHONPATH", "")]))
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            # subprocess.run kills and reaps the child on TimeoutExpired.
            completed = subprocess.run(
                [python, "-u", "-m", "indextts_web.services.audio.clearvoice_worker",
                 "--request", str(request_path)],
                cwd=str(app_dir.resolve()), env=env, timeout=timeout, check=False, **kwargs,
            )
            if not response_path.is_file():
                raise RuntimeError(f"ClearVoice worker exited with code {completed.returncode} without a result")
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if not isinstance(response, dict):
                raise RuntimeError("ClearVoice worker returned an invalid result")
            if completed.returncode or "error" in response:
                raise RuntimeError(f"ClearVoice worker failed: {response.get('error', completed.returncode)}")
            if (response.get("generated_paths") != generated
                    or response.get("final_path") != generated[-1]
                    or response.get("enhancement_path") != enhancement_path
                    or not all(Path(path).is_file() for path in generated)):
                raise RuntimeError("ClearVoice worker returned an invalid result")
            return response["final_path"], generated, enhancement_path
    except subprocess.TimeoutExpired as exc:
        _cleanup(generated)
        raise RuntimeError(f"ClearVoice worker exceeded {timeout:g} seconds; increase {TIMEOUT_ENV}") from exc
    except Exception:
        _cleanup(generated)
        raise


def run_job(request_path: Path, *, clearvoice_factory=None) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response_path = Path(request["response_path"])
    source = request["input_path"]
    enhancement = request["apply_enhancement"]
    super_resolution = request["apply_super_resolution"]
    generated = _outputs(request["output_base"], enhancement, super_resolution)
    try:
        if clearvoice_factory is None:
            # ClearVoice 0.1.2 exposes this API; never import the web application.
            from clearvoice import ClearVoice

            clearvoice_factory = ClearVoice
        current = source
        enhancement_path = None
        tasks = ((enhancement, "speech_enhancement", request["enhancement_model_name"]),
                 (super_resolution, "speech_super_resolution", "MossFormer2_SR_48K"))
        outputs = iter(generated)
        for enabled, task, model_name in tasks:
            if enabled:
                model = clearvoice_factory(task=task, model_names=[model_name])
                output = model(input_path=current, online_write=False)
                current = next(outputs)
                model.write(output, output_path=current)
                if task == "speech_enhancement":
                    enhancement_path = current
                del model
        atomic_write_json(response_path, {
            "final_path": current, "generated_paths": generated,
            "enhancement_path": enhancement_path,
        })
    except Exception as exc:
        _cleanup(generated)
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
