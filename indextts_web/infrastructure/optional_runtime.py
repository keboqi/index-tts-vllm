"""Opt-in local environment preparation; never imported by Modal startup.

Colab interpreter wrappers call this on first use, or the notebook can call it
before launch. The application continues using its existing worker interfaces.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

SMOKE = {
    "clearvoice": "from clearvoice import ClearVoice; import importlib.metadata as m; assert m.version('rotary-embedding-torch') == '0.8.3'",
    "moss": "from moss_transcribe_server import app, runtime; assert runtime.status()['state'] == 'unloaded'",
    "qwen-asr": """
from qwen_asr import Qwen3ASRModel
from omnivad import OmniVAD
from nemo.collections.asr.models import SortformerEncLabelModel
import qwen_omnivad_pipeline as pipeline
assert pipeline.is_qwen_omnivad_available()
assert pipeline._make_translation_llm is not None
assert pipeline.DiarizationPipeline is not None
import importlib.metadata as metadata
assert metadata.version('transformers') == '4.57.6'
""",
}


def interpreter(root: Path, kind: str) -> Path:
    return root / ("venv_index_tts_" + kind.replace("-", "_")) / "bin/python"


@contextlib.contextmanager
def setup_lock(path: Path):
    # File locks release automatically after cancellation or a failed install.
    with path.open("a") as stream:
        if os.name == "posix":
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def prepare(kind: str, workspace: Path, root: Path, cache: Path, *, force=False, run=None) -> Path:
    """Install one isolated environment and publish readiness only after validation."""
    if kind not in SMOKE:
        raise ValueError(f"Unknown optional runtime: {kind}")
    run = run or (lambda command, **kwargs: subprocess.run(command, check=True, **kwargs))
    manifest = workspace / ("requirements-clearvoice.txt" if kind == "clearvoice"
                            else f"requirements-colab-{kind}.txt")
    constraints = "".join(line for line in (workspace / "constraints-main.txt").read_text().splitlines(keepends=True)
                          if not line.startswith("transformers==")) if kind == "qwen-asr" else ""
    fingerprint = hashlib.sha256(manifest.read_bytes() + constraints.encode() + Path(__file__).read_bytes()).hexdigest()
    python = interpreter(root, kind)
    marker = python.parent.parent / "index-tts-ready.txt"
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    with setup_lock(root / (".setup-" + kind + ".lock")):
        if not force and python.is_file() and marker.is_file() and marker.read_text() == fingerprint:
            return python
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv is unavailable. Rerun the notebook dependency setup cell.")
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "PIP_CONSTRAINT", "UV_CONSTRAINT", "MPLBACKEND"):
            env.pop(key, None)
        env.update(VIRTUAL_ENV=str(python.parent.parent), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
        marker.unlink(missing_ok=True)
        print(f"[{kind}] Preparing isolated environment (first use can take several minutes)...", flush=True)
        if not python.is_file():
            run([uv, "venv", "--python", "3.12", "--seed", str(python.parent.parent)], env=env)
        run([str(python), "-c", "import sys; assert sys.version_info[:2] == (3, 12)"], env=env)
        install = [uv, "pip", "install", "--python", str(python), "--torch-backend", "cu128"]
        if constraints:
            constraint_path = cache / "constraints-qwen-asr.txt"
            constraint_path.write_text(constraints, encoding="utf-8")
            install += ["-c", str(constraint_path)]
        run(install + ["-r", str(manifest)], cwd=workspace, env=env)
        run([str(python), "-m", "pip", "check"], env=env)
        smoke = """
import torch
assert torch.__version__.split('+')[0] == '2.8.0'
assert torch.version.cuda == '12.8'
assert torch.cuda.is_available(), 'CUDA unavailable in optional environment'
""" + SMOKE[kind]
        run([str(python), "-c", smoke], cwd=workspace, env=env)
        marker.write_text(fingerprint, encoding="utf-8")
        print(f"[{kind}] Environment ready; model weights load on first use.", flush=True)
    return python


def moss_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/model/status", timeout=1) as response:
            payload = json.load(response)
            return isinstance(payload, dict) and payload.get("service") == "moss-transcribe"
    except (OSError, ValueError):
        return False


def start_moss(python: Path, workspace: Path, *, timeout=120):
    """Start a local MOSS child in the calling app's process group for cleanup."""
    url = "http://127.0.0.1:8003"
    if moss_ready(url):
        return
    directory = workspace / ".colab"
    directory.mkdir(exist_ok=True)
    log_path = directory / "moss-managed.log"
    env = {**os.environ, "VIRTUAL_ENV": str(python.parent.parent), "PYTHONUNBUFFERED": "1"}
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(python), "-m", "uvicorn", "moss_transcribe_server:app",
                                    "--host", "127.0.0.1", "--port", "8003"],
                                   cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"MOSS exited with code {process.returncode}:\n{log_path.read_text()[-4000:]}")
            if moss_ready(url):
                print("MOSS local service ready; model weights remain unloaded.", flush=True)
                return
            time.sleep(0.5)
        raise TimeoutError(f"MOSS startup timed out. See {log_path}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=tuple(SMOKE))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start-moss", action="store_true")
    parser.add_argument("--exec-python", action="store_true")
    args, remaining = parser.parse_known_args()
    python = prepare(args.kind, args.workspace.resolve(), args.root.resolve(), args.cache.resolve(), force=args.force)
    if args.start_moss:
        if args.kind != "moss":
            parser.error("--start-moss requires the moss runtime")
        start_moss(python, args.workspace.resolve())
    if args.exec_python:
        remaining = remaining[1:] if remaining[:1] == ["--"] else remaining
        # Keep the application's worker PYTHONPATH, but select this environment
        # for any tools launched by the worker after exec.
        os.environ["VIRTUAL_ENV"] = str(python.parent.parent)
        os.environ["PATH"] = str(python.parent) + os.pathsep + os.environ.get("PATH", "")
        os.execv(str(python), [str(python), *remaining])


if __name__ == "__main__":
    main()
