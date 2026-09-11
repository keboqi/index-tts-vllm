"""Local pure-Python MOSS transcription service for Modal."""

from __future__ import annotations

import gc
import os
import tempfile
from typing import Any, Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)
from transformers import AutoModelForCausalLM, AutoProcessor

from indextts_web.services.translation.moss_runtime import MossRuntime

MODEL_PATH = os.getenv(
    "MOSS_TRANSCRIBE_MODEL",
    "OpenMOSS-Team/MOSS-Transcribe-Diarize",
)
DEVICE_NAME = os.getenv("MOSS_TRANSCRIBE_DEVICE", "auto")

app = FastAPI(title="MOSS-Transcribe-Diarize local service")


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    device = resolve_device(DEVICE_NAME)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[MOSS server] Loading {MODEL_PATH} on {device} ({dtype}).")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor, device, dtype


def _release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


runtime = MossRuntime(loader=_load_runtime, release_cache=_release_cuda_cache)


@app.get("/model/status")
async def model_status() -> dict[str, Any]:
    return {"service": "moss-transcribe", "model": MODEL_PATH, **runtime.status()}


@app.post("/model/sleep")
async def sleep_model() -> dict[str, Any]:
    await run_in_threadpool(runtime.change, "sleep")
    return await model_status()


@app.post("/model/wake")
async def wake_model() -> dict[str, Any]:
    await run_in_threadpool(runtime.change, "wake")
    return await model_status()


@app.post("/model/unload")
async def unload_model() -> dict[str, Any]:
    await run_in_threadpool(runtime.change, "unload")
    return await model_status()


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_PATH, "object": "model"}]}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    max_new_tokens: Optional[int] = Form(None),
) -> dict[str, Any]:
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"text": ""}

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    return await run_in_threadpool(_transcribe_audio, audio_bytes, suffix, prompt, max_new_tokens)


def _transcribe_audio(audio_bytes: bytes, suffix: str, prompt: str, max_new_tokens: Optional[int]) -> dict[str, Any]:
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio_bytes)
            temp_path = handle.name
        messages = build_transcription_messages(temp_path, prompt=prompt)

        def infer(model, processor, device, dtype):
            return generate_transcription(
                model,
                processor,
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                device=device,
                dtype=dtype,
            )
        result = runtime.generate(infer)
        return result if isinstance(result, dict) else {"text": str(result)}
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
