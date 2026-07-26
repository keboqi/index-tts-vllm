"""Local pure-Python MOSS transcription service for Modal."""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Any, Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_PATH = os.getenv(
    "MOSS_TRANSCRIBE_MODEL",
    "OpenMOSS-Team/MOSS-Transcribe-Diarize",
)
DEVICE_NAME = os.getenv("MOSS_TRANSCRIBE_DEVICE", "auto")

app = FastAPI(title="MOSS-Transcribe-Diarize local service")
_load_lock = threading.Lock()
_inference_lock = threading.Lock()
_runtime: Optional[tuple[Any, Any, Any, Any]] = None


def _get_runtime() -> tuple[Any, Any, Any, Any]:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _load_lock:
        if _runtime is not None:
            return _runtime
        device = resolve_device(DEVICE_NAME)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        print(f"[MOSS server] Loading {MODEL_PATH} on {device} ({dtype}).")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            dtype="auto",
        ).to(dtype=dtype).to(device).eval()
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )
        _runtime = model, processor, device, dtype
        return _runtime


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
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio_bytes)
            temp_path = handle.name
        model, processor, device, dtype = _get_runtime()
        messages = build_transcription_messages(temp_path, prompt=prompt)
        with _inference_lock:
            result = generate_transcription(
                model,
                processor,
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                device=device,
                dtype=dtype,
            )
        return result if isinstance(result, dict) else {"text": str(result)}
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
