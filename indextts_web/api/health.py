"""Lightweight readiness endpoint used by containers and load balancers."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from indextts_web.services.tts.startup import startup_backend

router = APIRouter()


@router.get("/health", include_in_schema=True)
async def health(request: Request) -> JSONResponse:
    runtime = request.app.state.runtime
    backend = startup_backend(runtime.settings)
    if backend == "none":
        ready = True  # Backends load on demand; the server itself is ready.
    elif backend == "index":
        ready = bool(runtime.legacy.tts_manager.is_ready())
    else:
        ready = bool((await runtime.backends.get(backend).status()).get("ready"))
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "healthy" if ready else "starting",
            "ready": ready,
            "startup_tts_backend": backend,
            "timestamp": time.time(),
            "tts_backends": runtime.backends.names,
        },
    )

