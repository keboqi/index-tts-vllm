"""Lightweight readiness endpoint used by containers and load balancers."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health", include_in_schema=True)
async def health(request: Request) -> JSONResponse:
    runtime = request.app.state.runtime
    manager = runtime.legacy.tts_manager
    ready = bool(manager.is_ready())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "healthy" if ready else "starting",
            "ready": ready,
            "timestamp": time.time(),
            "tts_backends": runtime.backends.names,
        },
    )

