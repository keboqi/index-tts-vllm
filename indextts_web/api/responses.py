"""JSON error and server-sent-event response helpers shared by endpoints."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse


def status_error(message: str, status_code: int = 400, **extra: Any) -> JSONResponse:
    payload = {"status": "error", "message": message}
    payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def success_error(message: str, status_code: int = 500, **extra: Any) -> JSONResponse:
    payload = {"success": False, "error": message}
    payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def request_has_json_body(request: Request) -> bool:
    return "application/json" in request.headers.get("content-type", "").lower()


JSON_STREAM_MIN_CHUNK_BYTES = 4096


def json_event_bytes(event: dict[str, Any]) -> bytes:
    event_payload = event
    encoded = json.dumps(event_payload, ensure_ascii=False)
    if len(encoded.encode("utf-8")) < JSON_STREAM_MIN_CHUNK_BYTES:
        event_payload = dict(event)
        event_payload["_flush_padding"] = "0" * (
            JSON_STREAM_MIN_CHUNK_BYTES - len(encoded.encode("utf-8"))
        )
        encoded = json.dumps(event_payload, ensure_ascii=False)
    return (f"data: {encoded}\n\n").encode()


JSON_STREAM_RESPONSE_HEADERS = {
    "Cache-Control": "no-store, no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "X-Content-Type-Options": "nosniff",
}


async def json_stream_with_prelude(iterator: Any):
    yield json_event_bytes(
        {
            "event": "stream_open",
            "timestamp": time.time(),
        }
    )
    async for chunk in iterator:
        yield chunk


def json_stream_response(iterator: Any) -> StreamingResponse:
    return StreamingResponse(
        json_stream_with_prelude(iterator),
        media_type="text/event-stream",
        headers=JSON_STREAM_RESPONSE_HEADERS,
    )


async def read_optional_json_payload(
    request: Request,
    should_read: bool,
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    if not should_read:
        return None, None
    try:
        payload = await request.json()
    except Exception as exc:
        return None, status_error(f"Invalid JSON payload: {str(exc)}")
    if not isinstance(payload, dict):
        return None, status_error("Invalid JSON payload: expected a JSON object.")
    return payload, None
