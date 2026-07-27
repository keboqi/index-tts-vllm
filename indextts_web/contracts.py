"""Stable wire-format helpers shared by the server and clients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

STREAMING_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def audio_chunk_frame(chunk_index: int, audio: bytes, *, is_last: bool) -> bytes:
    state = "LAST" if is_last else "MORE"
    return f"CHUNK:{chunk_index}:{len(audio)}:{state}\n".encode() + audio


def keepalive_frame(message: str, *, elapsed_seconds: int | None = None) -> bytes:
    payload = json.dumps(
        {"message": message, "elapsed_seconds": elapsed_seconds},
        ensure_ascii=False,
    ).encode()
    return f"KEEPALIVE:{len(payload)}\n".encode() + payload


@dataclass(frozen=True, slots=True)
class ParsedStreamFrame:
    kind: str
    payload: bytes
    chunk_index: int | None = None
    is_last: bool = False


def parse_stream_frame(frame: bytes) -> ParsedStreamFrame:
    header, separator, payload = frame.partition(b"\n")
    if not separator:
        raise ValueError("stream frame has no header terminator")
    text = header.decode("utf-8")
    if text.startswith("CHUNK:"):
        parts = text.split(":")
        if len(parts) != 4:
            raise ValueError("invalid CHUNK header")
        index, length, state = int(parts[1]), int(parts[2]), parts[3]
        if length != len(payload) or state not in {"MORE", "LAST"}:
            raise ValueError("invalid CHUNK payload metadata")
        return ParsedStreamFrame("chunk", payload, index, state == "LAST")
    if text.startswith("KEEPALIVE:"):
        length = int(text.split(":", 1)[1])
        if length != len(payload):
            raise ValueError("invalid KEEPALIVE payload metadata")
        return ParsedStreamFrame("keepalive", payload)
    raise ValueError(f"unknown stream frame: {text!r}")


def decode_keepalive(frame: bytes) -> dict[str, Any]:
    parsed = parse_stream_frame(frame)
    if parsed.kind != "keepalive":
        raise ValueError("frame is not a keepalive")
    value = json.loads(parsed.payload)
    if not isinstance(value, dict):
        raise ValueError("keepalive payload is not an object")
    return value

