"""Lenient timestamp parsing for model output and SRT timestamp formatting."""

from __future__ import annotations

import re
from typing import Any


def parse_timestamp_ms(timestamp_value: Any) -> int | None:
    if timestamp_value is None:
        return None
    if isinstance(timestamp_value, int | float):
        # Treat integer millisecond values separately
        if isinstance(timestamp_value, int) and timestamp_value >= 1000:
            return max(0, int(timestamp_value))
        seconds = float(timestamp_value)
        return max(0, int(round(seconds * 1000)))
    value = str(timestamp_value).strip()
    if not value:
        return None
    value = value.replace(",", ".")
    # Try simple numeric parse
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        seconds = float(value)
        return max(0, int(seconds * 1000))
    # Handle colon-delimited formats (e.g., hh:mm:ss.xxx or mm:ss)
    parts = value.split(":")
    try:
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return None
        total_seconds = 0.0
        multiplier = 1.0
        for part in reversed(parts):
            if "." in part:
                seconds = float(part)
            else:
                seconds = float(int(part))
            total_seconds += seconds * multiplier
            multiplier *= 60.0
        return max(0, int(round(total_seconds * 1000)))
    except ValueError:
        return None


def coerce_segment_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        parsed = parse_timestamp_ms(stripped)
        if parsed is not None:
            return parsed
        try:
            numeric = float(stripped)
            return max(0, int(numeric))
        except ValueError:
            return None
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return None


def format_srt_timestamp(milliseconds: int) -> str:
    ms = max(0, int(milliseconds))
    hours = ms // 3_600_000
    minutes = (ms % 3_600_000) // 60_000
    seconds = (ms % 60_000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
