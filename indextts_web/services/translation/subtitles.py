"""Pure SRT/WebVTT parsing and translation-segment conversion."""

from __future__ import annotations

from typing import Any

SINGLE_SPEAKER_PROFILE = [
    {
        "id": "speaker1",
        "description": "Single speaker (from subtitle)",
    }
]


def format_timestamp(milliseconds: int) -> str:
    milliseconds = max(0, int(milliseconds))
    total_seconds = milliseconds / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    if abs(seconds - round(seconds)) < 1e-3:
        seconds_text = f"{int(round(seconds)):02d}"
    else:
        seconds_text = f"{seconds:06.3f}".rstrip("0").rstrip(".")
        if seconds < 10 and not seconds_text.startswith("0"):
            seconds_text = "0" + seconds_text
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds_text}"
    return f"{minutes:02d}:{seconds_text}"


def parse_timestamp(timestamp: str) -> int:
    normalized = timestamp.strip().replace(",", ".")
    parts = normalized.split(":")
    if len(parts) == 3:
        hours, minutes = int(parts[0]), int(parts[1])
        seconds_text = parts[2]
    elif len(parts) == 2:
        hours, minutes = 0, int(parts[0])
        seconds_text = parts[1]
    else:
        return 0
    second_parts = seconds_text.split(".", 1)
    seconds = int(second_parts[0])
    millis = int(second_parts[1].ljust(3, "0")[:3]) if len(second_parts) > 1 else 0
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def parse_subtitle_entries(content: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    lines = content.strip().splitlines()
    index = 0
    current_sequence: int | None = None
    auto_sequence = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if (
            line.upper().startswith("WEBVTT")
            or line.startswith("Kind:")
            or line.startswith("Language:")
            or line.startswith("NOTE")
        ):
            index += 1
            continue
        if line.isdigit():
            current_sequence = int(line)
            index += 1
            continue
        if "-->" not in line:
            index += 1
            continue
        timestamp_parts = line.split("-->")
        if len(timestamp_parts) != 2:
            index += 1
            continue
        start_text = timestamp_parts[0].strip()
        end_text = timestamp_parts[1].strip().split(" ", 1)[0]
        index += 1
        text_lines: list[str] = []
        while index < len(lines):
            text_line = lines[index].strip()
            if not text_line:
                index += 1
                break
            if text_line.isdigit() and index + 1 < len(lines) and "-->" in lines[index + 1]:
                break
            text_lines.append(text_line)
            index += 1
        text = "\n".join(text_lines).strip()
        if text:
            auto_sequence += 1
            entries.append(
                {
                    "sequence": current_sequence if current_sequence is not None else auto_sequence,
                    "start_ms": parse_timestamp(start_text),
                    "end_ms": parse_timestamp(end_text),
                    "text": text,
                }
            )
            current_sequence = None
    return entries


def combine_bilingual(
    original_entries: list[dict[str, Any]],
    translated_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    translated_by_sequence = {
        entry["sequence"]: entry
        for entry in translated_entries
        if entry.get("sequence") is not None
    }
    segments = []
    for original in original_entries:
        translated = translated_by_sequence.get(original.get("sequence"), {})
        segments.append(
            {
                "start": format_timestamp(original["start_ms"]),
                "end": format_timestamp(original["end_ms"]),
                "source_text": original["text"],
                "translated_text": translated.get("text", ""),
                "speaker": "speaker1",
            }
        )
    return segments, [dict(item) for item in SINGLE_SPEAKER_PROFILE]


def combine_translated_only(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments = [
        {
            "start": format_timestamp(entry["start_ms"]),
            "end": format_timestamp(entry["end_ms"]),
            "source_text": "",
            "translated_text": entry["text"],
            "speaker": "speaker1",
        }
        for entry in entries
    ]
    return segments, [dict(item) for item in SINGLE_SPEAKER_PROFILE]


def combine_original_only(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments = [
        {
            "start": format_timestamp(entry["start_ms"]),
            "end": format_timestamp(entry["end_ms"]),
            "source_text": entry["text"],
            "translated_text": entry["text"],
            "speaker": "speaker1",
        }
        for entry in entries
    ]
    return segments, [dict(item) for item in SINGLE_SPEAKER_PROFILE]


def parse_subtitle_input(
    original_content: str | None,
    translated_content: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    original = parse_subtitle_entries(original_content.strip()) if original_content and original_content.strip() else []
    translated = (
        parse_subtitle_entries(translated_content.strip())
        if translated_content and translated_content.strip()
        else []
    )
    if original and translated:
        return combine_bilingual(original, translated)
    if translated:
        return combine_translated_only(translated)
    if original:
        return combine_original_only(original)
    return None
