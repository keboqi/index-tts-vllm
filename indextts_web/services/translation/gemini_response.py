"""Parse Gemini transcription/translation responses into segments and speakers."""

from __future__ import annotations

import json
import re
from typing import Any

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


COERCE_SPEAKER_SEGMENTS_PATTERN = re.compile(
    r'^\s*(\[[\s\S]*?\])\s*,\s*"segments"\s*:\s*(\[[\s\S]*\])\s*$',
    re.DOTALL,
)


SPEAKER_ID_PATTERN = re.compile(r"^speaker\s*(\d+)$", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    match = JSON_FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        return text[3:-3].strip()
    # Fallback: extract JSON array if present
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    return text


def coerce_json_payload(raw: str) -> str | None:
    trimmed = (raw or "").strip()
    if not trimmed:
        return None
    match = COERCE_SPEAKER_SEGMENTS_PATTERN.match(trimmed)
    if match:
        speakers_part, segments_part = match.groups()
        return f'{{"speakers": {speakers_part}, "segments": {segments_part}}}'
    return None


def canonicalize_speaker_id(raw_value: Any, fallback_index: int) -> str:
    fallback_index = max(1, int(fallback_index))
    if isinstance(raw_value, int | float) and raw_value >= 1:
        return f"speaker{int(raw_value)}"
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if not candidate:
            return f"speaker{fallback_index}"
        match = SPEAKER_ID_PATTERN.match(candidate)
        if match:
            return f"speaker{int(match.group(1))}"
        digits = re.findall(r"\d+", candidate)
        if digits:
            return f"speaker{int(digits[0])}"
    return f"speaker{fallback_index}"


def normalize_speaker_profiles(raw_profiles: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_profiles, list):
        return []
    normalized: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for idx, entry in enumerate(raw_profiles, start=1):
        entry_dict: dict[str, Any]
        if isinstance(entry, dict):
            entry_dict = entry
        else:
            entry_dict = {"description": str(entry)}
        candidate_id = entry_dict.get("id") or entry_dict.get("speaker") or entry_dict.get("label") or entry_dict.get("name")
        speaker_id = canonicalize_speaker_id(candidate_id, idx)
        while speaker_id in used_ids:
            idx += 1
            speaker_id = canonicalize_speaker_id(None, idx)
        used_ids.add(speaker_id)
        description = str(
            entry_dict.get("description")
            or entry_dict.get("summary")
            or entry_dict.get("traits")
            or entry_dict.get("role")
            or ""
        ).strip()
        if not description:
            description = f"{speaker_id.title()} voice"
        display_name = str(entry_dict.get("label") or entry_dict.get("name") or speaker_id.title()).strip()
        normalized.append(
            {
                "id": speaker_id,
                "label": display_name or speaker_id.title(),
                "description": description,
            }
        )
    return normalized


def parse_gemini_json(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cleaned = strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        coerced_payload = coerce_json_payload(cleaned)
        if coerced_payload:
            try:
                data = json.loads(coerced_payload)
            except json.JSONDecodeError as coerced_exc:
                preview = cleaned if len(cleaned) <= 4000 else f"{cleaned[:4000]}...<truncated>"
                print("❌ Gemini JSON parse error (after coercion). Raw response preview:\n", preview)
                raise ValueError(f"Gemini returned invalid JSON: {coerced_exc}") from coerced_exc
        else:
            preview = cleaned if len(cleaned) <= 4000 else f"{cleaned[:4000]}...<truncated>"
            print("❌ Gemini JSON parse error. Raw response preview:\n", preview)
            raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc
    if isinstance(data, list):
        return data, []
    if isinstance(data, dict):
        segments = data.get("segments")
        if not isinstance(segments, list):
            raise ValueError("Gemini response must include a 'segments' array.")
        speaker_profiles = normalize_speaker_profiles(data.get("speakers"))
        return segments, speaker_profiles
    raise ValueError("Gemini response must be a JSON object or array.")


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""

    for attr in ("text", "output_text", "output_texts"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            concatenated = "\n".join(str(item) for item in value if item)
            if concatenated.strip():
                return concatenated.strip()

    parts_text: list[str] = []

    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts:
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text.strip():
                            parts_text.append(part_text.strip())
            if isinstance(candidate, dict):
                for part in candidate.get("content", {}).get("parts", []):
                    part_text = part.get("text")
                    if isinstance(part_text, str) and part_text.strip():
                        parts_text.append(part_text.strip())

    if not parts_text:
        contents = getattr(response, "contents", None)
        if contents:
            for content in contents:
                parts = getattr(content, "parts", None)
                if parts:
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text.strip():
                            parts_text.append(part_text.strip())
                elif isinstance(content, dict):
                    for part in content.get("parts", []):
                        part_text = part.get("text")
                        if isinstance(part_text, str) and part_text.strip():
                            parts_text.append(part_text.strip())

    return "\n".join(parts_text).strip()
