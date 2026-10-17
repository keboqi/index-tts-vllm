"""ASCII-safe base filenames for generated artifacts and caches."""

from __future__ import annotations

import os
import re
import unicodedata

MAX_SAFE_BASE_FILENAME_LENGTH = 180

_READABLE_REPLACEMENTS = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "－": "-",
        "　": " ",
        "&": " and ",
        "＆": " and ",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)


def sanitize_base_filename(raw_name: str | None) -> str | None:
    if not raw_name:
        return None
    base_name = os.path.splitext(os.path.basename(str(raw_name)))[0].strip().strip(".")
    if not base_name:
        return None
    normalized = unicodedata.normalize("NFKD", base_name.translate(_READABLE_REPLACEMENTS))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", ascii_name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip(" ._-")
    if len(sanitized) > MAX_SAFE_BASE_FILENAME_LENGTH:
        tail_length = 40
        head_length = MAX_SAFE_BASE_FILENAME_LENGTH - tail_length - 1
        sanitized = f"{sanitized[:head_length].rstrip('._-')}_{sanitized[-tail_length:].lstrip('._-')}"
        sanitized = sanitized.strip(" ._-")
    return sanitized or None
