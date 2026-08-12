#!/usr/bin/env python3
"""One-shot delegation of legacy subtitle helpers to the translation service."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "fastapi_webui_v2_impl.py"

REPLACEMENTS = {
    "_parse_srt_timestamp_to_ms": "return parse_subtitle_timestamp(timestamp)",
    "_parse_srt_file_with_timestamps": "return parse_subtitle_entries(srt_content)",
    "_combine_srt_subtitles_to_segments": (
        "return combine_bilingual_subtitles(original_srt_entries, translated_srt_entries)"
    ),
    "_combine_srt_translated_only_to_segments": (
        "return combine_translated_subtitles(translated_srt_entries)"
    ),
    "_combine_srt_original_only_to_segments": (
        "return combine_original_subtitles(original_srt_entries)"
    ),
    "_parse_srt_input_to_segments": (
        "return parse_subtitle_input(original_srt_content, translated_srt_content)"
    ),
}

IMPORT_BLOCK = """from indextts_web.services.translation.subtitles import (
    combine_bilingual as combine_bilingual_subtitles,
    combine_original_only as combine_original_subtitles,
    combine_translated_only as combine_translated_subtitles,
    parse_subtitle_entries,
    parse_subtitle_input,
    parse_timestamp as parse_subtitle_timestamp,
)
"""


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in REPLACEMENTS
    }
    missing = REPLACEMENTS.keys() - functions.keys()
    if missing:
        raise RuntimeError(f"missing subtitle functions: {sorted(missing)}")
    for name, node in sorted(functions.items(), key=lambda item: item[1].lineno, reverse=True):
        body_start = node.body[0].lineno - 1
        body_end = node.body[-1].end_lineno
        lines[body_start:body_end] = [f"    {REPLACEMENTS[name]}\n"]
    rewritten = "".join(lines)
    anchor = "from indextts_web.services.tts.factory import build_backend_registry\n"
    if anchor not in rewritten:
        raise RuntimeError("subtitle import anchor is missing")
    rewritten = rewritten.replace(anchor, anchor + IMPORT_BLOCK, 1)
    SOURCE.write_text(rewritten, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
