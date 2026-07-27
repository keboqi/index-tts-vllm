#!/usr/bin/env python3
"""Mechanically externalize the legacy single-file UI.

The source JavaScript was wrapped in one IIFE. The extractor removes that
wrapper and writes ordered classic scripts so top-level declarations remain
visible to later feature files without requiring a framework or bundler.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "index_new.html"
STATIC_ROOT = ROOT / "static"

SCRIPT_BOUNDARIES = (
    ("core.js", "const CHUNK_SPLIT_MIN_SILENCE_MS = Number("),
    ("stable-audio.js", "let stableAudioModels = [];"),
    ("video.js", "function getSelectedDownloadedVideoId()"),
    ("speakers.js", "async function loadSpeakers()"),
    ("synthesis-form.js", "/* ---------- Synthesis ---------- */"),
    ("translation-state.js", "/* ---------- Translate ---------- */"),
    ("translation-chunks.js", "function updateChunkSelectionUI()"),
    ("translation-media.js", "function formatTimestamp(ms)"),
    ("translation-segments.js", "function toggleSegmentExpand(card)"),
    ("translation-speakers.js", "let speakerColorMap = {};"),
    ("translation-requests.js", "function syncSegmentRulesFromMetadata(rules)"),
    ("synthesis-streaming.js", "async function handleRegularRequest("),
    ("bootstrap.js", "function init()"),
)

SCRIPT_TAGS = "\n".join(
    f'{"    " if index else ""}<script defer src="/static/js/{name}"></script>'
    for index, (name, _marker) in enumerate(SCRIPT_BOUNDARIES)
)


def _extract_single(pattern: str, text: str, label: str) -> tuple[str, str]:
    matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} block, found {len(matches)}")
    match = matches[0]
    return match.group("body"), text[: match.start()] + f"{{{{{label}}}}}" + text[match.end() :]


def _split_javascript(source: str) -> dict[str, str]:
    script = textwrap.dedent(source).strip()
    opening = re.compile(r"^\(function\s*\(\)\s*\{\s*['\"]use strict['\"];\s*", re.DOTALL)
    script, count = opening.subn('"use strict";\n\n', script, count=1)
    if count != 1:
        raise RuntimeError("could not find the UI IIFE opening")
    script, count = re.subn(r"\}\)\s*\(\s*\)\s*;\s*$", "", script, count=1)
    if count != 1:
        raise RuntimeError("could not find the UI IIFE closing")
    script = script.replace(
        "const CHUNK_SPLIT_MIN_SILENCE_MS = {{CHUNK_SPLIT_MIN_SILENCE_MS}};",
        "const CHUNK_SPLIT_MIN_SILENCE_MS = Number("
        "document.querySelector('meta[name=\"chunk-split-min-silence-ms\"]')?.content || 1000"
        ");",
        1,
    )

    positions: list[tuple[str, int]] = []
    for name, marker in SCRIPT_BOUNDARIES:
        position = script.find(marker)
        if position < 0:
            raise RuntimeError(f"could not find JavaScript boundary {marker!r}")
        positions.append((name, position))
    if positions != sorted(positions, key=lambda item: item[1]):
        raise RuntimeError("JavaScript boundaries are out of order")

    assets: dict[str, str] = {}
    for index, (name, start) in enumerate(positions):
        end = positions[index + 1][1] if index + 1 < len(positions) else len(script)
        prefix = '"use strict";\n\n'
        assets[name] = prefix + script[start:end].strip() + "\n"

    early_initializer = (
        "\n        updateFfmpegCommands();\n\n"
        "        function updateParallelSettingsVisibility()"
    )
    if early_initializer not in assets["translation-state.js"]:
        raise RuntimeError("could not relocate the deferred FFmpeg command initializer")
    assets["translation-state.js"] = assets["translation-state.js"].replace(
        early_initializer,
        "\n        function updateParallelSettingsVisibility()",
        1,
    )
    bootstrap_marker = (
        "function init() {\n"
        "            bindRangeOutputs();\n"
        "            bindDelegatedActions();\n"
    )
    if bootstrap_marker not in assets["bootstrap.js"]:
        raise RuntimeError("could not find the frontend bootstrap initializer")
    assets["bootstrap.js"] = assets["bootstrap.js"].replace(
        bootstrap_marker,
        bootstrap_marker + "            updateFfmpegCommands();\n",
        1,
    )
    return assets


def main() -> None:
    html = HTML_PATH.read_text(encoding="utf-8-sig")
    if "/static/js/core.js" in html:
        raise RuntimeError("frontend assets have already been extracted")

    style, html = _extract_single(
        r"<style\b[^>]*>(?P<body>.*?)</style>",
        html,
        "STYLE_BLOCK",
    )
    script, html = _extract_single(
        r"<script\b[^>]*>(?P<body>.*?)</script>",
        html,
        "SCRIPT_BLOCK",
    )

    css_dir = STATIC_ROOT / "css"
    js_dir = STATIC_ROOT / "js"
    css_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)
    (css_dir / "app.css").write_text(
        textwrap.dedent(style).strip() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for name, content in _split_javascript(script).items():
        (js_dir / name).write_text(content, encoding="utf-8", newline="\n")

    html = html.replace(
        "{{STYLE_BLOCK}}",
        '<link rel="stylesheet" href="/static/css/app.css">',
    )
    html = html.replace("{{SCRIPT_BLOCK}}", SCRIPT_TAGS)
    html = html.replace(
        "<head>",
        '<head>\n'
        '    <meta name="chunk-split-min-silence-ms" '
        'content="{{CHUNK_SPLIT_MIN_SILENCE_MS}}">\n'
        '    <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">',
        1,
    )
    HTML_PATH.write_text(html, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
