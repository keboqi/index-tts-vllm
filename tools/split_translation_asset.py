#!/usr/bin/env python3
"""One-shot split of the extracted translation JavaScript feature."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "index_new.html"
SOURCE = ROOT / "static" / "js" / "translation.js"
BOOTSTRAP = ROOT / "static" / "js" / "bootstrap.js"
BOUNDARIES = (
    ("translation-state.js", "/* ---------- Translate ---------- */"),
    ("translation-chunks.js", "function updateChunkSelectionUI()"),
    ("translation-media.js", "function formatTimestamp(ms)"),
    ("translation-segments.js", "function toggleSegmentExpand(card)"),
    ("translation-speakers.js", "let speakerColorMap = {};"),
    ("translation-requests.js", "function syncSegmentRulesFromMetadata(rules)"),
)


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    positions = []
    for name, marker in BOUNDARIES:
        position = source.find(marker)
        if position < 0:
            raise RuntimeError(f"missing translation boundary {marker!r}")
        positions.append((name, position))
    if positions != sorted(positions, key=lambda item: item[1]):
        raise RuntimeError("translation boundaries are out of order")

    assets = {}
    for index, (name, start) in enumerate(positions):
        end = positions[index + 1][1] if index + 1 < len(positions) else len(source)
        prefix = '"use strict";\n\n'
        assets[name] = prefix + source[start:end].strip() + "\n"

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

    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    bootstrap_marker = (
        "function init() {\n"
        "            bindRangeOutputs();\n"
        "            bindDelegatedActions();\n"
    )
    if bootstrap_marker not in bootstrap:
        raise RuntimeError("could not find the frontend bootstrap initializer")
    BOOTSTRAP.write_text(
        bootstrap.replace(
            bootstrap_marker,
            bootstrap_marker + "            updateFfmpegCommands();\n",
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )

    for name, _marker in BOUNDARIES:
        (SOURCE.parent / name).write_text(
            assets[name],
            encoding="utf-8",
            newline="\n",
        )

    tags = "\n".join(
        f'    <script defer src="/static/js/{name}"></script>'
        for name, _marker in BOUNDARIES
    )
    html = HTML.read_text(encoding="utf-8")
    old_tag = '    <script defer src="/static/js/translation.js"></script>'
    if old_tag not in html:
        raise RuntimeError("translation script tag is missing")
    HTML.write_text(html.replace(old_tag, tags, 1), encoding="utf-8", newline="\n")
    SOURCE.unlink()


if __name__ == "__main__":
    main()
