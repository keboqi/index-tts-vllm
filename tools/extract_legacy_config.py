#!/usr/bin/env python3
"""One-shot mechanical extraction of legacy CLI configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "legacy_fastapi_webui_v2.py"


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise RuntimeError(f"missing start marker: {start!r}")
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise RuntimeError(f"missing end marker: {end!r}")
    return text[:start_index] + replacement + text[end_index:]


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8-sig")
    source = replace_between(
        source,
        "@dataclass(frozen=True)\nclass AppSettings:",
        "# Global thread executor",
        "from indextts_web.config import load_settings\n\n\n",
    )
    source = replace_between(
        source,
        'parser = argparse.ArgumentParser(description="IndexTTS vLLM v2 FastAPI WebUI")',
        "SETTINGS = AppSettings.from_namespace(cmd_args)",
        "SETTINGS = load_settings(sys.argv[1:], allow_unknown=True)",
    )
    source = source.replace(
        "SETTINGS = load_settings(sys.argv[1:], allow_unknown=True)"
        "SETTINGS = AppSettings.from_namespace(cmd_args)",
        "SETTINGS = load_settings(sys.argv[1:], allow_unknown=True)",
        1,
    )
    source = source.replace("# Configuration\nimport argparse\n", "# Configuration\n", 1)
    SOURCE.write_text(source, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
