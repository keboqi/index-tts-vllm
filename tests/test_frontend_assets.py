import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class FrontendAssetTests(unittest.TestCase):
    def test_html_contains_no_inline_application_assets(self):
        html = (ROOT / "index_new.html").read_text(encoding="utf-8")
        self.assertNotIn("<style", html.lower())
        inline_scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", html, re.IGNORECASE)
        self.assertEqual(inline_scripts, [])
        self.assertIn('content="{{CHUNK_SPLIT_MIN_SILENCE_MS}}"', html)

    def test_referenced_static_assets_exist(self):
        html = (ROOT / "index_new.html").read_text(encoding="utf-8")
        references = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
        self.assertGreaterEqual(len(references), 9)
        for reference in references:
            self.assertTrue((ROOT / reference.lstrip("/")).is_file(), reference)

    def test_server_template_value_is_defined_in_core_script(self):
        core = (ROOT / "static" / "js" / "core.js").read_text(encoding="utf-8")
        self.assertIn("const CHUNK_SPLIT_MIN_SILENCE_MS = Number(", core)
        for script in (ROOT / "static" / "js").glob("*.js"):
            self.assertNotIn("{{CHUNK_SPLIT_MIN_SILENCE_MS}}", script.read_text(encoding="utf-8"))

    def test_ffmpeg_initializer_runs_after_translation_modules_load(self):
        html = (ROOT / "index_new.html").read_text(encoding="utf-8")
        state = (ROOT / "static" / "js" / "translation-state.js").read_text(
            encoding="utf-8"
        )
        chunks = (ROOT / "static" / "js" / "translation-chunks.js").read_text(
            encoding="utf-8"
        )
        bootstrap = (ROOT / "static" / "js" / "bootstrap.js").read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            "\n        updateFfmpegCommands();\n\n"
            "        function updateParallelSettingsVisibility()",
            state,
        )
        self.assertIn("function updateFfmpegCommands(options = {})", chunks)
        self.assertIn(
            "function init() {\n"
            "            bindRangeOutputs();\n"
            "            bindDelegatedActions();\n"
            "            updateFfmpegCommands();",
            bootstrap,
        )
        script_sources = re.findall(r'<script[^>]+\bsrc="([^"]+)"', html)
        self.assertLess(
            script_sources.index("/static/js/translation-chunks.js"),
            script_sources.index("/static/js/bootstrap.js"),
        )

    def test_favicon_is_a_static_asset(self):
        html = (ROOT / "index_new.html").read_text(encoding="utf-8")
        self.assertIn('href="/static/favicon.svg"', html)
        self.assertTrue((ROOT / "static" / "favicon.svg").is_file())


if __name__ == "__main__":
    unittest.main()
