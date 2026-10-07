import shutil
import subprocess
import unittest
from html.parser import HTMLParser

from tests.support import ROOT


class AssetInventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []
        self.scripts = []
        self.backends = {}
        self.select = None
        self.meta = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style"} and not attrs.get("src"):
            raise AssertionError("Application code must be served as a static asset")
        if tag == "script":
            self.scripts.append(attrs)
        for name in ("src", "href"):
            if attrs.get(name, "").startswith("/static/"):
                self.assets.append(attrs[name])
        if tag == "select":
            self.select = attrs.get("id")
        if tag == "option" and self.select in {"ttsBackend", "translateTtsBackend"}:
            self.backends.setdefault(self.select, set()).add(attrs.get("value"))
        if tag == "meta":
            self.meta[attrs.get("name")] = attrs.get("content")

    def handle_endtag(self, tag):
        if tag == "select":
            self.select = None


class FrontendAssetTests(unittest.TestCase):
    def test_public_html_assets_and_backend_controls(self):
        inventory = AssetInventory()
        inventory.feed((ROOT / "index_new.html").read_text(encoding="utf-8"))
        self.assertTrue(inventory.assets)
        for reference in inventory.assets:
            self.assertTrue((ROOT / reference.lstrip("/")).is_file(), reference)
        self.assertIn("/static/favicon.svg", inventory.assets)
        self.assertEqual(inventory.meta["chunk-split-min-silence-ms"], "{{CHUNK_SPLIT_MIN_SILENCE_MS}}")
        for backend_options in inventory.backends.values():
            self.assertEqual(backend_options, {"index", "index25", "confucius"})
        self.assertEqual(set(inventory.backends), {"ttsBackend", "translateTtsBackend"})
        sources = [script["src"] for script in inventory.scripts]
        self.assertLess(sources.index("/static/js/translation-chunks.js"), sources.index("/static/js/bootstrap.js"))
        for script in inventory.scripts:
            if script["src"].startswith("/static/"):
                self.assertIn("defer", script)

    @unittest.skipUnless(shutil.which("node"), "Node required for frontend behavior tests")
    def test_script_bootstrap_reads_server_settings_and_selects_backend_languages(self):
        script = r"""
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
for (const [metaContent, expected] of [['750', 750], ['', 1000]]) {
    const listeners = {};
    const backend = {value: 'index25', addEventListener: (event, fn) => listeners[event] = fn};
    const language = {value: 'ja', replaceChildren: (...options) => language.options = options};
    const elements = {ttsBackend: backend, ttsLanguage: language, emotionText: {}, emotionWeight: {}};
    let ffmpegUpdates = 0;
    const context = {
        document: {querySelector: selector => selector.startsWith('meta[') ? {content: metaContent} : null,
                   querySelectorAll: () => [],
                   getElementById: id => elements[id],
                   createElement: () => ({}), readyState: 'loading', addEventListener: () => {}},
        window: {localStorage: {getItem: () => null, setItem() {}}},
        bindRangeOutputs() {}, bindDelegatedActions() {}, initDurationControlControls() {},
        updateFfmpegCommands: () => ffmpegUpdates++,
    };
    vm.createContext(context);
    vm.runInContext(fs.readFileSync('static/js/core.js', 'utf8'), context);
    assert.strictEqual(vm.runInContext('CHUNK_SPLIT_MIN_SILENCE_MS', context), expected);
    vm.runInContext(fs.readFileSync('static/js/bootstrap.js', 'utf8'), context);
    assert.strictEqual(ffmpegUpdates, 1);
    assert.deepStrictEqual(Array.from(language.options, option => option.value), ['auto', 'en', 'zh', 'ja', 'es', 'ar']);
    assert.strictEqual(language.value, 'ja');
    assert.strictEqual(elements.emotionText.disabled, false);
    backend.value = 'index';
    listeners.change();
    assert.strictEqual(language.value, 'auto');
    backend.value = 'confucius';
    listeners.change();
    assert.strictEqual(elements.emotionText.disabled, true);
}
// Execute the translation modules in the HTML's load order, so eager calls
// cannot access globals that are initialized only by a later script.
const context = {
    document: {querySelector: () => null, querySelectorAll: () => [], getElementById: () => null,
               readyState: 'loading', addEventListener() {}, createElement: () => ({})},
    window: {localStorage: {getItem: () => null, setItem() {}}}, bindRangeOutputs() {},
};
vm.createContext(context);
const modules = new Set(['core.js', 'translation-state.js', 'translation-chunks.js', 'bootstrap.js']);
for (const match of fs.readFileSync('index_new.html', 'utf8').matchAll(/<script[^>]*src="(\/static\/js\/([^"]+))"/g)) {
    if (!modules.has(match[2])) continue;
    vm.runInContext(fs.readFileSync(match[1].slice(1), 'utf8'), context);
    if (match[2] === 'core.js') context.hideStatus = () => {};
}
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
