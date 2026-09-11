import asyncio
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, patch

from indextts_web.services.translation.moss_client import MossModelClient
from indextts_web.services.translation.moss_runtime import MossRuntime
from tests.test_modal_gpu_profiles import ROOT, load_definition


class MossClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_states_and_posts_lifecycle_actions(self):
        client = MossModelClient(url="http://127.0.0.1:8003/", enabled=True)
        for state, actions in (("unloaded", ["wake"]), ("loaded", ["sleep", "unload"]),
                               ("sleeping", ["wake", "unload"])):
            payload = {"service": "moss-transcribe", "state": state, "busy": False,
                       "capabilities": ["sleep", "wake", "unload"]}
            with self.subTest(state=state), patch("urllib.request.urlopen",
                    side_effect=lambda *_args, _payload=payload, **_kw: io.BytesIO(json.dumps(_payload).encode())) as http:
                rows = await client.inventory()
                self.assertEqual(rows[0]["state"], state)
                self.assertEqual(rows[0]["actions"], actions)
                self.assertEqual(http.call_args.args[0].full_url, client.url + "/model/status")
                for action in actions:
                    await client.change(action)
                    self.assertEqual(http.call_args.args[0].method, "POST")
                    self.assertEqual(http.call_args.args[0].full_url, client.url + "/model/" + action)

    async def test_unavailable_and_external_servers(self):
        client = MossModelClient(url="http://127.0.0.1:8003", enabled=True)
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            row = (await client.inventory())[0]
            self.assertEqual(row["state"], "unavailable")
            self.assertEqual(row["actions"], [])
            with self.assertRaises(OSError):
                await client.change("wake")
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(client.url, 404, "missing", {}, None)):
            self.assertEqual(await client.inventory(), [])
        with patch.dict(os.environ, {"MOSS_TRANSCRIBE_BACKEND": "python"}):
            self.assertEqual(await MossModelClient.from_env().inventory(), [])


class ModelManagerRoutingTests(unittest.IsolatedAsyncioTestCase):
    def namespace(self):
        class HttpError(Exception):
            def __init__(self, *, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code

        client = SimpleNamespace(inventory=AsyncMock(return_value=[{"key": "moss_transcribe", "state": "loaded"}]),
                                 change=AsyncMock())
        namespace = {"Dict": dict, "List": list, "Any": Any, "Request": object,
                     "JSONResponse": lambda **kwargs: kwargs["content"], "HTTPException": HttpError,
                     "moss_model_client": client, "_loaded_model_inventory": lambda: [],
                     "_cuda_memory_summary": lambda: {}, "_model_manager_lock": asyncio.Lock(),
                     "_request_has_json_body": lambda request: True,
                     "tts_manager": SimpleNamespace(sleep_engine=AsyncMock(), is_ready=lambda: False),
                     "confucius_backend_manager": SimpleNamespace(_process_running=lambda: False),
                     "indextts25_backend_manager": SimpleNamespace(process_running=lambda: False),
                     "_run_blocking": AsyncMock(return_value=[]), "_unload_optional_model_sync": Mock()}
        for name in ("_managed_model_inventory", "_change_moss_model", "api_models_status",
                     "api_models_unload", "api_models_wake"):
            load_definition(ROOT / "fastapi_webui_v2_impl.py", name, namespace)
        return namespace

    async def test_moss_is_visible_and_all_actions_reach_its_process(self):
        ns = self.namespace()
        self.assertEqual((await ns["api_models_status"]())["models"][0]["key"], "moss_transcribe")
        for mode in ("sleep", "unload"):
            result = await ns["api_models_unload"](SimpleNamespace(json=AsyncMock(
                return_value={"model_key": "moss_transcribe", "mode": mode})))
            self.assertEqual(result["unloaded"], ["moss_transcribe"])
            ns["moss_model_client"].change.assert_awaited_with(mode)
        await ns["api_models_wake"](SimpleNamespace(json=AsyncMock(return_value={"model_key": "moss_transcribe"})))
        ns["moss_model_client"].change.assert_awaited_with("wake")
        result = await ns["api_models_unload"](SimpleNamespace(json=AsyncMock(return_value={"model_key": "all"})))
        self.assertIn("moss_transcribe", result["unloaded"])
        ns["moss_model_client"].change.assert_awaited_with("unload")

    async def test_management_failure_is_not_reported_as_success(self):
        ns = self.namespace()
        ns["moss_model_client"].change.side_effect = RuntimeError("out of memory")
        with self.assertRaisesRegex(ns["HTTPException"], "MOSS wake failed") as error:
            await ns["api_models_wake"](SimpleNamespace(json=AsyncMock(return_value={"model_key": "moss_transcribe"})))
        self.assertEqual(error.exception.status_code, 502)


class MossServerRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcription_lazy_wakes_and_management_uses_same_runtime(self):
        runtime = MossRuntime(loader=lambda: (Mock(), Mock(), "cuda:0", "bf16"), release_cache=Mock())
        generate = Mock(return_value={"text": "hello"})
        ns = {"Any": Any, "Optional": Optional, "os": os, "tempfile": tempfile,
              "MODEL_PATH": "moss-checkpoint", "runtime": runtime, "run_in_threadpool": asyncio.to_thread,
              "build_transcription_messages": lambda path, **kw: Path(path).read_bytes(),
              "generate_transcription": generate, "UploadFile": object,
              "File": lambda *_: None, "Form": lambda *_: None}
        for name in ("model_status", "sleep_model", "wake_model", "unload_model", "models", "_transcribe_audio", "transcribe"):
            load_definition(ROOT / "moss_transcribe_server.py", name, ns)
        self.assertEqual(ns["models"]()["data"][0]["id"], "moss-checkpoint")
        self.assertEqual((await ns["model_status"]())["state"], "unloaded")
        for expected_before in ("unloaded", "sleeping", "unloaded"):
            self.assertEqual((await ns["model_status"]())["state"], expected_before)
            file = SimpleNamespace(filename="audio.wav", read=AsyncMock(return_value=b"audio"))
            self.assertEqual(await ns["transcribe"](file, "prompt", 123), {"text": "hello"})
            self.assertEqual(generate.call_args.args[2], b"audio")
            self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 123)
            if expected_before == "sleeping":
                await ns["unload_model"]()
            else:
                await ns["sleep_model"]()
        self.assertEqual((await ns["wake_model"]())["state"], "loaded")


class ModelManagerFrontendTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node required for frontend behavior test")
    def test_controls_follow_moss_state(self):
        script = r"""
const fs = require('fs');
const vm = require('vm');
const elements = {modelManagerGpu: {}, modelManagerList: {}};
const context = {document: {querySelector: () => null, getElementById: id => elements[id]},
                 escapeHtml: value => String(value)};
vm.createContext(context);
vm.runInContext(fs.readFileSync('static/js/core.js', 'utf8'), context);
for (const [state, actions, labels] of [
    ['loaded', ['sleep', 'unload'], ['Sleep', 'Unload']],
    ['sleeping', ['wake', 'unload'], ['Wake', 'Unload']],
    ['unloaded', ['wake'], ['Load']],
    ['unavailable', [], []],
]) {
    context.renderModelManager({models: [{key: 'moss_transcribe', name: 'MOSS', state, actions}]});
    const html = elements.modelManagerList.innerHTML;
    const actual = [...html.matchAll(/<button[^>]*>(.*?)<\/button>/g)].map(match => match[1]);
    if (JSON.stringify(actual) !== JSON.stringify(labels)) throw Error(state + ': ' + html);
}
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
