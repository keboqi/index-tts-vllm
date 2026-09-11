from __future__ import annotations

import ast
import asyncio
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from indextts_web.infrastructure.gpu_work import GpuWorkCoordinator
from indextts_web.services.translation.qwen_worker import QwenOmniVadWorker

ROOT = Path(__file__).resolve().parents[1]
START_PROCESS = asyncio.create_subprocess_exec
FAKE_PIPELINE = '''
import os, time
from pathlib import Path

def translate_audio(audio_bytes, **options):
    mode = options.pop("test_mode", "success")
    if mode == "error":
        raise RuntimeError("forced aligner unavailable")
    if mode == "malformed":
        return ["bad result"]
    if mode == "block":
        Path(os.environ["QWEN_TEST_STARTED"]).write_text(str(os.getpid()))
        time.sleep(60)
    print("ASR progress output must not corrupt the result")
    return ([{"source_text": "你好", "audio": audio_bytes.hex(), "options": options}],
            [{"speaker": "speaker1"}], "原始结果", {"hit": False})
'''


class QwenWorkerTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # The Modal SDK selects Windows' selector loop globally. These real
        # subprocess tests require the platform's subprocess-capable loop.
        cls.original_policy = asyncio.get_event_loop_policy()
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    @classmethod
    def tearDownClass(cls):
        asyncio.set_event_loop_policy(cls.original_policy)

    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="qwen worker test ")
        self.root = Path(self.directory.name)
        (self.root / "qwen_omnivad_pipeline.py").write_text(FAKE_PIPELINE, encoding="utf-8")
        self.marker = self.root / "started.txt"
        self.environment = patch.dict(os.environ, {
            "QWEN_OMNIVAD_PYTHON": sys.executable,
            "QWEN_OMNIVAD_WORKER_TIMEOUT": "10",
            "QWEN_TEST_STARTED": str(self.marker),
            "PYTHONPATH": str(ROOT),
        })
        self.environment.start()
        self.coordinator = GpuWorkCoordinator(enabled=True)
        self.prepare = AsyncMock()
        self.worker = QwenOmniVadWorker(app_dir=self.root, coordinator=self.coordinator,
                                       prepare_gpu=self.prepare)
        self.processes = []

        async def start_process(*args, **kwargs):
            process = await START_PROCESS(*args, stdout=asyncio.subprocess.DEVNULL,
                                          stderr=asyncio.subprocess.DEVNULL, **kwargs)
            self.processes.append(process)
            return process

        self.launcher = patch("indextts_web.services.translation.qwen_worker.asyncio.create_subprocess_exec",
                              side_effect=start_process)
        self.launcher.start()

    async def asyncTearDown(self):
        for process in self.processes:
            if process.returncode is None:
                await self.worker._stop(process)
        self.launcher.stop()
        self.environment.stop()
        self.directory.cleanup()

    async def test_worker_preserves_audio_options_unicode_and_result_contract(self):
        options = {"input_mime_type": "audio/wav", "dest_language": "Chinese",
                   "enable_translation": False, "enable_diarization": True,
                   "diarization_backend": "sortformer", "enable_forced_aligner": True,
                   "merge_gap_seconds": 0.001, "force_refresh": True}
        fallback = Mock(side_effect=AssertionError("must not load ASR in the TTS process"))
        result = await self.worker.translate(b"\x00\xffaudio", local_pipeline=fallback, **options)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0][0]["audio"], b"\x00\xffaudio".hex())
        self.assertEqual(result[0][0]["options"], options)
        self.assertEqual(result[0][0]["source_text"], "你好")
        self.assertEqual(result[2], "原始结果")
        self.assertEqual(result[3], {"hit": False})
        self.prepare.assert_awaited_once()
        self.assertEqual(self.processes[0].returncode, 0)
        fallback.assert_not_called()

    async def test_native_pipeline_remains_available_without_worker_configuration(self):
        with patch.dict(os.environ, {"QWEN_OMNIVAD_PYTHON": ""}):
            self.assertTrue(self.worker.is_available(local_available=True))
            fallback = Mock(return_value=([], [], "raw", {}))
            result = await self.worker.translate(b"audio", local_pipeline=fallback, dest_language="English")
        self.assertEqual(result, ([], [], "raw", {}))
        fallback.assert_called_once_with(b"audio", dest_language="English")
        self.assertEqual(self.processes, [])

    async def test_invalid_config_does_not_silently_use_main_environment(self):
        with patch.dict(os.environ, {"QWEN_OMNIVAD_PYTHON": str(self.root / "missing-python")}):
            self.assertFalse(self.worker.is_available(local_available=True))
            with self.assertRaisesRegex(RuntimeError, "executable Python"):
                await self.worker.translate(b"audio", local_pipeline=Mock())
        self.prepare.assert_not_awaited()

    async def test_worker_failure_and_invalid_result_are_reported(self):
        for mode, message in (("error", "forced aligner unavailable"), ("malformed", "invalid pipeline result")):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, message):
                await self.worker.translate(b"audio", test_mode=mode)
        self.assertTrue(all(process.returncode is not None for process in self.processes))

    async def test_worker_waits_for_active_tts(self):
        async with self.coordinator.use("index"):
            job = asyncio.create_task(self.worker.translate(b"audio"))
            await asyncio.sleep(0)
            self.assertFalse(job.done())
            self.prepare.assert_not_awaited()
            self.assertEqual(self.processes, [])
        await asyncio.wait_for(job, timeout=10)
        self.prepare.assert_awaited_once()

    async def test_cancellation_stops_worker_before_releasing_gpu(self):
        job = asyncio.create_task(self.worker.translate(b"audio", test_mode="block"))
        async def wait_started():
            while not self.marker.exists():  # noqa: ASYNC110 -- signal comes from a separate process
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait_started(), timeout=10)
        acquired = asyncio.Event()

        async def next_tts():
            async with self.coordinator.use("index"):
                self.assertIsNotNone(self.processes[0].returncode)
                acquired.set()

        tts_job = asyncio.create_task(next_tts())
        await asyncio.sleep(0)
        self.assertFalse(acquired.is_set())
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await job
        await asyncio.wait_for(tts_job, timeout=5)
        self.assertTrue(acquired.is_set())

    async def test_timeout_reaps_worker(self):
        with patch.dict(os.environ, {"QWEN_OMNIVAD_WORKER_TIMEOUT": "0.2"}):
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                await self.worker.translate(b"audio", test_mode="block")
        self.assertIsNotNone(self.processes[0].returncode)
        async with self.coordinator.use("index"):
            pass


class QwenRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_webui_uses_worker_when_main_environment_has_no_qwen_asr(self):
        path = ROOT / "fastapi_webui_v2_impl.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                        and node.name == "_build_translation_segments")
        branch = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Name) and node.test.id == "use_qwen_omnivad"
                      and any(isinstance(item, ast.Attribute) and item.attr == "translate"
                              for statement in node.body for item in ast.walk(statement)))
        wrapper = ast.parse("async def run_branch():\n    pass\n")
        wrapper.body[0].body = branch.body
        result = ([{"source_text": "hello"}], [], "raw", {"hit": False})
        worker = SimpleNamespace(is_available=Mock(return_value=True), translate=AsyncMock(return_value=result))
        options = {"gemini_mime_type": "audio/wav", "dest_language": "Chinese", "translate_enabled": True,
                   "resolved_translation_llm_model": "tencent/Hy-MT2-1.8B", "force_gemini_regenerate": False,
                   "qwen_omnivad_enable_diarization": True, "qwen_omnivad_diarization_backend": "sortformer",
                   "qwen_omnivad_diarization_min_seconds": 0, "qwen_omnivad_enable_forced_aligner": True,
                   "qwen_omnivad_merge_gap_seconds": 0.001}
        namespace = {**options, "qwen_omnivad_worker": worker, "is_qwen_omnivad_available": lambda: False,
                     "_run_qwen_omnivad_pipeline_sync": None, "processed_audio_bytes": b"audio", "print": Mock()}
        exec(compile(wrapper, str(path), "exec"), namespace)
        await namespace["run_branch"]()
        worker.is_available.assert_called_once_with(local_available=False)
        worker.translate.assert_awaited_once_with(
            b"audio", local_pipeline=None, input_mime_type="audio/wav", dest_language="Chinese",
            enable_translation=True, translation_llm_model="tencent/Hy-MT2-1.8B", force_refresh=False,
            enable_diarization=True, diarization_backend="sortformer", diarization_min_seconds=0,
            enable_forced_aligner=True, merge_gap_seconds=0.001,
        )

    @unittest.skipUnless(importlib.util.find_spec("modal"), "Modal SDK required")
    def test_modal_installs_asr_in_its_own_interpreter(self):
        import deploy_vllm_indextts_v2 as deploy

        tree = ast.parse((ROOT / "deploy_vllm_indextts_v2.py").read_text(encoding="utf-8-sig"))
        commands, image_env = [], {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "run_commands":
                commands.extend(eval(compile(ast.Expression(arg), "<image command>", "eval"), vars(deploy))
                                for arg in node.args)
            if node.func.attr == "env":
                image_env.update(eval(compile(ast.Expression(node.args[0]), "<image env>", "eval"), vars(deploy)))
        installations = [command for command in commands if "pip install" in command and "qwen-asr==" in command]
        self.assertEqual(len(installations), 1)
        self.assertTrue(installations[0].startswith(deploy.QWEN_ASR_PYTHON + " -m pip install "))
        self.assertIn("transformers==4.57.6", installations[0])
        self.assertEqual(image_env["QWEN_OMNIVAD_PYTHON"], deploy.QWEN_ASR_PYTHON)
        self.assertEqual(image_env["QWEN_OMNIVAD_MODEL_DIR"], "/persistent_app/checkpoints/qwen_omnivad")
        self.assertEqual(image_env["QWEN_OMNIVAD_CACHE_DIR"], "/persistent_cache/qwen_omnivad")
