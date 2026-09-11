import ast
import asyncio
import threading
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

from indextts_web.infrastructure.gpu_work import GpuWorkCoordinator, await_gpu_job, gpu_operation
from indextts_web.infrastructure.vllm_memory import GpuWakeError
from tests.test_modal_gpu_profiles import ROOT, load_definition


class IndexWakeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / "fastapi_webui_v2_impl.py").read_text(encoding="utf-8-sig"))
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TTSManager")
        cls.code = compile(ast.Module(body=[node], type_ignores=[]), "<TTSManager>", "exec")

    def setUp(self):
        self.memory = {"available": True, "total_mb": 24 * 1024, "free_mb": 10 * 1024}
        self.ns = {
            "asyncio": asyncio, "Dict": dict, "Any": Any, "gpu_operation": gpu_operation,
            "GpuWakeError": GpuWakeError, "GPU_COORDINATOR": GpuWorkCoordinator(enabled=True),
            "GPU_PROFILE": SimpleNamespace(index=SimpleNamespace(gpu_memory_utilization=0.25),
                                           emotion=SimpleNamespace(gpu_memory_utilization=0.125)),
            "indextts25_backend_manager": SimpleNamespace(process_running=lambda: False),
            "confucius_backend_manager": SimpleNamespace(_process_running=lambda: False),
            "_release_translation_gpu_models": AsyncMock(), "_release_cuda_cache": Mock(),
            "_run_blocking": AsyncMock(), "_cuda_memory_summary": lambda: self.memory,
        }
        exec(self.code, self.ns)
        self.manager = self.ns["TTSManager"]()
        self.manager._initialized = True
        self.manager.gpu_coordinator = self.ns["GPU_COORDINATOR"]
        self.manager.tts = SimpleNamespace(wake_indextts_vllm=AsyncMock(), wake_emotion_vllm=AsyncMock())
        self.manager._indextts_vllm_sleeping = True
        self.manager._emotion_vllm_sleeping = True

    async def test_manual_gpt_wake_reserves_room_for_emotion_too(self):
        self.memory["free_mb"] = 7 * 1024
        with self.assertRaisesRegex(GpuWakeError, "9.50 GiB"):
            await self.manager.wake_engine("indextts_vllm")
        self.manager.tts.wake_indextts_vllm.assert_not_awaited()
        self.manager.tts.wake_emotion_vllm.assert_not_awaited()
        self.assertTrue(self.manager._indextts_vllm_sleeping)
        self.assertIn("9.50 GiB", self.manager.vllm_status()["wake_error"])
        self.memory["free_mb"] = 10 * 1024
        await self.manager.ensure_awake()
        self.assertFalse(self.manager._indextts_vllm_sleeping)
        self.assertFalse(self.manager._emotion_vllm_sleeping)
        self.assertEqual(self.manager.vllm_status()["wake_error"], "")

    async def test_reclaims_translation_memory_before_waking_either_engine(self):
        events = []
        self.ns["_release_translation_gpu_models"].side_effect = lambda: events.append("reclaim")
        self.manager.tts.wake_indextts_vllm.side_effect = lambda: events.append("gpt")
        self.manager.tts.wake_emotion_vllm.side_effect = lambda: events.append("emotion")
        await self.manager.ensure_awake()
        self.assertEqual(events, ["reclaim", "gpt", "emotion"])

    async def test_failed_emotion_wake_preserves_state_and_retry_does_not_remap_gpt(self):
        self.manager.tts.wake_emotion_vllm.side_effect = RuntimeError("out of memory")
        with self.assertRaises(GpuWakeError):
            await self.manager.ensure_awake()
        self.assertFalse(self.manager._indextts_vllm_sleeping)
        self.assertTrue(self.manager._emotion_vllm_sleeping)
        self.manager.tts.wake_emotion_vllm.side_effect = None
        await self.manager.ensure_awake()
        self.manager.tts.wake_indextts_vllm.assert_awaited_once()
        self.assertFalse(self.manager._emotion_vllm_sleeping)

    def test_batch_propagates_wake_failure_instead_of_generating_silence_per_segment(self):
        tree = ast.parse((ROOT / "fastapi_webui_v2_impl.py").read_text(encoding="utf-8-sig"))
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
                     and any(isinstance(handler.type, ast.Name) and handler.type.id == "GpuWakeError"
                             for handler in node.handlers)
                     and any(isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
                             for handler in node.handlers))
        code = ast.parse("try:\n    raise GpuWakeError('no VRAM')\nexcept Exception:\n    pass\n")
        code.body[0].handlers = block.handlers
        with self.assertRaisesRegex(GpuWakeError, "no VRAM"):
            exec(compile(code, "<segment error handling>", "exec"), {"GpuWakeError": GpuWakeError})


class GpuMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_moss_jobs_cannot_reclaim_each_others_translation_models(self):
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def run(*args):
            calls.append(args)
            started.set()
            await release.wait()

        ns = {"Callable": Callable, "GPU_COORDINATOR": GpuWorkCoordinator(enabled=True),
              "_transcription_gpu_lock": asyncio.Lock(), "_run_blocking": run,
              "_prepare_gpu_for_qwen_asr": AsyncMock(), "await_gpu_job": await_gpu_job}
        wrapper = load_definition(ROOT / "fastapi_webui_v2_impl.py", "_run_transcription_gpu_job", ns)
        first = asyncio.create_task(wrapper("moss_transcribe", None, b"first"))
        await started.wait()
        second = asyncio.create_task(wrapper("moss_transcribe", None, b"second"))
        await asyncio.sleep(0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(ns["_prepare_gpu_for_qwen_asr"].await_count, 1)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 2)
        self.assertEqual(len(calls), 2)

    async def test_manual_sleep_waits_for_active_tts_on_small_and_large_profiles(self):
        for enabled in (True, False):
            coordinator = GpuWorkCoordinator(enabled=enabled)
            acquired = asyncio.Event()

            async def maintenance(coordinator=coordinator, acquired=acquired):
                async with coordinator.exclusive(), coordinator.use("index"):
                    acquired.set()

            async with coordinator.use("index"):
                action = asyncio.create_task(maintenance())
                await asyncio.sleep(0)
                self.assertFalse(acquired.is_set())
            await asyncio.wait_for(action, 2)
            self.assertTrue(acquired.is_set())

    async def test_cancelled_thread_holds_gpu_until_inference_really_exits(self):
        coordinator = GpuWorkCoordinator(enabled=True)
        started, release = threading.Event(), threading.Event()

        def worker():
            started.set()
            return release.wait(timeout=5)

        async def job():
            async with coordinator.use("moss"):
                await await_gpu_job(asyncio.to_thread(worker))

        task = asyncio.create_task(job())
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(coordinator._backend, "moss")
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        async with coordinator.exclusive():
            self.assertEqual(coordinator._backend, None)


class HyMtLifecycleTests(unittest.TestCase):
    def test_unload_waits_for_translation_and_drops_all_owning_references(self):
        lock = threading.RLock()
        ns = {"Dict": dict, "Any": Any, "List": list, "_HY_MT_LOCK": lock,
              "_HY_MT_MODEL": object(), "_HY_MT_TOKENIZER": object(), "_HY_MT_MODEL_REF": "hy-mt",
              "gc": SimpleNamespace(collect=Mock()), "torch": None}
        for name in ("hy_mt_model_status", "unload_hy_mt_model", "_translate_batch_with_hy_mt_model"):
            load_definition(ROOT / "whisperx_pipeline.py", name, ns)
        started, release = threading.Event(), threading.Event()

        def translate(*args):
            self.assertIsNotNone(ns["_HY_MT_MODEL"])
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release translation")
            self.assertIsNotNone(ns["_HY_MT_MODEL"])

        ns["_translate_batch_with_hy_mt_model_locked"] = translate
        with ThreadPoolExecutor(max_workers=2) as pool:
            job = pool.submit(ns["_translate_batch_with_hy_mt_model"], ["hello"], "Chinese")
            try:
                self.assertTrue(started.wait(timeout=2))
                unloading = pool.submit(ns["unload_hy_mt_model"])
                self.assertFalse(unloading.done())
            finally:
                release.set()
            job.result(timeout=2)
            self.assertTrue(unloading.result(timeout=2))
        self.assertFalse(ns["hy_mt_model_status"]()["loaded"])
        self.assertIsNone(ns["_HY_MT_TOKENIZER"])
        self.assertIsNone(ns["_HY_MT_MODEL_REF"])

    def test_hy_mt_inventory_and_unload_all_are_connected(self):
        state = {"loaded": True}

        def unload():
            state["loaded"] = False
            return True

        translator = SimpleNamespace(hy_mt_model_status=lambda: state, unload_hy_mt_model=unload)
        ns = {"Dict": dict, "Any": Any, "List": list, "sys": SimpleNamespace(modules={"whisperx_pipeline": translator}),
              "tts_manager": SimpleNamespace(is_ready=lambda: False),
              "confucius_backend_manager": SimpleNamespace(_process_running=lambda: False),
              "indextts25_backend_manager": SimpleNamespace(process_running=lambda: False),
              "stable_audio3_manager": SimpleNamespace(status=lambda: {}, unload=lambda _: []),
              "_voice_design_manager": None, "_enhancement_model": None, "_super_res_model": None,
              "_audio_separator": None, "_audio_separator_runtime_lock": threading.RLock(),
              "_release_cuda_cache": Mock()}
        for name in ("_loaded_model_inventory", "_unload_optional_model_sync"):
            load_definition(ROOT / "fastapi_webui_v2_impl.py", name, ns)
        self.assertEqual(ns["_loaded_model_inventory"]()[0]["key"], "hy_mt")
        self.assertIn("hy_mt", ns["_unload_optional_model_sync"]("all"))
        self.assertEqual(ns["_loaded_model_inventory"](), [])
