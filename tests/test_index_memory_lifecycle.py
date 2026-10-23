import ast
import asyncio
import os
import threading
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

from indextts_web.gpu_profiles import resolve_gpu_profile
from indextts_web.infrastructure.gpu_work import GpuWorkCoordinator, await_gpu_job, gpu_operation
from indextts_web.infrastructure.vllm_memory import GpuWakeError
from tests.support import ROOT, gpu, load_definition, source_tree


class IndexWakeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.memory = {"available": True, "total_mb": 24 * 1024, "free_mb": 10 * 1024}
        self.ns = {
            "asyncio": asyncio, "Dict": dict, "Any": Any, "gpu_operation": gpu_operation,
            "GpuWakeError": GpuWakeError, "GPU_COORDINATOR": GpuWorkCoordinator(enabled=True),
            "GPU_PROFILE": resolve_gpu_profile(gpu(), {}), "os": os,
            "drama3_backend_manager": SimpleNamespace(sleep_vllm=AsyncMock(), process_running=lambda: False),
            "indextts25_backend_manager": SimpleNamespace(process_running=lambda: False),
            "confucius_backend_manager": SimpleNamespace(_process_running=lambda: False),
            "_release_translation_gpu_models": AsyncMock(), "_release_cuda_cache": Mock(),
            "_run_blocking": AsyncMock(), "_cuda_memory_summary": lambda: self.memory,
        }
        manager_type = load_definition(ROOT / "fastapi_webui_v2_impl.py", "TTSManager", self.ns)
        self.manager = manager_type()
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
        self.memory["free_mb"] = 7 * 1024

        def reclaim(**kwargs):
            events.append("reclaim")
            self.memory["free_mb"] = 10 * 1024

        self.ns["_release_translation_gpu_models"].side_effect = reclaim
        self.manager.tts.wake_indextts_vllm.side_effect = lambda: events.append("gpt")
        self.manager.tts.wake_emotion_vllm.side_effect = lambda: events.append("emotion")
        await self.manager.ensure_awake()
        self.assertEqual(events, ["reclaim", "gpt", "emotion"])
        self.ns["_release_translation_gpu_models"].assert_awaited_once_with(required_mb=9.5 * 1024)

    async def test_sufficient_memory_preserves_translation_models_and_allocator_cache(self):
        await self.manager.ensure_awake()
        self.ns["_release_translation_gpu_models"].assert_not_awaited()
        self.ns["_run_blocking"].assert_not_awaited()

    async def test_allocator_cache_is_reclaimed_before_unloading_models(self):
        self.memory["free_mb"] = 9 * 1024

        def release_cache(*args):
            self.memory["free_mb"] = 10 * 1024

        self.ns["_run_blocking"].side_effect = release_cache
        await self.manager.ensure_awake()
        self.ns["_run_blocking"].assert_awaited_once_with(self.ns["_release_cuda_cache"])
        self.ns["_release_translation_gpu_models"].assert_not_awaited()

    async def test_large_gpu_does_not_reclaim_models_without_exclusive_backend_lease(self):
        self.ns["GPU_COORDINATOR"].enabled = False
        self.memory["free_mb"] = 7 * 1024
        with self.assertRaises(GpuWakeError):
            await self.manager.ensure_awake()
        self.ns["_release_translation_gpu_models"].assert_not_awaited()

    async def test_recovered_telemetry_still_reserves_both_engines(self):
        self.memory = {"available": False}

        def release_cache(*args):
            self.memory = {"available": True, "total_mb": 24 * 1024, "free_mb": 7 * 1024}

        self.ns["_run_blocking"].side_effect = release_cache
        with self.assertRaisesRegex(GpuWakeError, "9.50 GiB"):
            await self.manager.ensure_awake()
        self.manager.tts.wake_indextts_vllm.assert_not_awaited()
        self.ns["_release_translation_gpu_models"].assert_awaited_once_with(required_mb=9.5 * 1024)

    async def test_lazy_initialization_checks_full_model_budget(self):
        self.manager._initialized = False
        self.manager._indextts_vllm_sleeping = False
        self.manager._emotion_vllm_sleeping = False
        self.manager.initialize = AsyncMock()
        with self.assertRaisesRegex(GpuWakeError, "19.40 GiB"):
            await self.manager.ensure_awake()
        self.manager.initialize.assert_not_awaited()

    async def test_index_initializes_lazily_after_external_backend_startup(self):
        self.memory["free_mb"] = 24 * 1024
        self.manager._initialized = False
        self.manager._indextts_vllm_sleeping = False
        self.manager._emotion_vllm_sleeping = False

        async def initialize():
            self.manager._initialized = True
            return True

        self.manager.initialize = AsyncMock(side_effect=initialize)
        await self.manager.ensure_awake()
        self.manager.initialize.assert_awaited_once()
        self.ns["drama3_backend_manager"].sleep_vllm.assert_awaited_once()
        self.assertTrue(self.manager.is_ready())

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
        tree = source_tree(ROOT / "fastapi_webui_v2_impl.py")
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
                     and any(isinstance(handler.type, ast.Name) and handler.type.id == "GpuWakeError"
                             for handler in node.handlers)
                     and any(isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
                             for handler in node.handlers))
        code = ast.parse("try:\n    raise GpuWakeError('no VRAM')\nexcept Exception:\n    pass\n")
        code.body[0].handlers = block.handlers
        with self.assertRaisesRegex(GpuWakeError, "no VRAM"):
            exec(compile(code, "<segment error handling>", "exec"), {"GpuWakeError": GpuWakeError})


class TranslationReclamationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.free = 0
        self.client = SimpleNamespace(
            inventory=AsyncMock(return_value=[{"state": "loaded"}]), change=AsyncMock(),
        )
        self.unload = Mock()
        self.blocking = AsyncMock()
        self.reclaim = load_definition(ROOT / "fastapi_webui_v2_impl.py", "_release_translation_gpu_models", {
            "moss_model_client": self.client, "_run_blocking": self.blocking,
            "_unload_optional_model_sync": self.unload,
            "_cuda_memory_summary": lambda: {"available": True, "free_mb": self.free},
        })

    async def test_does_not_even_query_models_when_memory_is_sufficient(self):
        self.free = 100
        await self.reclaim(required_mb=100)
        self.client.inventory.assert_not_awaited()
        self.blocking.assert_not_awaited()

    async def test_stops_after_sleeping_moss_when_that_is_enough(self):
        def sleep(*args):
            self.free = 100

        self.client.change.side_effect = sleep
        await self.reclaim(required_mb=100)
        self.client.change.assert_awaited_once_with("sleep")
        self.blocking.assert_not_awaited()

    async def test_preserves_separator_when_unloading_translator_is_enough(self):
        def unload(*args):
            self.free = 100

        self.blocking.side_effect = unload
        await self.reclaim(required_mb=100)
        self.blocking.assert_awaited_once_with(self.unload, "hy_mt")

    async def test_no_target_keeps_existing_full_reclaim_behavior(self):
        self.free = 100
        await self.reclaim()
        self.client.change.assert_awaited_once_with("sleep")
        self.assertEqual([call.args[1] for call in self.blocking.await_args_list], ["hy_mt", "audio_separator"])


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
              "drama3_backend_manager": SimpleNamespace(sleep_vllm=AsyncMock(), process_running=lambda: False),
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
