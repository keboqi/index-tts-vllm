import asyncio
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from indextts_web.config import load_settings
from indextts_web.gpu_profiles import (
    GIB,
    PROFILE_ENV,
    EngineProfile,
    GpuInfo,
    GpuProfile,
    resolve_gpu_profile,
    runtime_gpu_profile,
    write_omni_deploy_config,
)
from indextts_web.infrastructure.concurrency import ConcurrencyBudget
from indextts_web.infrastructure.gpu import probe_gpu
from indextts_web.infrastructure.gpu_work import GpuWorkCoordinator, gpu_operation
from indextts_web.services.tts.confucius_launcher import install_engine_options


def gpu(gib=24, free=None):
    return GpuInfo("test GPU", int(gib * GIB), int((gib if free is None else free) * GIB), "8.9")


class GpuProfileTests(unittest.TestCase):
    def test_capacity_boundaries_and_reported_capacity(self):
        for capacity, expected in ((22.49, "24gb"), (31.99, "24gb"), (32, "48gb"),
                                   (44.5, "48gb"), (63.99, "48gb"), (64, "96gb"), (95, "96gb")):
            with self.subTest(capacity=capacity):
                self.assertEqual(resolve_gpu_profile(gpu(capacity), {}).name, expected)
        with self.assertRaisesRegex(ValueError, "24 GB-class"):
            resolve_gpu_profile(gpu(16), {})

    def test_small_budget_uses_total_memory_not_free_memory(self):
        profile = resolve_gpu_profile(gpu(22.49, free=21), {})
        self.assertAlmostEqual(profile.index.gpu_memory_utilization * profile.gpu.total_gib, 6)
        self.assertAlmostEqual(profile.emotion.gpu_memory_utilization * profile.gpu.total_gib, 3)
        self.assertTrue(profile.index.enforce_eager)
        self.assertEqual(profile.index_concurrency, 1)
        self.assertFalse(profile.use_torch_compile)

    def test_medium_limits_and_high_modal_compatibility(self):
        medium = resolve_gpu_profile(gpu(48), {}, modal=True)
        self.assertEqual((medium.index.max_num_seqs, medium.emotion.max_num_seqs), (16, 4))
        self.assertEqual(medium.parallel_segments, 4)
        high = resolve_gpu_profile(gpu(96), {}, modal=True)
        self.assertEqual(high.index.kwargs(), {"gpu_memory_utilization": 0.15})
        self.assertEqual(high.emotion.kwargs(), {"gpu_memory_utilization": 0.05, "max_model_len": 2048})
        self.assertEqual(high.confucius.gpu_memory_utilization, 0.20)
        self.assertTrue(high.use_torch_compile)
        self.assertEqual(high.index_concurrency, 100)
        self.assertFalse(resolve_gpu_profile(gpu(96), {}, modal=False).use_torch_compile)

    def test_explicit_overrides_and_cli_precedence(self):
        env = {"GPU_MEMORY_UTILIZATION": "0.3", "QWENEMO_VLLM_MAX_NUM_SEQS": "2",
               "INDEXTTS_VLLM_ENFORCE_EAGER": "false", "INDEXTTS_GPU_WORK_CONCURRENCY": "3",
               "TRANSLATION_TTS_CONCURRENCY": "8", "INDEXTTS_USE_TORCH_COMPILE": "true"}
        profile = resolve_gpu_profile(gpu(), env)
        settings = load_settings(["--gpu_memory_utilization", "0.28", "--no-use_torch_compile"], environ=env)
        profile = profile.with_settings(settings)
        resolved = profile.apply_settings(settings)
        self.assertEqual(resolved.gpu_memory_utilization, 0.28)
        self.assertEqual(profile.emotion.max_num_seqs, 2)
        self.assertFalse(profile.index.enforce_eager)
        self.assertFalse(resolved.use_torch_compile)
        self.assertEqual(profile.translation_concurrency, 3)

    def test_unspecified_settings_stay_auto_until_runtime(self):
        settings = load_settings([], environ={})
        for attr in ("gpu_memory_utilization", "qwenemo_gpu_memory_utilization",
                     "confucius_vllm_gpu_memory_utilization", "use_torch_compile",
                     "indextts25_max_parallel_segments"):
            self.assertIsNone(getattr(settings, attr))

    def test_bad_overrides_fail_instead_of_silently_clamping(self):
        for variable, value in (("GPU_MEMORY_UTILIZATION", "nan"), ("GPU_MEMORY_UTILIZATION", "1"),
                                ("GPU_MEMORY_UTILIZATION", "0"), ("INDEXTTS_VLLM_MAX_NUM_SEQS", "0"),
                                ("QWENEMO_VLLM_ENFORCE_EAGER", "maybe")):
            with self.subTest(variable=variable, value=value), self.assertRaises(ValueError):
                resolve_gpu_profile(gpu(), {variable: value})

    def test_profile_transport_and_stable_cache_key(self):
        profile = resolve_gpu_profile(gpu(), {})
        self.assertEqual(GpuProfile.from_json(profile.to_json()), profile)
        other = replace(profile, gpu=replace(profile.gpu, free_bytes=10 * GIB))
        self.assertEqual(other.cache_key, profile.cache_key)
        self.assertNotEqual(replace(profile, index_concurrency=2).cache_key, profile.cache_key)
        with patch.dict(os.environ, {PROFILE_ENV: profile.to_json()}, clear=True), \
                patch("indextts_web.infrastructure.gpu.probe_gpu") as probe:
            self.assertEqual(runtime_gpu_profile(), profile)
            probe.assert_not_called()
            with self.assertRaisesRegex(ValueError, "targets"):
                runtime_gpu_profile(device="cuda:1")
        values = json.loads(profile.to_json())
        values["version"] = -1
        with self.assertRaisesRegex(ValueError, "schema"):
            GpuProfile.from_json(json.dumps(values))

    def test_memory_preflight_and_invalid_reserves(self):
        resolve_gpu_profile(gpu(22.49), {}).check_startup_memory()
        with self.assertRaisesRegex(RuntimeError, "only 12.00"):
            resolve_gpu_profile(gpu(24, 12), {}).check_startup_memory()
        with self.assertRaises(RuntimeError):
            resolve_gpu_profile(gpu(), {"GPU_MEMORY_UTILIZATION": "0.8"}).check_startup_memory()
        with self.assertRaises(ValueError):
            resolve_gpu_profile(gpu(), {}).check_startup_memory(non_vllm_gib=float("nan"))

    def test_checkpoint_context_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(json.dumps({"n_positions": 4096}))
            options = resolve_gpu_profile(gpu(), {}).index.kwargs(model)
            self.assertEqual(options["max_num_batched_tokens"], 4096)
            self.assertNotIn("max_model_len", options)
        with self.assertRaises(ValueError):
            EngineProfile(-0.1)

    def test_probe_uses_isolated_process_and_inherits_device_mapping(self):
        report = gpu()
        result = SimpleNamespace(stdout="library message\nINDEXTTS_GPU=" + json.dumps({
            "name": report.name, "total_bytes": report.total_bytes, "free_bytes": report.free_bytes,
            "capability": report.capability, "device": "cuda:1"}))
        with patch("subprocess.run", return_value=result) as run:
            info = probe_gpu(device="cuda:1")
        self.assertEqual(info.device, "cuda:1")
        self.assertEqual(run.call_args.args[0][-1], "cuda:1")
        self.assertNotIn("env", run.call_args.kwargs)  # Inherit CUDA_VISIBLE_DEVICES unchanged.
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "probe", stderr="CUDA missing")), \
                self.assertRaisesRegex(RuntimeError, "CUDA missing"):
            probe_gpu()

    def test_confucius_adapter_forwards_options_without_overwriting_explicit_values(self):
        calls = []

        class Runtime:
            def __init__(self, *, gpu_memory_utilization, engine_kwargs=None):
                calls.append((gpu_memory_utilization, engine_kwargs))

        install_engine_options(Runtime, resolve_gpu_profile(gpu(), {}).confucius.kwargs())
        Runtime(gpu_memory_utilization=0.4, engine_kwargs={"max_num_seqs": 2})
        self.assertEqual(calls, [(0.4, {"max_num_seqs": 2, "enforce_eager": True})])


@unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML required for deployment config tests")
class OmniProfileTests(unittest.TestCase):
    def test_stage_limits_and_unrelated_config_preservation(self):
        import yaml

        # JSON is valid YAML; test against a self-contained version of the
        # backend contract rather than depending on a sibling checkout.
        base = {"pipeline": "indextts2_5", "connectors": {"shm": {"name": "SharedMemoryConnector"}},
                "stages": [
                    {"stage_id": 0, "attention_backend": "TRITON_ATTN", "max_model_len": 2560,
                     "max_num_seqs": 32, "gpu_memory_utilization": 0.3, "enable_chunked_prefill": False,
                     "default_sampling_params": {"max_tokens": 1500}},
                    {"stage_id": 1, "max_num_seqs": 16, "gpu_memory_utilization": 0.3,
                     "max_model_len": 32768, "max_num_batched_tokens": 8192, "enforce_eager": True,
                     "hf_overrides": {"s2mel_cfm_batch_size": 16, "diffusion_steps": 25,
                                      "s2mel_dit_torch_compile": True, "s2mel_vocoder_torch_compile": True}}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "base.yaml"
            source.write_text(json.dumps(base))
            for capacity, batches in ((24, (4, 1)), (48, (16, 4)), (96, (32, 16))):
                with self.subTest(capacity=capacity):
                    profile = resolve_gpu_profile(gpu(capacity), {}, modal=True)
                    target = write_omni_deploy_config(source, root / "generated", profile)
                    config = yaml.safe_load(target.read_text())
                    ar, mel = config["stages"]
                    self.assertEqual((ar["max_num_seqs"], mel["max_num_seqs"]), batches)
                    self.assertEqual(mel["hf_overrides"]["s2mel_cfm_batch_size"], batches[1])
                    self.assertEqual(config["connectors"], base["connectors"])
                    self.assertEqual(ar["default_sampling_params"], base["stages"][0]["default_sampling_params"])
                    self.assertEqual(ar["attention_backend"], "TRITON_ATTN")
                    self.assertEqual((ar["max_model_len"], mel["max_model_len"]), (2560, 32768))
                    self.assertEqual(mel["max_num_batched_tokens"], 8192)
                    self.assertEqual(mel["hf_overrides"]["s2mel_dit_torch_compile"], capacity == 96)
                    self.assertEqual(target, write_omni_deploy_config(source, root / "generated", profile))
                    if capacity == 96:
                        self.assertEqual(target, source)
            self.assertEqual(json.loads(source.read_text()), base)


class GpuAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_other_backend_waits_for_active_work_and_reentry_is_safe(self):
        coordinator = GpuWorkCoordinator(enabled=True)
        entered = asyncio.Event()
        release = asyncio.Event()
        other_entered = asyncio.Event()

        async def index_work():
            async with coordinator.use("index"), coordinator.use("index"):
                entered.set()
                await release.wait()

        async def confucius_work():
            async with coordinator.use("confucius"):
                other_entered.set()

        index_task = asyncio.create_task(index_work())
        await entered.wait()
        other_task = asyncio.create_task(confucius_work())
        await asyncio.sleep(0)
        self.assertFalse(other_entered.is_set())
        release.set()
        await asyncio.gather(index_task, other_task)
        self.assertTrue(other_entered.is_set())

    async def test_stream_holds_backend_until_closed_and_cancellation_releases(self):
        coordinator = GpuWorkCoordinator(enabled=True)

        class Engine:
            gpu_coordinator = coordinator

            @gpu_operation("index")
            async def stream(self):
                yield b"first chunk"
                yield b"second chunk"

        chunks = Engine().stream()
        self.assertEqual(await anext(chunks), b"first chunk")
        other_started = asyncio.Event()

        async def other():
            async with coordinator.use("index25"):
                other_started.set()

        task = asyncio.create_task(other())
        await asyncio.sleep(0)
        self.assertFalse(other_started.is_set())
        await chunks.aclose()
        await task
        self.assertTrue(other_started.is_set())

        started = asyncio.Event()

        async def cancellable():
            async with coordinator.use("index"):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(cancellable())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        async with coordinator.use("confucius"):
            self.assertEqual(coordinator._backend, "confucius")

    async def test_same_backend_can_run_concurrently(self):
        coordinator = GpuWorkCoordinator(enabled=True)
        both_entered = asyncio.Event()
        count = 0

        async def work():
            nonlocal count
            async with coordinator.use("index"):
                count += 1
                if count == 2:
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), 1)

        await asyncio.gather(work(), work())

    async def test_startup_profile_bounds_actual_work(self):
        budget = ConcurrencyBudget()
        profile = resolve_gpu_profile(gpu(), {})
        budget.configure_gpu_limits(profile.index_concurrency, profile.translation_concurrency)
        running = peak = 0

        async def work():
            nonlocal running, peak
            async with budget.index_tts:
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0)
                running -= 1

        try:
            await asyncio.gather(*(work() for _ in range(8)))
            self.assertEqual(peak, 1)
        finally:
            budget.shutdown()
