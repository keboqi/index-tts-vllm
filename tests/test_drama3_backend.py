import asyncio
import io
import tempfile
import threading
import unittest
import wave
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from indextts_web.config import AppSettings, load_settings
from indextts_web.infrastructure.gpu_work import GpuWorkCoordinator
from indextts_web.services.tts.base import SynthesisRequest
from indextts_web.services.tts.drama3 import Drama3Backend
from indextts_web.services.tts.drama3_manager import ManagedDrama3Backend, emotion_prefix
from indextts_web.services.tts.drama3_serving.requirements import drama3_sglang_install_commands
from tests.support import ROOT, load_definition


class Drama3Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manager = ManagedDrama3Backend(AppSettings(), app_dir=self.root, output_root=self.root)

    def test_payload_prefix_sampling_and_reference(self):
        payload = self.manager.build_payload(
            text="Hello.", emotion_text=" [softly, sadly] ", prompt_wav=str(self.root / "voice.wav"),
            reference_text="A reference.", sampling={"top_k": 0, "top_p": None, "max_new_tokens": 900},
        )
        self.assertEqual(payload["input"], "[softly, sadly] Hello.")
        self.assertEqual(payload["top_k"], 1 << 30)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["max_new_tokens"], 900)
        self.assertEqual(payload["references"][0]["text"], "A reference.")
        self.assertNotIn("seed", payload)
        self.assertNotIn("duration", payload)
        self.assertEqual(emotion_prefix("Hello.", "  "), "Hello.")

    def test_payload_caps_max_new_tokens_at_server_limit(self):
        payload = self.manager.build_payload(text="Hello.", sampling={"max_new_tokens": 4096})
        self.assertEqual(payload["max_new_tokens"], 2048)

    async def test_all_duration_modes_automatically_use_ffmpeg(self):
        manager = SimpleNamespace(synthesize_to_file=AsyncMock(return_value="out.wav"))
        legacy = SimpleNamespace(
            drama3_backend_manager=manager,
            _resolve_external_prompt_audio=AsyncMock(return_value="voice.wav"),
            _postprocess_ffmpeg_duration=AsyncMock(),
        )
        backend = Drama3Backend(legacy)
        for mode in ("original", "ffmpeg"):
            await backend.synthesize(SynthesisRequest(
                text="Hello.", output_path=Path("out.wav"), duration_control=mode,
                target_duration_ms=1600, emotion_text="happy",
            ))
            legacy._postprocess_ffmpeg_duration.assert_awaited_with("out.wav", 1600)
            self.assertEqual(manager.synthesize_to_file.call_args.kwargs["emotion_text"], "happy")
            self.assertNotIn("speech_length", manager.synthesize_to_file.call_args.kwargs)
        legacy._postprocess_ffmpeg_duration.reset_mock()
        await backend.synthesize(SynthesisRequest(text="Hello.", output_path=Path("out.wav")))
        legacy._postprocess_ffmpeg_duration.assert_not_awaited()

    def test_config_and_modal_launch_cover_persistent_isolated_runtime(self):
        settings = load_settings(["--tts_backend", "drama3"], environ={"DRAMA3_PORT": "8097"})
        self.assertEqual(settings.tts_backend, "drama3")
        self.assertEqual(settings.drama3_port, 8097)
        commands = drama3_sglang_install_commands(PurePosixPath("/opt/drama3-sglang-venv"))
        self.assertEqual(commands[1][0], "/opt/drama3-sglang-venv/bin/python")
        self.assertIn("--torch-backend=cu130", commands[2])
        self.assertIn("sglang-omni[fishaudio]", commands[2][-1])
        namespace = {
            "Path": Path, "List": list, "VLLM_PORT": 8000, "DEFAULT_TTS_BACKEND": "index",
            "CONFUCIUS_APP_SUBDIR": "confucius", "CONFUCIUS_PORT": 8001,
            "CONFUCIUS_STARTUP_TIMEOUT": 1200, "CONFUCIUS_REQUEST_TIMEOUT": 900,
            "INDEXTTS25_APP_SUBDIR": "index25", "INDEXTTS25_PORT": 8092,
            "INDEXTTS25_PERSISTENT_MODEL_DIR": "/models/index25", "INDEXTTS25_MODEL_REPO_ID": "index25",
            "INDEXTTS25_PERSISTENT_DATA_DIR": "/cache/index25", "INDEXTTS25_STARTUP_TIMEOUT": 1800,
            "INDEXTTS25_REQUEST_TIMEOUT": 900, "DRAMA3_VENV_DIR": "/opt/drama3-sglang-venv",
            "PERSISTENT_CACHE_DIR": "/persistent_cache", "_build_confucius_start_command": Mock(return_value="confucius"),
            "_build_indextts25_start_command": Mock(return_value="index25"),
        }
        build = load_definition(ROOT / "deploy_vllm_indextts_v2.py", "_build_webui_command", namespace)
        profile = SimpleNamespace(index=SimpleNamespace(gpu_memory_utilization=0.2),
                                  emotion=SimpleNamespace(gpu_memory_utilization=0.1),
                                  confucius=SimpleNamespace(gpu_memory_utilization=0.4),
                                  parallel_segments=100, use_torch_compile=True)
        command = build(Path("/persistent_app"), profile)
        for flag, value in (("--drama3_venv", "/opt/drama3-sglang-venv"),
                            ("--drama3_model_dir", str(Path("/persistent_app/checkpoints/drama3"))),
                            ("--drama3_data_dir", "/persistent_cache/drama3-serving"),
                            ("--drama3_max_parallel_segments", "32")):
            self.assertEqual(command[command.index(flag) + 1], value)
        quick = build(Path("/persistent_app"), profile, startup_tts_backend="none")
        self.assertEqual(quick[quick.index("--tts_backend") + 1], "index")
        self.assertEqual(quick[quick.index("--startup_tts_backend") + 1], "none")

    async def test_sleep_refuses_active_jobs_and_cancel_keeps_gpu_lease(self):
        self.manager.gpu_coordinator = GpuWorkCoordinator(enabled=True)
        self.manager._ensure_ready_locked = AsyncMock()
        entered, finish = threading.Event(), threading.Event()
        def synthesize(**kwargs):
            entered.set()
            finish.wait(5)
            return "out.wav"
        self.manager._synthesize_sync = synthesize
        task = asyncio.create_task(self.manager.synthesize_to_file(text="Hello.", output_path="out.wav"))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            with self.assertRaisesRegex(RuntimeError, "active requests"):
                await self.manager.sleep_vllm()
            task.cancel()
            await asyncio.sleep(0)
            self.assertEqual(self.manager._active_jobs, 1)
            self.assertFalse(task.done())
        finally:
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.manager._active_jobs, 0)

    def test_stages_reference_and_saves_real_wav(self):
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\0\0" * 240)
        reference = self.root / "voice.wav"
        reference.write_bytes(stream.getvalue())
        output = self.root / "output.wav"
        with patch.object(self.manager._server, "speech", return_value=stream.getvalue()) as speech:
            self.manager._synthesize_sync(text="Hello.", output_path=output, prompt_wav=reference)
        staged = Path(speech.call_args.args[0]["references"][0]["audio_path"])
        self.assertTrue(staged.is_file())
        self.assertNotEqual(staged, reference)
        self.assertEqual(output.read_bytes(), stream.getvalue())


if __name__ == "__main__":
    unittest.main()
