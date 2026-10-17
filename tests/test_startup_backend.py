import ast
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from indextts_web.config import load_settings
from indextts_web.services.tts.startup import initialize_backend, restore_backend, startup_backend, warmup_backend
from tests.support import ROOT, load_definition, source_tree


class StartupBackendTests(unittest.IsolatedAsyncioTestCase):
    def legacy(self, backend="drama3"):
        manager = SimpleNamespace(ensure_ready=AsyncMock(), process_running=lambda: True,
                                  status=AsyncMock(return_value={"ready": True}),
                                  _server=SimpleNamespace(checkpoint_launch_enabled=False))
        adapter = SimpleNamespace(manager=manager, synthesize=AsyncMock())
        legacy = SimpleNamespace(
            SETTINGS=load_settings(["--startup_tts_backend", backend], environ={}),
            os=os, GPU_PROFILE=SimpleNamespace(check_startup_memory=Mock()),
            tts_manager=SimpleNamespace(initialize=AsyncMock(return_value=True),
                                        wake_from_snapshot=AsyncMock(), refresh_post_snapshot_state=Mock()),
            SpeakerPresetManager=Mock(), SpeakerAPIWrapper=Mock(), warmup_model=AsyncMock(),
            TTS_BACKEND_REGISTRY=SimpleNamespace(get=Mock(return_value=adapter)),
        )
        return legacy, adapter

    def test_cli_defaults_and_explicit_inference_override(self):
        settings = load_settings([], environ={})
        self.assertEqual(startup_backend(settings), "none")
        self.assertEqual(settings.tts_backend, "index")
        settings = load_settings(["--tts_backend", "drama3"], environ={})
        self.assertEqual((startup_backend(settings), settings.tts_backend), ("none", "drama3"))
        settings = load_settings(["--startup_tts_backend", "drama3"], environ={})
        self.assertEqual((startup_backend(settings), settings.tts_backend), ("drama3", "drama3"))
        settings = load_settings(["--tts_backend", "index", "--startup_tts_backend", "drama3"], environ={})
        self.assertEqual((startup_backend(settings), settings.tts_backend), ("drama3", "index"))
        settings = load_settings([], environ={"STARTUP_TTS_BACKEND": "drama3"})
        self.assertEqual(settings.tts_backend, "drama3")

    async def test_drama_startup_loads_only_drama_and_initializes_reference_presets(self):
        legacy, adapter = self.legacy()
        await initialize_backend(legacy)
        legacy.tts_manager.initialize.assert_not_awaited()
        legacy.GPU_PROFILE.check_startup_memory.assert_not_called()
        legacy.warmup_model.assert_not_awaited()
        adapter.manager.ensure_ready.assert_awaited_once()
        legacy.SpeakerPresetManager.assert_called_once_with()
        self.assertIs(legacy.speaker_api, legacy.SpeakerAPIWrapper.return_value)

    async def test_default_startup_loads_no_backend_and_initializes_reference_presets(self):
        legacy, adapter = self.legacy("none")
        await initialize_backend(legacy)
        await warmup_backend(legacy)
        await restore_backend(legacy)
        legacy.tts_manager.initialize.assert_not_awaited()
        legacy.tts_manager.wake_from_snapshot.assert_not_awaited()
        legacy.GPU_PROFILE.check_startup_memory.assert_not_called()
        legacy.warmup_model.assert_not_awaited()
        adapter.manager.ensure_ready.assert_not_awaited()
        adapter.synthesize.assert_not_awaited()
        legacy.SpeakerPresetManager.assert_called_once_with()
        self.assertIs(legacy.speaker_api, legacy.SpeakerAPIWrapper.return_value)

    async def test_index_startup_keeps_existing_initialization(self):
        legacy, adapter = self.legacy("index")
        await initialize_backend(legacy)
        legacy.tts_manager.initialize.assert_awaited_once()
        legacy.GPU_PROFILE.check_startup_memory.assert_called_once()
        adapter.manager.ensure_ready.assert_not_awaited()

    async def test_drama_snapshot_warmup_generates_audio_and_cleans_it(self):
        legacy, adapter = self.legacy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "examples").mkdir()
            (root / "examples/voice_01.wav").write_bytes(b"reference")
            legacy.current_dir = root
            legacy.ROOT_OUTPUT_DIR = root
            adapter.synthesize.side_effect = lambda request: request.output_path.write_bytes(b"audio" * 20)
            await warmup_backend(legacy)
            request = adapter.synthesize.call_args.args[0]
            self.assertEqual(request.prompt_audio, str(root / "examples/voice_01.wav"))
            self.assertFalse(request.output_path.exists())
            legacy.warmup_model.assert_not_awaited()
            adapter.synthesize.side_effect = None
            with self.assertRaisesRegex(RuntimeError, "no audio"):
                await warmup_backend(legacy)

    async def test_drama_restore_verifies_ready_pipeline_without_cold_start(self):
        legacy, adapter = self.legacy()
        await restore_backend(legacy)
        adapter.manager.ensure_ready.assert_not_awaited()
        self.assertTrue(adapter.manager._server.checkpoint_launch_enabled)
        legacy.tts_manager.wake_from_snapshot.assert_not_awaited()
        legacy.tts_manager.refresh_post_snapshot_state.assert_called_once()
        adapter.manager.status.return_value = {"ready": False}
        with self.assertRaisesRegex(RuntimeError, "rebuild the snapshot"):
            await restore_backend(legacy)
        adapter.manager.ensure_ready.assert_not_awaited()

    def test_modal_endpoints_have_separate_gpu_snapshots_and_unique_labels(self):
        tree = source_tree(ROOT / "deploy_vllm_indextts_v2.py")
        labels = []
        for name in ("IndexTTSVllmServer", "Drama3SglangServer"):
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            decorator = next(item for item in cls.decorator_list
                             if isinstance(item, ast.Call) and getattr(item.func, "attr", "") == "cls")
            options = {item.arg: item.value for item in decorator.keywords}
            self.assertIs(ast.literal_eval(options["enable_memory_snapshot"]), True)
            self.assertEqual(ast.literal_eval(options["experimental_options"]), {"enable_gpu_snapshot": True})
            serve = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "serve")
            labels.append(ast.literal_eval(next(item.value for item in serve.decorator_list[0].keywords
                                                if item.arg == "label")))
        self.assertEqual(labels, ["tts", "tts-drama3"])

    def test_drama_modal_capture_warms_pipeline_without_sleeping(self):
        events = []
        builder = Mock(return_value=["python", "webui.py", "--drama3_data_dir", "/persistent_cache/drama3-serving"])
        popen = Mock()
        namespace = {"print": Mock(), "os": os, "subprocess": SimpleNamespace(Popen=popen),
                     "_configure_persistent_runtime": lambda **kwargs: ROOT, "_build_webui_command": builder,
                     "_prepare_drama3_snapshot_runtime": lambda: Path("/tmp/drama3-serving"),
                     "_persist_drama3_snapshot_runtime": lambda: events.append("persist"),
                     "_commit_snapshot_volumes": lambda phase: events.append(phase),
                     "_wait_ready": Mock(), "_call_local_json": lambda path, **kwargs: events.append(path),
                     "SNAPSHOT_REQUEST_TIMEOUT": 900, "SNAPSHOT_STARTUP_TIMEOUT": 1800}
        start = load_definition(ROOT / "deploy_vllm_indextts_v2.py", "Drama3SglangServer.start", namespace)
        start(SimpleNamespace())
        builder.assert_called_once_with(ROOT, startup_tts_backend="drama3")
        self.assertEqual(popen.call_args.kwargs["env"]["DRAMA3_MODAL_SNAPSHOT_CAPTURE"], "1")
        self.assertEqual(events, ["before model startup", "/internal/snapshot/warmup", "persist", "before snapshot capture"])

    def test_quick_start_endpoint_preloads_no_backend_without_snapshot(self):
        tree = source_tree(ROOT / "deploy_vllm_indextts_v2.py")
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QuickStartServer")
        decorator = next(item for item in cls.decorator_list
                         if isinstance(item, ast.Call) and getattr(item.func, "attr", "") == "cls")
        options = {item.arg for item in decorator.keywords}
        self.assertFalse(options & {"enable_memory_snapshot", "experimental_options"})
        serve = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "serve")
        self.assertEqual(ast.literal_eval(next(item.value for item in serve.decorator_list[0].keywords
                                               if item.arg == "label")), "tts-quick")

        calls = []
        builder = Mock(return_value=["python", "webui.py"])
        namespace = {"print": Mock(), "os": os, "subprocess": SimpleNamespace(Popen=Mock()),
                     "_configure_persistent_runtime": Mock(return_value=ROOT), "_build_webui_command": builder,
                     "_commit_snapshot_volumes": lambda phase: calls.append(phase),
                     "_wait_ready": Mock(), "_call_local_json": lambda path, **kwargs: calls.append(path),
                     "SNAPSHOT_STARTUP_TIMEOUT": 1800}
        start = load_definition(ROOT / "deploy_vllm_indextts_v2.py", "QuickStartServer.start", namespace)
        start(SimpleNamespace())
        namespace["_configure_persistent_runtime"].assert_called_once_with(startup_tts_backend="none")
        builder.assert_called_once_with(ROOT, startup_tts_backend="none")
        self.assertEqual(calls, [])

    def test_external_reference_presets_do_not_require_index_embeddings(self):
        namespace = {"os": os, "time": time}
        add = load_definition(ROOT / "speaker_preset_manager.py", "SpeakerPresetManager.add_speaker_preset", namespace)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "voice.wav"
            audio.write_bytes(b"reference")
            manager = SimpleNamespace(tts_model=None, presets={}, _memory_cache={}, _save_presets=Mock(),
                                      _get_audio_hash=Mock(return_value="hash"),
                                      _get_cache_path=Mock(return_value=Path(directory) / "voice.cache"), logger=Mock())
            self.assertTrue(add(manager, "voice", str(audio), "reference"))
            self.assertTrue(manager.presets["voice"]["reference_only"])
            self.assertEqual(manager.presets["voice"]["audio_path"], str(audio))
            manager._save_presets.assert_called_once()

    def test_snapshot_runtime_copies_kernels_and_logs_without_user_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "runtime"
            persistent = root / "volume" / "drama3-serving"
            for name in ("cache", "logs", "media"):
                (persistent / name).mkdir(parents=True)
            (persistent / "cache/kernel.bin").write_bytes(b"cached kernel")
            (persistent / "logs/old.log").write_text("old log")
            (persistent / "media/user.wav").write_bytes(b"private reference")
            namespace = {"Path": Path, "DRAMA3_SNAPSHOT_DATA_DIR": str(local),
                         "PERSISTENT_CACHE_DIR": str(root / "volume")}
            prepare = load_definition(ROOT / "deploy_vllm_indextts_v2.py", "_prepare_drama3_snapshot_runtime", namespace)
            persist = load_definition(ROOT / "deploy_vllm_indextts_v2.py", "_persist_drama3_snapshot_runtime", namespace)
            self.assertEqual(prepare(), local)
            self.assertEqual((local / "cache/kernel.bin").read_bytes(), b"cached kernel")
            self.assertEqual(list((local / "media").iterdir()), [])
            self.assertEqual(list((local / "logs").iterdir()), [])
            (local / "logs/current.log").write_text("warmup completed")
            (local / "media/new.wav").write_bytes(b"another reference")
            persist()
            self.assertEqual((persistent / "logs/current.log").read_text(), "warmup completed")
            self.assertFalse((persistent / "media/new.wav").exists())
            self.assertEqual((persistent / "media/user.wav").read_bytes(), b"private reference")


if __name__ == "__main__":
    unittest.main()
