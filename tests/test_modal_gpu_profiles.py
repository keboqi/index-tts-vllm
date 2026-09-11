import ast
import importlib.util
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from indextts_web.gpu_profiles import resolve_gpu_profile
from indextts_web.infrastructure.modal_runtime import PERSISTENT_DIRECTORIES, prepare_runtime_code
from tests.test_gpu_profiles import gpu

ROOT = Path(__file__).resolve().parents[1]


def load_definition(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    if "." in name:
        class_name, name = name.split(".", 1)
        tree = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == class_name)
    definition = next(node for node in ast.walk(tree)
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    definition.decorator_list = []
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@unittest.skipUnless(importlib.util.find_spec("modal") and importlib.util.find_spec("yaml"),
                     "Modal SDK and PyYAML required for Modal command tests")
class ModalGpuCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import deploy_vllm_indextts_v2

        cls.deploy = deploy_vllm_indextts_v2

    def test_all_three_gpu_commands_without_operator_tuning(self):
        import yaml

        deploy = self.deploy
        with tempfile.TemporaryDirectory(prefix="gpu profile ") as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            config_dir = root / deploy.INDEXTTS25_APP_SUBDIR / "vllm_omni" / "deploy"
            config_dir.mkdir(parents=True)
            base = config_dir / "indextts2_5.yaml"
            base.write_text(json.dumps({"stages": [
                {"stage_id": 0, "max_num_seqs": 32, "gpu_memory_utilization": 0.3},
                {"stage_id": 1, "max_num_seqs": 16, "gpu_memory_utilization": 0.3,
                 "hf_overrides": {"s2mel_cfm_batch_size": 16}}]}))
            with patch.object(deploy, "INDEXTTS25_PERSISTENT_DATA_DIR", str(root / "data")):
                for size, batches, segments in ((24, (4, 1), 1), (48, (16, 4), 4), (96, (32, 16), 100)):
                    with self.subTest(size=size):
                        profile = resolve_gpu_profile(gpu(size), {}, modal=True)
                        command = deploy._build_webui_command(root, profile)
                        self.assertEqual(command[:3], ["python", "-u", "fastapi_webui_v2.py"])
                        index_budget = float(command[command.index("--gpu_memory_utilization") + 1])
                        emotion_budget = float(command[command.index("--qwenemo_gpu_memory_utilization") + 1])
                        self.assertEqual(index_budget, profile.index.gpu_memory_utilization)
                        self.assertEqual(emotion_budget, profile.emotion.gpu_memory_utilization)
                        self.assertEqual(command[command.index("--indextts25_max_parallel_segments") + 1], str(segments))
                        self.assertIn("--use_torch_compile", command)
                        conf = shlex.split(command[command.index("--confucius_start_command") + 1])
                        self.assertIn("indextts_web.services.tts.confucius_launcher", conf)
                        self.assertIn(f"PYTHONPATH={root / deploy.CONFUCIUS_APP_SUBDIR}{os.pathsep}{root}", conf)
                        self.assertIn("--compile-s2a", conf)
                        self.assertEqual(conf[conf.index("--warmup-mode") + 1],
                                         "background" if size == 96 else "foreground")
                        self.assertEqual(float(conf[conf.index("--vllm-gpu-memory-utilization") + 1]),
                                         profile.confucius.gpu_memory_utilization)
                        omni = shlex.split(command[command.index("--indextts25_start_command") + 1])
                        self.assertIn(deploy.INDEXTTS25_VLLM, omni)
                        self.assertIn("--enable-sleep-mode", omni)
                        config = yaml.safe_load(Path(omni[omni.index("--deploy-config") + 1]).read_text())
                        self.assertEqual(tuple(stage["max_num_seqs"] for stage in config["stages"]), batches)

    def test_explicit_omni_config_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "custom.yaml"
            custom.write_text("stages: []")
            with patch.dict(os.environ, {"INDEXTTS25_DEPLOY_CONFIG": str(custom)}, clear=True):
                command = shlex.split(self.deploy._build_indextts25_start_command(
                    Path(directory), resolve_gpu_profile(gpu(), {})))
            self.assertEqual(command[command.index("--deploy-config") + 1], str(custom))
            self.assertEqual(custom.read_text(), "stages: []")

    def test_readiness_requires_loaded_model(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(self.deploy, "_call_local_json", side_effect=[{"ready": False}, {"ready": True}]) as call, \
                patch.object(self.deploy.socket, "create_connection"), patch.object(self.deploy.time, "sleep"):
            self.deploy._wait_ready(process, timeout_seconds=5)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args.args[0], "/health")
        process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "exited"):
            self.deploy._wait_ready(process, timeout_seconds=5)

    def test_source_bundle_excludes_local_data_and_credentials(self):
        for relative in (".env", ".env.example", ".git/config", ".venv/qa-packages/modal.py",
                         "checkpoints/config.json", "speaker_presets/presets.json", "outputs/audio.wav",
                         "deploy_voice_design_modal.py", "indextts/__pycache__/infer.pyc"):
            with self.subTest(relative=relative):
                self.assertTrue(self.deploy._ignore_runtime_source(ROOT / relative))
        for relative in ("indextts_web/gpu_profiles.py", "indextts/infer_vllm_v2.py",
                         "fastapi_webui_v2_impl.py", "index_new.html", "examples/voice_01.wav"):
            with self.subTest(relative=relative):
                self.assertFalse(self.deploy._ignore_runtime_source(ROOT / relative))


class RuntimeCodeTests(unittest.TestCase):
    def test_deployed_code_does_not_use_or_replace_stale_volume_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, persistent, destination = root / "source", root / "persistent", root / "runtime"
            source.mkdir()
            persistent.mkdir()
            (source / "app.py").write_text("new code")
            (persistent / "app.py").write_text("old code")
            for name in PERSISTENT_DIRECTORIES:
                (persistent / name).mkdir()
            (persistent / "outputs" / "existing.wav").write_bytes(b"existing audio")
            links = []

            def fake_link(path, target, target_is_directory=False):
                links.append((path, target, target_is_directory))

            # Windows test hosts need elevated rights to create symlinks. The
            # path/ownership contract is testable without requiring that right.
            with patch.object(Path, "symlink_to", fake_link):
                result = prepare_runtime_code(source, persistent, destination)
            self.assertEqual(result, destination)
            self.assertEqual((result / "app.py").read_text(), "new code")
            self.assertEqual((persistent / "app.py").read_text(), "old code")
            self.assertEqual((persistent / "outputs" / "existing.wav").read_bytes(), b"existing audio")
            self.assertEqual({link[0].name for link in links}, set(PERSISTENT_DIRECTORIES))
            self.assertTrue(all(target.parent == persistent for _, target, _ in links))


class EngineConstructionTests(unittest.TestCase):
    def test_emotion_factory_receives_resolved_options(self):
        for size in (24, 48, 96):
            with self.subTest(size=size):
                profile = resolve_gpu_profile(gpu(size), {}, modal=True)
                factory = Mock(return_value="engine")
                namespace = {"AutoTokenizer": SimpleNamespace(from_pretrained=Mock()),
                             "_vllm_sleep_mode_enabled": lambda: True,
                             "AsyncEngineArgs": lambda **kwargs: kwargs,
                             "AsyncLLM": SimpleNamespace(from_engine_args=factory)}
                initialize = load_definition(ROOT / "indextts/infer_vllm_v2.py", "QwenEmotion.__init__", namespace)
                instance = SimpleNamespace(_init_cache=Mock())
                initialize(instance, "emotion-model", engine_profile=profile.emotion)
                options = factory.call_args.args[0]
                self.assertEqual(options["gpu_memory_utilization"], profile.emotion.gpu_memory_utilization)
                self.assertEqual(options["max_model_len"], 2048)
                self.assertEqual(instance.max_model_len, 2048)
                self.assertTrue(options["enable_sleep_mode"])
                if size == 24:
                    self.assertEqual(options["max_num_seqs"], 1)
                    self.assertTrue(options["enforce_eager"])

    def test_gpt_factory_receives_resolved_options(self):
        for size in (24, 48, 96):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "config.json").write_text('{"n_positions": 4096}')
                profile = resolve_gpu_profile(gpu(size), {}, modal=True)
                factory = Mock(return_value="engine")
                namespace = {"self": SimpleNamespace(gpu_profile=profile), "vllm_dir": directory, "print": Mock(),
                             "_time": SimpleNamespace(time=lambda: 0), "_vllm_sleep_mode_enabled": lambda: True,
                             "AsyncEngineArgs": lambda **kwargs: kwargs,
                             "AsyncLLM": SimpleNamespace(from_engine_args=factory)}
                initialize = load_definition(ROOT / "indextts/infer_vllm_v2.py", "init_gpt_vllm", namespace)
                self.assertEqual(initialize(), "engine")
                options = factory.call_args.args[0]
                self.assertTrue(options["enable_sleep_mode"])
                self.assertEqual(options["gpu_memory_utilization"], profile.index.gpu_memory_utilization)
                if size == 24:
                    self.assertTrue(options["enforce_eager"])
                    self.assertEqual(options["max_num_batched_tokens"], 4096)
                if size == 96:
                    self.assertNotIn("max_num_seqs", options)


class SnapshotWarmupTests(unittest.IsolatedAsyncioTestCase):
    def test_restore_validates_gpu_before_wake_and_synthesis(self):
        events = []
        namespace = {"print": Mock(), "_wait_moss_ready": lambda proc: events.append("moss"),
                     "_call_local_json": lambda path, **kwargs: events.append(path),
                     "_wait_ready": lambda proc, **kwargs: events.append("ready"),
                     "SNAPSHOT_REQUEST_TIMEOUT": 900, "SNAPSHOT_STARTUP_TIMEOUT": 1800}
        restore = load_definition(ROOT / "deploy_vllm_indextts_v2.py", "wake_up", namespace)
        server = SimpleNamespace(moss_server_proc=Mock(), server_proc=Mock())
        for size in (24, 48, 96):
            events.clear()
            with self.subTest(size=size), \
                    patch("indextts_web.gpu_profiles.runtime_gpu_profile",
                          return_value=resolve_gpu_profile(gpu(size), {}, modal=True)), \
                    patch("indextts_web.infrastructure.gpu.probe_gpu", return_value=gpu(size)):
                restore(server)
                self.assertEqual(events, ["moss", "/internal/snapshot/wake", "ready", "/internal/snapshot/warmup"])
        events.clear()
        with patch("indextts_web.gpu_profiles.runtime_gpu_profile",
                   return_value=resolve_gpu_profile(gpu(96), {}, modal=True)), \
                patch("indextts_web.infrastructure.gpu.probe_gpu", return_value=gpu(24)):
            with self.assertRaisesRegex(RuntimeError, "Snapshot GPU does not match"):
                restore(server)
        self.assertEqual(events, [])

    async def test_missing_audio_must_fail_strict_snapshot_warmup(self):
        manager = SimpleNamespace(ensure_awake=AsyncMock(), get_tts=lambda: object())
        with tempfile.TemporaryDirectory() as directory:
            namespace = {"os": os, "current_dir": directory, "tts_manager": manager, "print": Mock()}
            warmup = load_definition(ROOT / "fastapi_webui_v2_impl.py", "warmup_model", namespace)
            with self.assertRaisesRegex(FileNotFoundError, "Warmup audio missing"):
                await warmup(strict=True)
