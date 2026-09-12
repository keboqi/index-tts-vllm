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
                self.assertEqual(options["worker_cls"], "indextts_web.infrastructure.vllm_worker.RecoverableSleepWorker")
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
                self.assertEqual(options["worker_cls"], "indextts_web.infrastructure.vllm_worker.RecoverableSleepWorker")
                self.assertEqual(options["gpu_memory_utilization"], profile.index.gpu_memory_utilization)
                if size == 24:
                    self.assertTrue(options["enforce_eager"])
                    self.assertEqual(options["max_num_batched_tokens"], 4096)
                if size == 96:
                    self.assertNotIn("max_num_seqs", options)


class SnapshotWarmupTests(unittest.IsolatedAsyncioTestCase):
    def test_snapshot_creation_still_warms_up_before_sleep(self):
        events = []
        namespace = {"print": Mock(), "os": os,
                     "subprocess": SimpleNamespace(Popen=Mock()),
                     "_configure_persistent_runtime": lambda: ROOT,
                     "_commit_snapshot_volumes": lambda phase: events.append(phase),
                     "_start_moss_transcribe_server": Mock(),
                     "_wait_moss_ready": lambda proc: events.append("moss"),
                     "_build_webui_command": lambda path: ["python", "webui.py"],
                     "_wait_ready": lambda proc, **kwargs: events.append("ready"),
                     "_call_local_json": lambda path, **kwargs: events.append(path),
                     "SNAPSHOT_REQUEST_TIMEOUT": 900, "SNAPSHOT_STARTUP_TIMEOUT": 1800}
        start = load_definition(ROOT / "deploy_vllm_indextts_v2.py", "IndexTTSVllmServer.start", namespace)
        start(SimpleNamespace())
        self.assertEqual(events, ["before model startup", "moss", "ready", "/internal/snapshot/warmup",
                                  "/internal/snapshot/sleep?level=1", "before snapshot capture"])

    def test_restore_validates_gpu_and_readiness_without_repeating_warmup(self):
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
                self.assertEqual(events, ["moss", "/internal/snapshot/wake", "ready"])
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


class SnapshotVolumeTests(unittest.TestCase):
    def run_snapshot_start(self, fail_volume=None, fail_commit=None):
        # Model a fresh mount: local writes only become restorable after commit.
        pending = {"cache": set(), "app": set()}
        durable = {"cache": set(), "app": set()}
        commit_counts = {"cache": 0, "app": 0}
        volumes = {}

        def commit(name):
            commit_counts[name] += 1
            if name == fail_volume and commit_counts[name] == fail_commit:
                raise OSError("Volume storage unavailable")
            durable[name].update(pending[name])

        for name in pending:
            volumes[name] = Mock(commit=Mock(side_effect=lambda name=name: commit(name)))

        def configure():
            pending["cache"].add("gpu-profiles/new-profile")
            pending["app"].add("outputs")
            return ROOT

        def build_command(path):
            pending["cache"].add("omni-deploy-config.yaml")
            return ["python", "webui.py"]

        def start_worker(*args, **kwargs):
            self.assertEqual(durable["cache"], {"gpu-profiles/new-profile", "omni-deploy-config.yaml"})
            self.assertEqual(durable["app"], {"outputs"})
            return Mock()

        def snapshot_request(path, **kwargs):
            if path == "/internal/snapshot/warmup":
                pending["cache"].add("gpu-profiles/new-profile/compiled-kernel")
                pending["app"].add("emotion_cache/warmup.json")

        namespace = {"print": Mock(), "os": os,
                     "cache_storage": volumes["cache"], "app_storage": volumes["app"],
                     "_configure_persistent_runtime": configure,
                     "_build_webui_command": build_command,
                     "_start_moss_transcribe_server": Mock(side_effect=start_worker),
                     "_wait_moss_ready": Mock(),
                     "subprocess": SimpleNamespace(Popen=Mock(side_effect=start_worker)),
                     "_wait_ready": Mock(), "_call_local_json": Mock(side_effect=snapshot_request),
                     "SNAPSHOT_REQUEST_TIMEOUT": 900, "SNAPSHOT_STARTUP_TIMEOUT": 1800}
        source = ROOT / "deploy_vllm_indextts_v2.py"
        load_definition(source, "_commit_snapshot_volumes", namespace)
        start = load_definition(source, "IndexTTSVllmServer.start", namespace)
        if fail_volume is None:
            start(SimpleNamespace())
            self.assertEqual(durable, pending)
            self.assertIn("gpu-profiles/new-profile/compiled-kernel", durable["cache"])
            self.assertIn("emotion_cache/warmup.json", durable["app"])
            self.assertEqual(commit_counts, {"cache": 2, "app": 2})
        else:
            phase = "before model startup" if fail_commit == 1 else "before snapshot capture"
            with self.assertRaisesRegex(RuntimeError, f"audio-studio-{fail_volume}.*{phase}") as error:
                start(SimpleNamespace())
            self.assertIsInstance(error.exception.__cause__, OSError)
            if fail_commit == 1:
                namespace["_start_moss_transcribe_server"].assert_not_called()
                namespace["subprocess"].Popen.assert_not_called()
            else:
                self.assertEqual(namespace["_call_local_json"].call_args.args[0],
                                 "/internal/snapshot/sleep?level=1")
        for volume in volumes.values():
            volume.reload.assert_not_called()

    def test_first_snapshot_persists_new_paths_and_warmup_artifacts(self):
        self.run_snapshot_start()

    def test_failed_commit_aborts_startup_or_capture_for_either_volume(self):
        for name in ("cache", "app"):
            for commit_number in (1, 2):
                with self.subTest(volume=name, commit_number=commit_number):
                    self.run_snapshot_start(name, commit_number)
