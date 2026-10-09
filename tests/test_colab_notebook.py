"""CPU regressions for the Colab notebook's dependency-check policy."""

import ast
import contextlib
import hashlib
import io
import json
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from indextts_web.config import load_settings
from indextts_web.gpu_profiles import GIB, GpuInfo, resolve_gpu_profile
from tests.support import ROOT, load_definition


def notebook_cell(title):
    notebook = json.loads((ROOT / "index_tts_vllm_colab.ipynb").read_text(encoding="utf-8"))
    return next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and title in "".join(cell["source"]).splitlines()[0])


class ColabModelOptionsTests(unittest.TestCase):
    def options(self, **overrides):
        tree = ast.parse(notebook_cell("5. Download models"))
        flags = {"DOWNLOAD_HY_MT": True, "DOWNLOAD_VOICE_DESIGN": False, "DOWNLOAD_MOSS_MODEL": False,
                 "STABLE_AUDIO_DOWNLOAD": "off", "USE_HF_SECRET": False, **overrides}
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in flags for target in node.targets))]
        scope = {**flags, "INSTALL_OPTIONAL_FEATURES": True, "WORKSPACE_DIR": Path("/repo"),
                 "PYTHON": "/venv/bin/python", "RUN_ENV": {}, "run_logged": Mock()}
        exec(compile(tree, "<Colab model options>", "exec"), scope)
        return scope

    def test_default_models_and_voice_design_lazy_download(self):
        scope = self.options()
        self.assertEqual(scope["targets"], ["index", "hy-mt"])
        self.assertEqual(scope["RUN_ENV"]["QWEN3_VOICE_DESIGN_MODEL"], "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

    def test_hy_mt_download_off_enables_huggingface_fallback(self):
        scope = self.options(DOWNLOAD_HY_MT=False)
        self.assertEqual(scope["targets"], ["index"])
        self.assertEqual(scope["RUN_ENV"]["HY_MT_TRANSLATION_LOCAL_DIR"], "")

    def test_explicit_voice_design_and_gated_model_downloads(self):
        scope = self.options(DOWNLOAD_VOICE_DESIGN=True, STABLE_AUDIO_DOWNLOAD="all")
        self.assertEqual(scope["targets"], ["index", "hy-mt", "voice-design", "stable-audio-medium",
                                           "stable-audio-small-music", "stable-audio-small-sfx"])
        self.assertEqual(scope["RUN_ENV"]["QWEN3_VOICE_DESIGN_MODEL"],
                         str(Path("/repo/checkpoints/Qwen3-TTS-12Hz-1.7B-VoiceDesign")))

    def test_downloads_use_modal_catalog_and_reject_incomplete_bundles(self):
        from indextts_web.infrastructure import model_setup
        scope = self.options(STABLE_AUDIO_DOWNLOAD="small-sfx")
        hub = ModuleType("huggingface_hub")
        hub.snapshot_download = Mock()
        argv = ["-c", "/repo", json.dumps(scope["targets"])]
        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.object(sys, "argv", argv), \
                patch.object(model_setup, "missing_model_files", return_value=[]) as validate, \
                contextlib.redirect_stdout(io.StringIO()):
            exec(scope["MODEL_DOWNLOAD_SOURCE"], {})
        self.assertEqual([call.kwargs["repo_id"] for call in hub.snapshot_download.call_args_list],
                         ["garyswansrs/index_tts_2_vllm", "tencent/Hy-MT2-1.8B", "stabilityai/stable-audio-3-small-sfx"])
        self.assertEqual(validate.call_count, 3)
        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.object(sys, "argv", argv), \
                patch.object(model_setup, "missing_model_files", return_value=["missing-shard.safetensors"]), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "missing-shard"):
            exec(scope["MODEL_DOWNLOAD_SOURCE"], {})

    def test_additional_model_downloads_and_service_preparation_default_off(self):
        for title in ("3. Install Python", "5. Download models", "6. Launch"):
            tree = ast.parse(notebook_cell(title))
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and (target.id.startswith("PREPARE_") or target.id in {
                            "DOWNLOAD_VOICE_DESIGN", "DOWNLOAD_MOSS_MODEL", "DOWNLOAD_QWEN_MODELS",
                            "DOWNLOAD_CLEARVOICE_MODELS", "ENABLE_WARMUP",
                        }):
                            self.assertIs(node.value.value, False, target.id)


class ColabPreparationTests(unittest.TestCase):
    def test_checked_tts_services_use_shared_wake_and_sleep_routes_without_synthesis(self):
        tree = ast.parse(notebook_cell("6. Launch"))
        launcher = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "LAUNCHER_SOURCE" for t in node.targets))
        function = next(node for node in ast.parse(launcher).body
                        if isinstance(node, ast.FunctionDef) and node.name == "prepare_tts_services")
        import urllib.request
        for keys in ([], ["confucius_vllm", "indextts25_omni"]):
            with self.subTest(keys=keys), patch.dict(os.environ, {"COLAB_PREPARE_TTS": json.dumps(keys)}), \
                    patch.object(urllib.request, "urlopen", side_effect=lambda *_a, **_kw: io.BytesIO(b'{"status":"success"}')) as http:
                scope = {"json": json, "os": os, "events": queue.Queue(), "port": 8000,
                         "startup_timeout": 1800, "urllib": SimpleNamespace(request=urllib.request),
                         "preparation_errors": [], "preparation_done": Mock()}
                exec(compile(ast.Module(body=[function], type_ignores=[]), "<TTS preparation>", "exec"), scope)
                scope["prepare_tts_services"]()
                self.assertEqual(scope["preparation_errors"], [])
                scope["preparation_done"].set.assert_called_once()
                self.assertEqual([call.args[0].full_url.rsplit("/", 1)[-1] for call in http.call_args_list],
                                 [action for _ in keys for action in ("wake", "unload")])

    def test_failed_service_preparation_is_reported_and_signals_completion(self):
        tree = ast.parse(notebook_cell("6. Launch"))
        launcher = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "LAUNCHER_SOURCE" for t in node.targets))
        function = next(node for node in ast.parse(launcher).body
                        if isinstance(node, ast.FunctionDef) and node.name == "prepare_tts_services")
        scope = {"json": json, "os": os, "events": queue.Queue(), "port": 8000, "startup_timeout": 1800,
                 "urllib": SimpleNamespace(request=SimpleNamespace(Request=Mock(), urlopen=Mock(side_effect=OSError("setup failed")))),
                 "preparation_errors": [], "preparation_done": Mock()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<TTS preparation>", "exec"), scope)
        with patch.dict(os.environ, {"COLAB_PREPARE_TTS": '["confucius_vllm"]'}):
            scope["prepare_tts_services"]()
        self.assertEqual(str(scope["preparation_errors"][0]), "setup failed")
        scope["preparation_done"].set.assert_called_once()

    def test_lazy_wrappers_preserve_worker_arguments_and_use_one_installer(self):
        tree = ast.parse(notebook_cell("3. Install Python"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"optional_runtime_command", "write_optional_wrappers"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope = {"WORKSPACE_DIR": root, "NOTEBOOK_DIR": root, "CACHE_DIR": root / "cache", "PYTHON": root / "bin/python"}
            exec(compile(ast.Module(body=functions, type_ignores=[]), "<optional wrappers>", "exec"), scope)
            scope["write_optional_wrappers"]()
            for kind in ("clearvoice", "qwen-asr"):
                wrapper = (root / ".colab" / (kind + "-python")).read_text()
                self.assertIn("indextts_web.infrastructure.optional_runtime", wrapper)
                self.assertTrue(wrapper.endswith('"$@"\n'))
                self.assertIn("--exec-python --", wrapper)
            manager = (root / ".colab/moss-start.sh").read_text()
            self.assertIn("--start-moss", manager)
            self.assertIn("| tee", manager)
            self.assertNotIn("docker", manager)


class ColabNodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        self.node = self.cache / "old-node"
        self.node.write_text("stub")
        tree = ast.parse(notebook_cell("3. Install Python"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setup_node")
        self.calls = []
        self.scope = {"CACHE_DIR": self.cache, "Path": Path, "RUN_ENV": {"PATH": "/usr/bin"},
                      "shutil": SimpleNamespace(which=Mock(return_value=str(self.node))),
                      "subprocess": SimpleNamespace(check_output=Mock(return_value="v18.0.0\n"), CalledProcessError=subprocess.CalledProcessError),
                      "run_logged": self.run_command}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<Colab Node setup>", "exec"), self.scope)

    def run_command(self, command, **kwargs):
        self.calls.append(command)
        if command[0] == "curl":
            Path(command[-1]).write_bytes(b"test node archive")

    def test_uses_existing_supported_node_without_downloading(self):
        self.scope["subprocess"].check_output.return_value = "v22.0.0\n"
        self.scope["setup_node"]()
        self.assertEqual(self.scope["RUN_ENV"]["YTDLP_NODE_PATH"], str(self.node))
        self.assertEqual(self.calls, [])

    def test_downloads_and_verifies_archive_before_installing(self):
        import urllib.request
        checksum = hashlib.sha256(b"test node archive").hexdigest()
        manifest = f"{checksum}  node-v22.1.0-linux-x64.tar.xz\n".encode()
        with patch.object(urllib.request, "urlopen", return_value=io.BytesIO(manifest)):
            self.scope["setup_node"]()
        self.assertEqual([command[0] for command in self.calls[:2]], ["curl", "tar"])
        self.assertEqual(self.scope["RUN_ENV"]["YTDLP_NODE_PATH"], str(self.cache / "node/bin/node"))

    def test_checksum_failure_never_extracts_or_exports_node(self):
        import urllib.request
        manifest = f"{'0' * 64}  node-v22.1.0-linux-x64.tar.xz\n".encode()
        with patch.object(urllib.request, "urlopen", return_value=io.BytesIO(manifest)), \
                self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self.scope["setup_node"]()
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("YTDLP_NODE_PATH", self.scope["RUN_ENV"])


class ColabOptionalRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tree = ast.parse(notebook_cell("2. Clone"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "provision_optional_repository")
        self.commands = []
        self.entrypoint = "scripts/serve.sh"
        self.scope = {"WORKSPACE_DIR": self.root / "app", "run_logged": self.clone}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<optional checkout>", "exec"), self.scope)

    def clone(self, command):
        self.commands.append(command)
        destination = Path(command[-1])
        (destination / ".git").mkdir(parents=True)
        entrypoint = destination / self.entrypoint
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("stub")

    def provision(self, enabled=True):
        with contextlib.redirect_stdout(io.StringIO()):
            self.scope["provision_optional_repository"]("Backend", "https://example.com/backend.git",
                                                       "backend", self.entrypoint, enabled)

    def test_clone_provisions_source_without_building_environment_or_downloading_models(self):
        self.provision()
        self.assertEqual(self.commands, [["git", "clone", "--depth", "1", "https://example.com/backend.git",
                                          str(self.root / "backend")]])

    def test_rerun_reuses_source_without_pulling_or_resetting(self):
        self.provision()
        self.commands.clear()
        self.provision()
        self.assertEqual(self.commands, [])

    def test_disabled_repository_does_nothing(self):
        self.provision(enabled=False)
        self.assertEqual(self.commands, [])
        self.assertFalse((self.root / "backend").exists())

    def test_incomplete_checkout_is_preserved_and_reported(self):
        (self.root / "backend").mkdir()
        preserved = self.root / "backend/user-data.txt"
        preserved.write_text("keep")
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            self.provision()
        self.assertEqual(preserved.read_text(), "keep")
        self.assertEqual(self.commands, [])


class VoiceDesignConfigCompatibilityTests(unittest.TestCase):
    def test_standalone_and_modal_defaults_preserved_with_optional_flash_override(self):
        for override in (None, "0", "1"):
            with self.subTest(override=override), patch.dict(os.environ, {}, clear=True):
                if override is not None:
                    os.environ["QWEN3_TTS_USE_FLASH_ATTENTION"] = override
                    os.environ["QWEN3_VOICE_DESIGN_MODEL"] = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                manager = SimpleNamespace(preset_manager=object())
                scope = {"os": os, "_voice_design_manager": None, "Qwen3TTSConfig": Mock(),
                         "Qwen3VoiceDesignManager": Mock(return_value=manager), "APP_DIR": ROOT,
                         "SPEAKER_REFERENCE_DIR": "speaker_presets/references",
                         "TTSManager": SimpleNamespace(get_instance=lambda: None)}
                scope["_env_flag"] = load_definition(ROOT / "fastapi_webui_v2_impl.py", "_env_flag", {"os": os})
                load_definition(ROOT / "fastapi_webui_v2_impl.py", "get_voice_design_manager", scope)()
                kwargs = scope["Qwen3TTSConfig"].call_args.kwargs
                self.assertEqual(kwargs["use_flash_attention"], override != "0")
                expected = "./checkpoints/Qwen3-TTS-12Hz-1.7B-VoiceDesign" if override is None else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                self.assertEqual(kwargs["voice_design_model_path"], expected)


class ColabDependencyCheckTests(unittest.TestCase):
    # Actual pip-check diagnostics from the Colab installation report.
    KNOWN_CONFLICTS = "\n".join((
        "stable-audio-tools 0.0.20 has requirement importlib-resources==5.12.0, but you have importlib-resources 7.1.0.",
        "stable-audio-tools 0.0.20 has requirement PyWavelets==1.4.1, but you have pywavelets 1.10.0.",
        "stable-audio-tools 0.0.20 has requirement sentencepiece==0.1.99, but you have sentencepiece 0.2.2.",
        "stable-audio-tools 0.0.20 has requirement torch==2.7.1, but you have torch 2.8.0+cu128.",
        "stable-audio-tools 0.0.20 has requirement torchaudio==2.7.1, but you have torchaudio 2.8.0+cu128.",
        "stable-audio-tools 0.0.20 has requirement vector-quantize-pytorch==1.14.41, but you have vector-quantize-pytorch 1.31.6.",
    ))

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parent.parent / "index_tts_vllm_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        trees = [ast.parse("".join(cell["source"])) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        function = next(node for tree in trees for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "check_dependencies")
        namespace = {"subprocess": subprocess}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        cls.check_dependencies = staticmethod(namespace["check_dependencies"])

    def run_check(self, stdout, *, returncode=1, stderr=""):
        command = ["/venv/bin/python", "-m", "pip", "check"]
        result = subprocess.CompletedProcess(command, returncode, stdout, stderr)
        with patch.object(subprocess, "run", return_value=result) as run, contextlib.redirect_stdout(io.StringIO()) as logs:
            self.check_dependencies(command[0], {"PYTHONNOUSERSITE": "1"})
        self.assertEqual(run.call_args.args[0], command)
        return logs.getvalue()

    def test_clean_environment_passes(self):
        self.assertIn("No broken requirements", self.run_check("No broken requirements found.\n", returncode=0))

    def test_reported_stable_audio_overrides_pass_and_remain_visible(self):
        logs = self.run_check(self.KNOWN_CONFLICTS)
        self.assertEqual(logs.count("[Known Stable Audio metadata override]"), 6)
        self.assertIn("Dependency check passed", logs)

    def test_unrelated_conflict_still_fails(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_check(self.KNOWN_CONFLICTS + "\nvllm 0.10.2 has requirement torch==2.8.0, but you have torch 2.7.1.")

    def test_missing_stable_audio_dependency_still_fails(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_check(self.KNOWN_CONFLICTS + "\nstable-audio-tools 0.0.20 requires torchsde, which is not installed.")

    def test_unexpected_installed_version_still_fails(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_check(self.KNOWN_CONFLICTS.replace("torch 2.8.0+cu128", "torch 2.9.0+cu128"))

    def test_unreviewed_stable_audio_version_still_fails(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_check(self.KNOWN_CONFLICTS.replace("stable-audio-tools 0.0.20", "stable-audio-tools 0.0.21"))

    def test_checker_execution_failures_are_not_ignored(self):
        for stdout, returncode, stderr in (("", 1, ""), (self.KNOWN_CONFLICTS, 2, ""),
                                           (self.KNOWN_CONFLICTS, 1, "pip failed")):
            with self.subTest(returncode=returncode, stderr=stderr), self.assertRaises(subprocess.CalledProcessError):
                self.run_check(stdout, returncode=returncode, stderr=stderr)


class ColabClearVoiceSetupTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parent.parent / "index_tts_vllm_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.trees = [ast.parse("".join(cell["source"])) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        function = next(node for tree in self.trees for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "setup_clearvoice")
        self.calls = []
        self.run_env = {"PATH": "/main/bin", "VIRTUAL_ENV": "/main", "CLEARVOICE_PYTHON": "/stale/python"}
        self.namespace = {
            "RUN_ENV": self.run_env, "NOTEBOOK_DIR": Path("/content"),
            "WORKSPACE_DIR": Path("/content/index-tts-vllm"), "UV": ["python", "-m", "uv"], "os": os,
            "subprocess": SimpleNamespace(check_output=Mock(return_value="3.12\n")),
            "optional_runtime_command": lambda kind: ["python", "-m", "optional_runtime", kind],
            "run_logged": lambda command, **options: self.calls.append((command, options)),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), self.namespace)

    def setup_worker(self, *, enabled=True, existing=False):
        with patch.object(Path, "is_file", return_value=existing), contextlib.redirect_stdout(io.StringIO()):
            self.namespace["setup_clearvoice"](enabled)

    def test_checked_option_delegates_to_same_installer_as_first_use(self):
        self.setup_worker()
        self.assertEqual(self.calls[0][0], ["python", "-m", "optional_runtime", "clearvoice", "--force"])
        self.assertEqual(self.run_env["CLEARVOICE_PYTHON"], str(Path("/content/index-tts-vllm/.colab/clearvoice-python")))

    def test_default_deferred_worker_remains_available_without_installing(self):
        self.setup_worker(enabled=False)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.run_env["CLEARVOICE_PYTHON"], str(Path("/content/index-tts-vllm/.colab/clearvoice-python")))

    def test_failed_preparation_remains_on_demand_and_reports_failure(self):
        self.namespace["run_logged"] = Mock(side_effect=subprocess.CalledProcessError(1, ["uv"]))
        with self.assertRaises(subprocess.CalledProcessError):
            self.setup_worker()
        self.assertTrue(self.run_env["CLEARVOICE_PYTHON"].endswith("clearvoice-python"))

    def test_terminal_launcher_preserves_optional_worker_path(self):
        tree = next(tree for tree in self.trees if any(
            isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "persistent_env"
                                                for target in node.targets) for node in tree.body))
        start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "persistent_env" for target in node.targets))
        end = next(i for i, node in enumerate(tree.body[start:], start) if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                   and node.value.func.attr == "write_text")
        script = compile(ast.Module(body=tree.body[start:end], type_ignores=[]), "<terminal launcher>", "exec")
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                env = {"CLEARVOICE_PYTHON": "/content/clearvoice worker/bin/python"} if enabled else {}
                asr_env = {
                    "COLAB_MOSS_PYTHON": "/content/moss/bin/python",
                    "MOSS_TRANSCRIBE_BACKEND": "http", "MOSS_TRANSCRIBE_MANAGE_BACKEND": "0",
                    "MOSS_TRANSCRIBE_SGLANG_URL": "http://127.0.0.1:8003",
                    "MOSS_TRANSCRIBE_MANAGER_SCRIPT": "/repo/.colab/moss-start.sh",
                    "MOSS_TRANSCRIBE_DEVICE": "cuda:0",
                    "MOSS_TRANSCRIBE_MODEL": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
                    "QWEN_OMNIVAD_PYTHON": "/content/qwen/bin/python",
                    "QWEN_OMNIVAD_MODEL_DIR": "/repo/checkpoints/qwen_omnivad",
                    "QWEN_OMNIVAD_CACHE_DIR": "/content/cache/qwen_omnivad",
                    "YTDLP_NODE_PATH": "/content/node/bin/node",
                    "QWEN3_VOICE_DESIGN_MODEL": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "QWEN3_TTS_USE_FLASH_ATTENTION": "0",
                    "CUDA_CACHE_PATH": "/content/cache/cuda",
                    "TORCHINDUCTOR_CACHE_DIR": "/content/cache/torchinductor",
                    "TORCHINDUCTOR_COMPILE_THREADS": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                    "ORT_DISABLE_TELEMETRY": "1",
                    "CONFUCIUS_USE_TORCH_COMPILE": "0", "WARMUP": "0",
                    "COLAB_PREPARE_TTS": '["confucius_vllm"]',
                }
                if enabled:
                    env.update(asr_env)
                env["INDEXTTS_USE_TORCH_COMPILE"] = str(int(enabled))
                scope = {"RUN_ENV": env, "VENV_DIR": Path("/main"), "PYTHON": "/main/bin/python",
                         "COLAB_DIR": Path("/repo/.colab"), "SERVER_PORT": 8000,
                         "ENABLE_CLOUDFLARE_TUNNEL": True, "ENABLE_WARMUP": enabled,
                         "STARTUP_TIMEOUT_SECONDS": 1800, "shlex": shlex}
                exec(script, scope)
                exports = [shlex.split(line) for line in scope["launcher_text"].splitlines()
                           if line.startswith("export CLEARVOICE_PYTHON=")]
                expected = [["export", "CLEARVOICE_PYTHON=/content/clearvoice worker/bin/python"]] if enabled else []
                self.assertEqual(exports, expected)
                self.assertIn(f"export INDEXTTS_USE_TORCH_COMPILE={int(enabled)}", scope["launcher_text"])
                self.assertIn("unset PYTHONPATH PYTHONHOME MPLBACKEND CLEARVOICE_PYTHON", scope["launcher_text"])
                for key, value in asr_env.items():
                    expected = f"export {key}={shlex.quote(value)}"
                    self.assertEqual(expected in scope["launcher_text"], enabled)
                self.assertIn("COLAB_MOSS_PYTHON QWEN_OMNIVAD_PYTHON\n", scope["launcher_text"])


class ColabASRSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "constraints-main.txt").write_text("torch==2.8.0\ntransformers==4.57.3\nwhisperx==3.3.1\n")
        path = Path(__file__).resolve().parent.parent / "index_tts_vllm_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        trees = [ast.parse("".join(cell["source"])) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        function = next(node for tree in trees for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "setup_asr")
        self.calls = []
        self.env = {"PATH": "/main/bin", "VIRTUAL_ENV": "/main",
                    "COLAB_MOSS_PYTHON": "stale", "QWEN_OMNIVAD_PYTHON": "stale"}
        self.scope = {
            "RUN_ENV": self.env, "NOTEBOOK_DIR": root, "WORKSPACE_DIR": root,
            "CACHE_DIR": root / "cache", "UV": ["python", "-m", "uv"], "os": os,
            "subprocess": SimpleNamespace(check_output=Mock(return_value="3.12\n")),
            "optional_runtime_command": lambda kind: ["python", "-m", "optional_runtime", kind],
            "run_logged": lambda command, **options: self.calls.append((command, options)),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), self.scope)

    def setup_asr(self, kind, enabled=True):
        with contextlib.redirect_stdout(io.StringIO()):
            self.scope["setup_asr"](kind, enabled)

    def test_checked_workers_delegate_to_common_optional_installer(self):
        for kind in ("moss", "qwen-asr"):
            with self.subTest(kind=kind):
                self.calls.clear()
                self.setup_asr(kind)
                self.assertEqual(self.calls[0][0], ["python", "-m", "optional_runtime", kind, "--force"])

    def test_moss_preparation_exports_interpreter_for_early_service_start(self):
        self.setup_asr("moss")
        self.assertEqual(self.env["MOSS_TRANSCRIBE_BACKEND"], "http")
        self.assertEqual(self.env["MOSS_TRANSCRIBE_MANAGE_BACKEND"], "0")
        self.assertEqual(self.env["COLAB_MOSS_PYTHON"], str(self.scope["NOTEBOOK_DIR"] / "venv_index_tts_moss/bin/python"))

    def test_unchecked_moss_uses_lazy_local_manager_without_docker(self):
        self.setup_asr("moss", enabled=False)
        self.assertEqual(self.calls, [])
        self.assertNotIn("COLAB_MOSS_PYTHON", self.env)
        self.assertEqual(self.env["MOSS_TRANSCRIBE_MANAGE_BACKEND"], "1")
        self.assertTrue(self.env["MOSS_TRANSCRIBE_MANAGER_SCRIPT"].endswith("moss-start.sh"))

    def test_unchecked_qwen_exports_lazy_interpreter_without_installing(self):
        self.setup_asr("qwen-asr", enabled=False)
        self.assertEqual(self.calls, [])
        self.assertTrue(self.env["QWEN_OMNIVAD_PYTHON"].endswith("qwen-asr-python"))
        self.assertEqual(self.env["QWEN_OMNIVAD_MODEL_DIR"], str(self.scope["WORKSPACE_DIR"] / "checkpoints/qwen_omnivad"))

    def test_failed_moss_preparation_never_publishes_direct_interpreter(self):
        self.scope["run_logged"] = Mock(side_effect=subprocess.CalledProcessError(1, ["uv"]))
        with self.assertRaises(subprocess.CalledProcessError):
            self.setup_asr("moss")
        self.assertNotIn("COLAB_MOSS_PYTHON", self.env)


class ColabMossServiceTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent.parent
        notebook = json.loads((root / "index_tts_vllm_colab.ipynb").read_text(encoding="utf-8"))
        code = next("".join(cell["source"]) for cell in notebook["cells"] if "LAUNCHER_SOURCE =" in "".join(cell["source"]))
        launcher = next(ast.literal_eval(node.value) for node in ast.parse(code).body
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LAUNCHER_SOURCE" for t in node.targets))
        tree = ast.parse(launcher)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"start", "start_moss_service"}]
        self.process = Mock(returncode=None)
        self.process.poll.return_value = None
        self.env = {"COLAB_MOSS_PYTHON": "/content/moss/bin/python"}
        self.scope = {
            "os": SimpleNamespace(environ=self.env), "workspace": root,
            "port": 8000, "startup_timeout": 1800, "processes": [], "relay": Mock(),
            "socket": SimpleNamespace(socket=MagicMock()), "threading": SimpleNamespace(Thread=Mock()),
            "subprocess": SimpleNamespace(Popen=Mock(return_value=self.process), PIPE=-1, STDOUT=-2),
            "queue": queue, "events": Mock(), "json": json,
            "time": SimpleNamespace(monotonic=Mock(side_effect=[0, 0])),
            "urllib": SimpleNamespace(request=SimpleNamespace(urlopen=Mock()), error=SimpleNamespace(HTTPError=OSError)),
        }
        self.scope["events"].get.side_effect = queue.Empty
        response = io.BytesIO(b'{"service":"moss-transcribe","state":"unloaded"}')
        self.scope["urllib"].request.urlopen.return_value = response
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<Colab MOSS launcher>", "exec"), self.scope)
        main = next(node for node in tree.body if isinstance(node, ast.Try))
        calls = [node.value for node in main.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)]
        self.assertLess(next(i for i, c in enumerate(calls) if isinstance(c.func, ast.Name) and c.func.id == "start_moss_service"),
                        next(i for i, c in enumerate(calls) if isinstance(c.func, ast.Name) and c.func.id == "server_command"))

    def start_moss(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.scope["start_moss_service"]()

    def test_starts_local_server_and_checks_identity_without_loading_weights(self):
        self.assertIs(self.start_moss(), self.process)
        call = self.scope["subprocess"].Popen.call_args
        self.assertEqual(call.args[0][:4], ["/content/moss/bin/python", "-m", "uvicorn", "moss_transcribe_server:app"])
        self.assertTrue(call.kwargs["start_new_session"])
        self.assertEqual(self.scope["processes"], [self.process])
        self.assertEqual(self.scope["urllib"].request.urlopen.call_args.args[0], "http://127.0.0.1:8003/model/status")

    def test_disabled_service_does_not_start(self):
        self.env.clear()
        self.assertIsNone(self.start_moss())
        self.scope["subprocess"].Popen.assert_not_called()

    def test_port_collision_fails_before_start(self):
        self.scope["port"] = 8003
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.start_moss()
        self.scope["subprocess"].Popen.assert_not_called()

    def test_failed_service_remains_owned_for_launcher_cleanup(self):
        self.process.poll.return_value = 1
        self.process.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "exited with code 1"):
            self.start_moss()
        self.assertEqual(self.scope["processes"], [self.process])

    def test_timeout_keeps_process_owned_for_cleanup(self):
        self.scope["time"].monotonic.side_effect = [0, 121]
        with self.assertRaisesRegex(TimeoutError, "startup timed out"):
            self.start_moss()
        self.assertEqual(self.scope["processes"], [self.process])


class ColabWarmupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent.parent
        notebook = json.loads((root / "index_tts_vllm_colab.ipynb").read_text(encoding="utf-8"))
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        launch_tree = ast.parse(code[-1])
        cls.default = next(node.value.value for node in launch_tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "ENABLE_WARMUP" for target in node.targets))
        source = next(node.value.value for node in launch_tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "LAUNCHER_SOURCE" for target in node.targets))
        command = next(node for node in ast.parse(source).body
                       if isinstance(node, ast.FunctionDef) and node.name == "server_command")
        cls.command_code = compile(ast.Module(body=[command], type_ignores=[]), "<server command>", "exec")
        production = ast.parse((root / "fastapi_webui_v2_impl.py").read_text(encoding="utf-8"))
        cls.model_call = next(node for node in ast.walk(production) if isinstance(node, ast.Assign)
                              and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                              and node.value.func.id == "IndexTTS2")
        lifespan = next(node for node in production.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan")
        warmup = next(node for node in lifespan.body if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Attribute) and node.test.attr == "use_torch_compile")
        runner = ast.parse("async def run_warmup():\n    pass\n").body[0]
        runner.body = [warmup]
        cls.warmup_code = compile(ast.fix_missing_locations(ast.Module(body=[runner], type_ignores=[])), "<warmup>", "exec")
        engine = ast.parse((root / "indextts/infer_vllm_v2.py").read_text(encoding="utf-8"))
        constructor = next(node for node in engine.body if isinstance(node, ast.ClassDef) and node.name == "IndexTTS2")
        constructor = next(node for node in constructor.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        compile_setting = next(node for node in constructor.body if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "use_torch_compile" for target in node.targets))
        compile_branch = next(node for node in constructor.body if isinstance(node, ast.If)
                              and isinstance(node.test, ast.Attribute) and node.test.attr == "use_torch_compile")
        cls.compile_code = compile(ast.Module(body=[compile_setting, compile_branch], type_ignores=[]), "<s2mel compile>", "exec")

    def test_default_disables_both_warmup_and_compilation(self):
        self.assertIs(self.default, False)

    def test_one_switch_controls_environment_s2mel_and_startup_warmup(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), contextlib.redirect_stdout(io.StringIO()):
                env = {"INDEXTTS_USE_TORCH_COMPILE": str(int(not enabled))}
                scope = {"os": SimpleNamespace(environ=env), "warmup_enabled": enabled,
                         "sys": SimpleNamespace(executable="/venv/bin/python"), "workspace": Path("/repo"), "port": 8000}
                exec(self.command_code, scope)
                command = scope["server_command"]()
                self.assertEqual(env["INDEXTTS_USE_TORCH_COMPILE"], str(int(enabled)))
                settings = load_settings(command[3:], environ=env)
                # G4's automatic compilation default must respect the single switch.
                gpu = GpuInfo("RTX PRO 6000", 96 * GIB, 96 * GIB, "12.0")
                profile = resolve_gpu_profile(gpu, {}).with_settings(settings)
                settings = profile.apply_settings(settings)
                factory = Mock()
                exec(compile(ast.Module(body=[self.model_call], type_ignores=[]), "<model routing>", "exec"),
                     {"IndexTTS2": factory, "self": SimpleNamespace(), "SETTINGS": settings, "GPU_PROFILE": profile})
                compiler = Mock()
                engine = SimpleNamespace(gpu_profile=profile, s2mel=SimpleNamespace(enable_torch_compile=compiler))
                exec(self.compile_code, {"self": engine, "use_torch_compile": factory.call_args.kwargs["use_torch_compile"]})
                warmup = AsyncMock()
                scope = {"SETTINGS": settings, "warmup_model": warmup}
                exec(self.warmup_code, scope)
                # AsyncMock completes immediately; no Windows event-loop sockets are needed.
                with self.assertRaises(StopIteration):
                    scope["run_warmup"]().send(None)
                self.assertIs(settings.use_torch_compile, enabled)
                self.assertEqual(compiler.call_count, int(enabled))
                self.assertEqual(warmup.await_count, int(enabled))


if __name__ == "__main__":
    unittest.main()
