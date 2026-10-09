"""CPU regressions for the Colab notebook's dependency-check policy."""

import ast
import contextlib
import io
import json
import os
import queue
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from indextts_web.config import load_settings
from indextts_web.gpu_profiles import GIB, GpuInfo, resolve_gpu_profile


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
            "run_logged": lambda command, **options: self.calls.append((command, options)),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), self.namespace)

    def setup_worker(self, *, enabled=True, existing=False):
        with patch.object(Path, "is_file", return_value=existing), contextlib.redirect_stdout(io.StringIO()):
            self.namespace["setup_clearvoice"](enabled)

    def test_installs_isolated_manifest_and_exports_worker_interpreter(self):
        self.setup_worker()
        create = next(command for command, _ in self.calls if "venv" in command)
        self.assertNotIn("--system-site-packages", create)
        install, options = next((command, options) for command, options in self.calls if "-r" in command)
        self.assertEqual(Path(install[install.index("-r") + 1]).name, "requirements-clearvoice.txt")
        self.assertNotIn("-c", install)
        self.assertEqual(install[install.index("--torch-backend") + 1], "cu128")
        worker_python = str(Path("/content/venv_index_tts_clearvoice/bin/python"))
        self.assertEqual(install[install.index("--python") + 1], worker_python)
        self.assertEqual(self.run_env["CLEARVOICE_PYTHON"], worker_python)
        self.assertEqual(options["env"]["VIRTUAL_ENV"], str(Path(worker_python).parent.parent))
        self.assertTrue(any("indextts_web.services.audio.clearvoice_worker" in command for command, _ in self.calls))

    def test_rerun_reuses_environment_and_revalidates_worker(self):
        self.setup_worker(existing=True)
        self.assertFalse(any("venv" in command for command, _ in self.calls))
        self.assertTrue(any("check" in command for command, _ in self.calls))
        self.assertIn("CLEARVOICE_PYTHON", self.run_env)

    def test_disabled_worker_clears_stale_configuration_without_installing(self):
        self.setup_worker(enabled=False)
        self.assertEqual(self.calls, [])
        self.assertNotIn("CLEARVOICE_PYTHON", self.run_env)

    def test_failed_install_does_not_publish_an_unavailable_worker(self):
        self.namespace["run_logged"] = Mock(side_effect=subprocess.CalledProcessError(1, ["uv"]))
        with self.assertRaises(subprocess.CalledProcessError):
            self.setup_worker(existing=True)
        self.assertNotIn("CLEARVOICE_PYTHON", self.run_env)

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
                    "MOSS_TRANSCRIBE_DEVICE": "cuda:0",
                    "MOSS_TRANSCRIBE_MODEL": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
                    "QWEN_OMNIVAD_PYTHON": "/content/qwen/bin/python",
                    "QWEN_OMNIVAD_MODEL_DIR": "/repo/checkpoints/qwen_omnivad",
                    "QWEN_OMNIVAD_CACHE_DIR": "/content/cache/qwen_omnivad",
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
            "run_logged": lambda command, **options: self.calls.append((command, options)),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), self.scope)

    def setup_asr(self, kind, enabled=True):
        with contextlib.redirect_stdout(io.StringIO()):
            self.scope["setup_asr"](kind, enabled)

    def test_both_workers_install_clean_environments_and_validate_before_exporting(self):
        for kind, key in (("moss", "COLAB_MOSS_PYTHON"), ("qwen-asr", "QWEN_OMNIVAD_PYTHON")):
            with self.subTest(kind=kind):
                self.calls.clear()
                self.setup_asr(kind)
                create = next(command for command, _ in self.calls if "venv" in command)
                self.assertNotIn("--system-site-packages", create)
                install, options = next((command, options) for command, options in self.calls if "-r" in command)
                self.assertEqual(Path(install[install.index("-r") + 1]).name, f"requirements-colab-{kind}.txt")
                self.assertEqual(install[install.index("--torch-backend") + 1], "cu128")
                self.assertEqual(install[install.index("--python") + 1], self.env[key])
                self.assertNotEqual(options["env"]["VIRTUAL_ENV"], "/main")
                self.assertTrue(any("check" in command for command, _ in self.calls))
                smoke = next(command[-1] for command, _ in self.calls if "-c" in command and "import torch" in command[-1])
                self.assertIn("torch.cuda.is_available()", smoke)
                self.assertNotIn("from_pretrained(", smoke)

    def test_qwen_uses_vetted_diarization_constraints_with_its_transformers_pin(self):
        self.setup_asr("qwen-asr")
        constraints = (self.scope["CACHE_DIR"] / "constraints-qwen-asr.txt").read_text()
        self.assertEqual(constraints, "torch==2.8.0\nwhisperx==3.3.1\n")
        self.assertTrue(any("indextts_web.services.translation.qwen_worker" in command for command, _ in self.calls))
        self.assertEqual(self.env["QWEN_OMNIVAD_MODEL_DIR"], str(self.scope["WORKSPACE_DIR"] / "checkpoints/qwen_omnivad"))

    def test_moss_uses_http_service_without_managed_docker_startup(self):
        self.setup_asr("moss")
        self.assertEqual(self.env["MOSS_TRANSCRIBE_BACKEND"], "http")
        self.assertEqual(self.env["MOSS_TRANSCRIBE_MANAGE_BACKEND"], "0")
        self.assertFalse(any("-c" in command for command, _ in self.calls if "-r" in command))

    def test_disabled_workers_clear_stale_interpreters_and_never_install(self):
        self.setup_asr("moss", enabled=False)
        self.setup_asr("qwen-asr", enabled=False)
        self.assertEqual(self.calls, [])
        self.assertNotIn("COLAB_MOSS_PYTHON", self.env)
        self.assertNotIn("QWEN_OMNIVAD_PYTHON", self.env)
        self.assertEqual(self.env["MOSS_TRANSCRIBE_MANAGE_BACKEND"], "0")

    def test_failed_validation_does_not_export_worker(self):
        for kind, key in (("moss", "COLAB_MOSS_PYTHON"), ("qwen-asr", "QWEN_OMNIVAD_PYTHON")):
            with self.subTest(kind=kind):
                self.scope["run_logged"] = Mock(side_effect=subprocess.CalledProcessError(1, ["uv"]))
                with self.assertRaises(subprocess.CalledProcessError):
                    self.setup_asr(kind)
                self.assertNotIn(key, self.env)


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
