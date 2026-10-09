"""CPU regressions for the Colab notebook's dependency-check policy."""

import ast
import contextlib
import io
import json
import os
import shlex
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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
