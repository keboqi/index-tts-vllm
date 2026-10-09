import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from indextts_web.infrastructure import optional_runtime as runtime


class OptionalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "app"
        self.workspace.mkdir()
        for name in ("requirements-clearvoice.txt", "requirements-colab-moss.txt", "requirements-colab-qwen-asr.txt"):
            (self.workspace / name).write_text("torch==2.8.0\n")
        (self.workspace / "constraints-main.txt").write_text("torch==2.8.0\ntransformers==4.57.3\nomnivad==0.2.13\n")
        self.calls = []

    def record(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if "venv" in command:
            python = Path(command[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()

    def prepare(self, kind="moss", **kwargs):
        with patch.object(runtime.shutil, "which", return_value="uv"), contextlib.redirect_stdout(io.StringIO()):
            return runtime.prepare(kind, self.workspace, self.root, self.root / "cache", run=self.record, **kwargs)

    def test_each_runtime_is_isolated_and_validated_without_loading_models(self):
        for kind in runtime.SMOKE:
            with self.subTest(kind=kind):
                self.calls.clear()
                python = self.prepare(kind)
                create = next(command for command, _ in self.calls if "venv" in command)
                self.assertNotIn("--system-site-packages", create)
                install = next(command for command, _ in self.calls if "-r" in command)
                self.assertEqual(install[install.index("--python") + 1], str(python))
                self.assertEqual(install[install.index("--torch-backend") + 1], "cu128")
                self.assertTrue(any("check" in command for command, _ in self.calls))
                smoke = self.calls[-1][0][-1]
                self.assertIn("torch.cuda.is_available()", smoke)
                self.assertNotIn("from_pretrained(", smoke)
                self.assertTrue((python.parent.parent / "index-tts-ready.txt").is_file())

    def test_qwen_replaces_only_transformers_constraint(self):
        self.prepare("qwen-asr")
        self.assertEqual((self.root / "cache/constraints-qwen-asr.txt").read_text(),
                         "torch==2.8.0\nomnivad==0.2.13\n")

    def test_successful_environment_is_reused_and_manifest_changes_trigger_revalidation(self):
        self.prepare()
        self.calls.clear()
        self.prepare()
        self.assertEqual(self.calls, [])
        (self.workspace / "requirements-colab-moss.txt").write_text("torch==2.8.0\ntransformers==5.6.0\n")
        self.prepare()
        self.assertTrue(self.calls)
        self.assertFalse(any("venv" in command for command, _ in self.calls))

    def test_failed_validation_clears_ready_marker_and_can_retry(self):
        python = self.prepare()
        original = self.record

        def fail(command, **kwargs):
            if "check" in command:
                raise subprocess.CalledProcessError(1, command)
            original(command, **kwargs)

        self.record = fail
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare(force=True)
        self.assertFalse((python.parent.parent / "index-tts-ready.txt").exists())
        self.record = original
        self.prepare()
        self.assertTrue((python.parent.parent / "index-tts-ready.txt").exists())

    def test_inherited_constraints_and_main_python_path_do_not_leak(self):
        with patch.dict(os.environ, {"PIP_CONSTRAINT": "main", "UV_CONSTRAINT": "main", "PYTHONPATH": "main"}):
            self.prepare()
        for _, kwargs in self.calls:
            self.assertNotIn("PIP_CONSTRAINT", kwargs["env"])
            self.assertNotIn("UV_CONSTRAINT", kwargs["env"])
            self.assertNotIn("PYTHONPATH", kwargs["env"])

    def test_wrapper_passes_worker_arguments_unchanged_to_real_interpreter(self):
        args = ["optional-runtime", "qwen-asr", "--workspace", str(self.workspace), "--root", str(self.root),
                "--cache", str(self.root / "cache"), "--exec-python", "--", "-u", "-m", "worker", "--request", "my job.json"]
        python = runtime.interpreter(self.root, "qwen-asr")
        with patch.dict(os.environ, {"VIRTUAL_ENV": "/main", "PATH": "/main/bin", "PYTHONPATH": str(self.workspace)}), \
                patch.object(sys, "argv", args), patch.object(runtime, "prepare", return_value=python), \
                patch.object(runtime.os, "execv") as execute:
            runtime.main()
            self.assertEqual(os.environ["VIRTUAL_ENV"], str(python.parent.parent))
            self.assertTrue(os.environ["PATH"].startswith(str(python.parent) + os.pathsep))
            self.assertEqual(os.environ["PYTHONPATH"], str(self.workspace))
        execute.assert_called_once_with(str(python), [str(python), "-u", "-m", "worker", "--request", "my job.json"])

    def test_local_moss_inherits_app_process_group_and_redirects_pipes(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(runtime, "moss_ready", side_effect=[False, True]), \
                patch.object(runtime.subprocess, "Popen", return_value=process) as popen, \
                contextlib.redirect_stdout(io.StringIO()):
            runtime.start_moss(runtime.interpreter(self.root, "moss"), self.workspace)
        self.assertNotIn("start_new_session", popen.call_args.kwargs)
        self.assertNotEqual(popen.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertIn("moss_transcribe_server:app", popen.call_args.args[0])

    def test_moss_startup_timeout_terminates_child(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(runtime, "moss_ready", return_value=False), \
                patch.object(runtime.time, "monotonic", side_effect=[0, 121]), \
                patch.object(runtime.subprocess, "Popen", return_value=process), \
                self.assertRaises(TimeoutError):
            runtime.start_moss(runtime.interpreter(self.root, "moss"), self.workspace)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
