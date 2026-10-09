"""CPU regressions for the Colab notebook's dependency-check policy."""

import ast
import contextlib
import io
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
