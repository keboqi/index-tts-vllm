import ast
import importlib.util
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from indextts_web.infrastructure.modal_dependencies import MAIN_DEPENDENCY_FILES, install_main_dependencies
from tests.support import ROOT


class FakeImage:
    """Keep image calls immutable, and snapshot the shipped file contents."""

    def __init__(self, calls=(), files=None, events=None):
        self.calls = calls
        self.files = {} if files is None else files
        self.events = [] if events is None else events

    def add_local_file(self, local_path, remote_path, *, copy=False):
        call = ("copy", str(local_path), remote_path, copy)
        self.events.append(call)
        files = {**self.files, remote_path: Path(local_path).read_text(encoding="utf-8")}
        return FakeImage((*self.calls, call), files, self.events)

    def run_commands(self, command):
        call = ("install", command)
        self.events.append(call)
        return FakeImage((*self.calls, call), self.files, self.events)


class ModalDependencyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name)
        self.contents = {
            "requirements.txt": "-r requirements-core.txt\n",
            "requirements-core.txt": "vllm==0.10.2\n",
            "requirements-modal.txt": "-c constraints-main.txt\n-r requirements-core.txt\n",
            "constraints-main.txt": "transformers==4.57.3\n",
        }
        for name, content in self.contents.items():
            (self.source / name).write_text(content, encoding="utf-8")

    def test_installs_current_local_files_before_running_pip(self):
        original = FakeImage()
        image = install_main_dependencies(original, self.source)
        self.assertEqual(original.calls, ())
        self.assertEqual(len(image.calls), len(MAIN_DEPENDENCY_FILES) + 1)
        for call, filename in zip(image.calls, MAIN_DEPENDENCY_FILES, strict=False):
            self.assertEqual(
                call, ("copy", str(self.source.resolve() / filename), f"/app/index-tts-vllm/{filename}", True)
            )
            self.assertEqual(image.files[f"/app/index-tts-vllm/{filename}"], self.contents[filename])
        self.assertEqual(
            image.calls[-1], ("install", "python -m pip install -r /app/index-tts-vllm/requirements-modal.txt")
        )

    def test_updated_manifest_contents_are_shipped_on_the_next_build(self):
        before = install_main_dependencies(FakeImage(), self.source)
        (self.source / "constraints-main.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
        after = install_main_dependencies(FakeImage(), self.source)
        path = "/app/index-tts-vllm/constraints-main.txt"
        self.assertNotEqual(before.files[path], after.files[path])
        self.assertEqual(after.files[path], "numpy==1.26.4\n")

    def test_local_only_requirements_do_not_change_the_modal_dependency_layer(self):
        before = install_main_dependencies(FakeImage(), self.source)
        (self.source / "requirements.txt").write_text("unrelated-local-package\n", encoding="utf-8")
        after = install_main_dependencies(FakeImage(), self.source)
        self.assertEqual(before.calls, after.calls)
        self.assertEqual(before.files, after.files)
        (self.source / "requirements.txt").unlink()
        self.assertEqual(install_main_dependencies(FakeImage(), self.source).files, before.files)

    def test_missing_manifests_fail_before_any_image_changes(self):
        for name in MAIN_DEPENDENCY_FILES:
            with self.subTest(manifest=name):
                target = self.source / name
                target.unlink()
                original = FakeImage()
                try:
                    with self.assertRaisesRegex(FileNotFoundError, name):
                        install_main_dependencies(original, self.source)
                    self.assertEqual(original.events, [])
                finally:
                    target.write_text(self.contents[name], encoding="utf-8")

    def test_remote_root_with_spaces_is_used_and_shell_quoted(self):
        image = install_main_dependencies(FakeImage(), self.source, "/opt/current app/")
        self.assertIn("/opt/current app/requirements-modal.txt", image.files)
        self.assertEqual(
            shlex.split(image.calls[-1][1]),
            ["python", "-m", "pip", "install", "-r", "/opt/current app/requirements-modal.txt"],
        )


class ModalImageCacheTests(unittest.TestCase):
    def build_layers(self, runtime_changes=None):
        class RecordingImage:
            def __init__(self, layers=()):
                self.layers = layers

            def __getattr__(self, method):
                def record(*args, **kwargs):
                    # Local manifests contribute their contents to the build cache.
                    if method == "add_local_file":
                        args = (*args, Path(args[0]).read_bytes())
                    kwargs = {key: value.__name__ if callable(value) else value
                              for key, value in kwargs.items()}
                    return RecordingImage((*self.layers, (method, args, kwargs)))
                return record

        source = ROOT / "deploy_vllm_indextts_v2.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "app"
                                                   for target in node.targets):
                break
            if isinstance(node, (ast.Assign, ast.If, ast.FunctionDef)):
                nodes.append(node)

        class ChangeRuntimeValues(ast.NodeTransformer):
            def visit_Dict(self, node):
                for index, key in enumerate(node.keys):
                    if isinstance(key, ast.Constant) and key.value in (runtime_changes or {}):
                        node.values[index] = ast.Constant(runtime_changes[key.value])
                return self.generic_visit(node)

        tree = ast.fix_missing_locations(ChangeRuntimeValues().visit(ast.Module(body=nodes, type_ignores=[])))
        namespace = {"Path": Path, "__file__": str(source),
                     "modal": SimpleNamespace(is_local=lambda: True, Image=RecordingImage())}
        exec(compile(tree, str(source), "exec"), namespace)
        return namespace["image"].layers

    def test_runtime_changes_reuse_every_build_layer(self):
        before = self.build_layers()
        after = self.build_layers({"HF_HOME": "/persistent_cache/other-hf",
                                   "TORCHINDUCTOR_COMPILE_THREADS": "2",
                                   "CLEARVOICE_PYTHON": "/opt/another-worker/bin/python"})
        self.assertEqual(before[:-1], after[:-1])
        self.assertEqual(before[-1][0], "env")
        self.assertNotEqual(before[-1], after[-1])
        # All runtime variables must be in the last layer, including ones not
        # exercised above; only compiler/build variables may precede installs.
        early_env = {key for method, args, _ in before[:-1] if method == "env" for key in args[0]}
        self.assertEqual(early_env, {"CUDA_HOME", "CUDA_PATH", "TORCH_CUDA_ARCH_LIST", "FORCE_CUDA", "CC", "CXX"})


@unittest.skipUnless(importlib.util.find_spec("modal"), "Modal SDK required for remote import test")
class ModalContainerImportTests(unittest.TestCase):
    def test_container_import_registers_services_without_checkout_or_build_files(self):
        import modal

        with tempfile.TemporaryDirectory(prefix="modal container import ") as directory:
            root = Path(directory)
            (root / "deploy_vllm_indextts_v2.py").write_bytes((ROOT / "deploy_vllm_indextts_v2.py").read_bytes())
            # Only the mounted deployment script and SDK are importable. This
            # reproduces /root in a Function container, without the checkout.
            script = '''
import importlib.abc
import sys
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])
import modal

class RejectApplicationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "indextts_web" or fullname.startswith("indextts_web."):
            raise ModuleNotFoundError("Application source is not on the container import path")

sys.meta_path.insert(0, RejectApplicationImports())
with patch.object(modal, "is_local", return_value=False), patch.object(
    modal.Image, "from_registry", side_effect=AssertionError("Remote import must not construct an image")
):
    import deploy_vllm_indextts_v2 as deploy
    assert deploy.app.name == "audio-studio"
    assert deploy.image is None
    assert isinstance(deploy.prepare_model, modal.Function)
    assert not hasattr(deploy, "run_setup_job")
    assert not hasattr(deploy, "environment_storage")
    assert not hasattr(deploy, "clear_cache")
    assert isinstance(deploy.IndexTTSVllmServer, modal.Cls)
print("Container import registered all services")
'''
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            result = subprocess.run(
                [sys.executable, "-c", script, str(Path(modal.__file__).resolve().parent.parent)],
                cwd=root, env=environment, capture_output=True, text=True, timeout=30, **options,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Container import registered all services", result.stdout)


if __name__ == "__main__":
    unittest.main()
