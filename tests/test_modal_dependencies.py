import shlex
import tempfile
import unittest
from pathlib import Path

from indextts_web.infrastructure.modal_dependencies import MAIN_DEPENDENCY_FILES, install_main_dependencies


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


if __name__ == "__main__":
    unittest.main()
