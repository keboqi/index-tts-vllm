import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from indextts_web.infrastructure.concurrency import ConcurrencyBudget
from indextts_web.infrastructure.ffmpeg import Ffmpeg
from indextts_web.infrastructure.files import atomic_write_json, contained_path, safe_component
from indextts_web.infrastructure.processes import ProcessResult


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, args, *, timeout=None, env=None):
        self.calls.append((list(args), timeout, env))
        return ProcessResult(tuple(args), 0, "", "")


class InfrastructureTests(unittest.TestCase):
    def test_safe_component(self):
        self.assertEqual(safe_component("../../a song?.wav"), "a_song_.wav")
        self.assertEqual(safe_component("..."), "artifact")

    def test_contained_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = contained_path(root, "../../escape")
            self.assertEqual(result.parent, root.resolve())

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "record.json"
            atomic_write_json(target, {"text": "你好"})
            self.assertIn("你好", target.read_text(encoding="utf-8"))

    def test_ffmpeg_command_is_deterministic(self):
        runner = FakeRunner()
        ffmpeg = Ffmpeg(runner, threads=4)
        ffmpeg.transcode_audio(Path("input.wav"), Path("output.mp3"), codec_args=["-b:a", "128k"])
        command, timeout, _env = runner.calls[0]
        self.assertEqual(command[:7], ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "4"])
        self.assertEqual(command[-5:], ["input.wav", "-vn", "-b:a", "128k", "output.mp3"])
        self.assertEqual(timeout, 600)

    def test_concurrency_budget_preserves_nested_higgs_defaults(self):
        with patch.dict(
            "os.environ",
            {
                "INDEXTTS_GPU_WORK_CONCURRENCY": "8",
                "TRANSLATION_TTS_CONCURRENCY": "40",
                "HIGGS_TTS_MAX_RUNNING_REQUESTS": "12",
                "HIGGS_TTS_WORK_CONCURRENCY": "10",
            },
            clear=True,
        ):
            budget = ConcurrencyBudget.from_environ()
        try:
            self.assertEqual(budget.index_tts_requests, 8)
            self.assertEqual(budget.translation_tts_requests, 8)
            self.assertEqual(budget.higgs_workers, 10)
        finally:
            budget.shutdown()


if __name__ == "__main__":
    unittest.main()
