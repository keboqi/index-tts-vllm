import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from indextts_web.infrastructure.concurrency import ConcurrencyBudget
from indextts_web.infrastructure.files import atomic_write_json


class InfrastructureTests(unittest.TestCase):
    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "record.json"
            atomic_write_json(target, {"text": "你好"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"text": "你好"})
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_failed_atomic_write_preserves_existing_record_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "record.json"
            atomic_write_json(target, {"text": "original"})
            with patch.object(Path, "replace", side_effect=OSError("disk unavailable")):
                with self.assertRaisesRegex(OSError, "disk unavailable"):
                    atomic_write_json(target, {"text": "replacement"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"text": "original"})
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_concurrency_budget_bounds_translation_to_index_capacity(self):
        with patch.dict(
            "os.environ",
            {
                "INDEXTTS_GPU_WORK_CONCURRENCY": "8",
                "TRANSLATION_TTS_CONCURRENCY": "40",
            },
            clear=True,
        ):
            budget = ConcurrencyBudget.from_environ()
        try:
            self.assertEqual(budget.index_tts_requests, 8)
            self.assertEqual(budget.translation_tts_requests, 8)
        finally:
            budget.shutdown()


if __name__ == "__main__":
    unittest.main()
