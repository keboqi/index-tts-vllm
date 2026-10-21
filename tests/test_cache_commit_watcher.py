import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.support import ROOT, load_definition


class CacheCommitWatcherTests(unittest.TestCase):
    def test_failed_commit_retries_unchanged_marker_then_stops_recommitting(self):
        for failing_volume in ("cache_storage", "app_storage"):
            with self.subTest(failing_volume=failing_volume):
                request = Mock()
                request.stat.return_value.st_mtime = 123.0
                namespace = {
                    "Path": Mock(return_value=request),
                    "DRAMA3_COMMIT_REQUEST_FILE": "/tmp/test-commit.request",
                    "time": SimpleNamespace(sleep=Mock(side_effect=[None, None, None, StopIteration])),
                    "cache_storage": Mock(),
                    "app_storage": Mock(),
                    "print": Mock(),
                }
                namespace[failing_volume].commit.side_effect = [RuntimeError("temporary failure"), None]
                start = load_definition(ROOT / "deploy_vllm_indextts_v2.py",
                                        "_start_cache_commit_watcher", namespace)
                with patch("threading.Thread") as thread:
                    start()
                    watch = thread.call_args.kwargs["target"]
                    with self.assertRaises(StopIteration):
                        watch()
                self.assertEqual(namespace["cache_storage"].commit.call_count, 2)
                self.assertEqual(namespace["app_storage"].commit.call_count,
                                 1 if failing_volume == "cache_storage" else 2)
