import unittest
from typing import Any

from indextts_web.infrastructure.callables import filter_supported_keyword_arguments
from tests.support import ROOT, load_definition


class WhisperXDiarizationCompatibilityTests(unittest.TestCase):
    def load(self, factory):
        return load_definition(
            ROOT / "whisperx_pipeline.py", "_load_whisperx_diarization_pipeline",
            {"Any": Any, "DiarizationPipeline": factory, "WHISPERX_PYANNOTE_CACHE": "model-cache",
             "filter_supported_keyword_arguments": filter_supported_keyword_arguments},
        )

    def test_older_constructor_receives_token_under_its_supported_name(self):
        def factory(use_auth_token=None, device="cpu"):
            return {"authentication": use_auth_token, "device": device}

        self.assertEqual(self.load(factory)("hf-test-token", "cuda"),
                         {"authentication": "hf-test-token", "device": "cuda"})

    def test_current_constructor_receives_authentication_and_cache_directory(self):
        def factory(token=None, device="cpu", cache_dir=None):
            return {"authentication": token, "device": device, "cache_dir": cache_dir}

        self.assertEqual(self.load(factory)("hf-test-token", "cuda"),
                         {"authentication": "hf-test-token", "device": "cuda", "cache_dir": "model-cache"})

    def test_model_loading_failure_is_propagated_without_retrying_unauthenticated(self):
        calls = []

        def factory(use_auth_token=None, device="cpu"):
            calls.append(use_auth_token)
            raise RuntimeError("gated model access denied")

        with self.assertRaisesRegex(RuntimeError, "gated model access denied"):
            self.load(factory)("hf-test-token", "cuda")
        self.assertEqual(calls, ["hf-test-token"])
