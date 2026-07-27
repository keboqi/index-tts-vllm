from __future__ import annotations

import unittest
from unittest import mock

import moss_transcribe_pipeline as pipeline


class MossBackendSelectionTests(unittest.TestCase):
    def test_auto_starts_managed_sglang_before_native_fallback(self) -> None:
        response = {"text": "managed"}
        with (
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_BACKEND", "auto"),
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_MANAGE_BACKEND", True),
            mock.patch.object(pipeline, "_check_health", return_value=False),
            mock.patch.object(pipeline, "_ensure_backend_ready") as ensure_ready,
            mock.patch.object(
                pipeline,
                "_transcribe_with_sglang",
                return_value=response,
            ) as transcribe_sglang,
            mock.patch.object(pipeline, "_transcribe_with_python") as transcribe_python,
        ):
            result = pipeline._transcribe(b"audio", input_mime_type="audio/wav")

        self.assertEqual(result, response)
        ensure_ready.assert_called_once_with()
        transcribe_sglang.assert_called_once()
        transcribe_python.assert_not_called()

    def test_auto_uses_native_only_when_it_is_installed(self) -> None:
        response = {"text": "native"}
        with (
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_BACKEND", "auto"),
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_MANAGE_BACKEND", True),
            mock.patch.object(pipeline, "_check_health", return_value=False),
            mock.patch.object(
                pipeline,
                "_ensure_backend_ready",
                side_effect=RuntimeError("docker unavailable"),
            ),
            mock.patch.object(
                pipeline,
                "_python_backend_available",
                return_value=True,
            ),
            mock.patch.object(
                pipeline,
                "_transcribe_with_python",
                return_value=response,
            ) as transcribe_python,
        ):
            result = pipeline._transcribe(b"audio", input_mime_type="audio/wav")

        self.assertEqual(result, response)
        transcribe_python.assert_called_once()

    def test_auto_reports_both_unavailable_backends(self) -> None:
        with (
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_BACKEND", "auto"),
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_MANAGE_BACKEND", True),
            mock.patch.object(pipeline, "_check_health", return_value=False),
            mock.patch.object(
                pipeline,
                "_ensure_backend_ready",
                side_effect=RuntimeError("docker unavailable"),
            ),
            mock.patch.object(
                pipeline,
                "_python_backend_available",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "native `moss_transcribe_diarize` fallback is not installed",
            ):
                pipeline._transcribe(b"audio", input_mime_type="audio/wav")
