from __future__ import annotations

import unittest
from unittest import mock

import moss_transcribe_pipeline as pipeline


class MossBackendSelectionTests(unittest.TestCase):
    def test_colab_http_backend_works_without_native_package_or_docker(self) -> None:
        response = {"text": "local Transformers service"}
        with (
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_BACKEND", "http"),
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_MANAGE_BACKEND", False),
            mock.patch.object(pipeline, "_check_health", return_value=True),
            mock.patch.object(pipeline, "_python_backend_available", return_value=False),
            mock.patch.object(pipeline, "_start_managed_backend") as managed_start,
            mock.patch.object(pipeline, "_transcribe_with_python") as native,
            mock.patch.object(pipeline, "_transcribe_with_sglang", return_value=response) as http,
        ):
            result = pipeline._transcribe(b"audio", input_mime_type="audio/wav")
        self.assertEqual(result, response)
        http.assert_called_once()
        managed_start.assert_not_called()
        native.assert_not_called()

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


class MossOnDemandServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(setattr, pipeline, "_SERVER_PROCESS", None)

    def test_first_request_launches_bundled_server_once(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        health = iter([False, False, False, True])
        with (
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_MANAGE_BACKEND", True),
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_SERVER_PYTHON", "/opt/moss/bin/python"),
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_SGLANG_URL", "http://127.0.0.1:8003"),
            mock.patch("os.path.isfile", return_value=True),
            mock.patch.object(pipeline, "_check_health", side_effect=lambda **_: next(health)),
            mock.patch.object(pipeline, "_resolve_manager_script") as script,
            mock.patch.object(pipeline.subprocess, "Popen", return_value=process) as popen,
            mock.patch.object(pipeline.time, "sleep"),
        ):
            self.assertTrue(pipeline.can_launch_moss_server())
            pipeline.ensure_moss_server()
            pipeline._start_server_process()
        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertEqual(command[:4], ["/opt/moss/bin/python", "-m", "uvicorn", "moss_transcribe_server:app"])
        self.assertEqual(command[-1], "8003")
        script.assert_not_called()

    def test_server_exit_fails_fast_and_stop_terminates(self) -> None:
        process = mock.Mock(returncode=3)
        process.poll.return_value = 3
        with (
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_MANAGE_BACKEND", True),
            mock.patch.object(pipeline, "MOSS_TRANSCRIBE_SERVER_PYTHON", "/opt/moss/bin/python"),
            mock.patch("os.path.isfile", return_value=True),
            mock.patch.object(pipeline, "_check_health", return_value=False),
            mock.patch.object(pipeline.subprocess, "Popen", return_value=process),
        ):
            with self.assertRaisesRegex(RuntimeError, "exited with code 3"):
                pipeline.ensure_moss_server()
        running = mock.Mock()
        running.poll.return_value = None
        pipeline._SERVER_PROCESS = running
        pipeline.stop_moss_server()
        running.terminate.assert_called_once_with()
        self.assertIsNone(pipeline._SERVER_PROCESS)
