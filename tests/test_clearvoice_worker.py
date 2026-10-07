from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

from indextts_web.infrastructure.files import atomic_write_json
from indextts_web.services.audio import clearvoice_worker as worker
from tests.support import ROOT, load_definition


class ClearVoiceWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="clearvoice test ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "输入.wav"
        self.source.write_bytes(b"input audio")
        environment = patch.dict(os.environ, {worker.PYTHON_ENV: sys.executable,
                                              worker.TIMEOUT_ENV: "10"})
        environment.start()
        self.addCleanup(environment.stop)
        self.calls = []

    def factory(self, *, task, model_names):
        self.calls.append((task, model_names))

        def inference(*, input_path, online_write):
            self.calls.append((input_path, online_write))
            return Path(input_path).read_bytes() + task.encode()

        def write(output, *, output_path):
            Path(output_path).write_bytes(output)

        return SimpleNamespace(__call__=inference, write=write)

    def child_factory(self, **options):
        model = self.factory(**options)

        class CallableModel:
            def __call__(self, **kwargs):
                return model.__call__(**kwargs)

            write = staticmethod(model.write)

        return CallableModel()

    def launch(self, command, **kwargs):
        self.command, self.launch_options = command, kwargs
        request_path = Path(command[-1])
        self.request = json.loads(request_path.read_text(encoding="utf-8"))
        code = worker.run_job(request_path, clearvoice_factory=self.child_factory)
        return subprocess.CompletedProcess(command, code)

    def process(self, enhancement=True, super_resolution=True):
        return worker.process(str(self.source), enhancement, super_resolution,
                              enhancement_model_name="MossFormerGAN_SE_16K", app_dir=self.root)

    def test_job_preserves_transform_paths_and_separate_interpreter_launch(self):
        with patch.object(worker.subprocess, "run", side_effect=self.launch):
            final, generated, enhanced = self.process()
        base = Path(self.request["output_base"])
        expected = [str(base.with_stem(base.stem + "_se")), str(base.with_stem(base.stem + "_se_sr"))]
        self.assertEqual((final, generated, enhanced), (expected[-1], expected, expected[0]))
        self.assertEqual(self.source.read_bytes(), b"input audio")
        self.assertEqual(self.calls, [
            ("speech_enhancement", ["MossFormerGAN_SE_16K"]), (str(self.source), False),
            ("speech_super_resolution", ["MossFormer2_SR_48K"]), (expected[0], False),
        ])
        self.assertEqual(self.command[1:4], ["-u", "-m", "indextts_web.services.audio.clearvoice_worker"])
        self.assertEqual(self.launch_options["cwd"], str(self.root))
        self.assertEqual(self.launch_options["env"]["PYTHONPATH"].split(os.pathsep)[0], str(self.root))
        self.assertEqual(self.launch_options["timeout"], 10)
        if os.name == "nt":
            self.assertEqual(self.launch_options["creationflags"], subprocess.CREATE_NO_WINDOW)
        self.assertFalse(Path(self.request["response_path"]).exists())

    def test_super_resolution_without_enhancement_has_original_suffix_contract(self):
        with patch.object(worker.subprocess, "run", side_effect=self.launch):
            result = self.process(enhancement=False)
        self.assertEqual(result[1], [result[0]])
        self.assertIsNone(result[2])
        self.assertTrue(Path(result[0]).stem.endswith("_sr"))
        self.assertEqual(self.calls[-1], (str(self.source), False))

    def test_child_failure_cleans_complete_and_partial_outputs(self):
        original_factory = self.child_factory

        def failing_factory(**options):
            model = original_factory(**options)
            if options["task"] == "speech_super_resolution":
                def fail(output, *, output_path):
                    Path(output_path).write_bytes(b"partial")
                    raise RuntimeError("super-resolution failure")
                model.write = fail
            return model

        with patch.object(self, "child_factory", side_effect=failing_factory), \
                patch.object(worker.subprocess, "run", side_effect=self.launch), \
                patch.object(worker.traceback, "print_exc"), \
                self.assertRaisesRegex(RuntimeError, "super-resolution failure"):
            self.process()
        self.assertEqual(list(self.root.glob("*.wav")), [self.source])

    def test_timeout_and_missing_or_malformed_result_clean_outputs(self):
        for mode in ("timeout", "missing", "invalid", "wrong-path"):
            def fail(command, _mode=mode, **kwargs):
                request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
                base = Path(request["output_base"])
                base.with_stem(base.stem + "_se").write_bytes(b"partial")
                if _mode == "timeout":
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                if _mode == "invalid":
                    Path(request["response_path"]).write_text("[]", encoding="utf-8")
                if _mode == "wrong-path":
                    atomic_write_json(Path(request["response_path"]), {"final_path": str(self.source)})
                return subprocess.CompletedProcess(command, 0)

            with self.subTest(mode=mode), patch.object(worker.subprocess, "run", side_effect=fail), \
                    self.assertRaisesRegex(RuntimeError, "exceeded|without a result|invalid result"):
                self.process()
            self.assertEqual(list(self.root.glob("*.wav")), [self.source])

    def test_invalid_interpreter_and_timeout_are_unavailable_before_launch(self):
        with patch.object(worker.subprocess, "run") as launch:
            with patch.dict(os.environ, {worker.PYTHON_ENV: str(self.root / "missing-python")}):
                self.assertFalse(worker.is_available(local_available=True))
                with self.assertRaisesRegex(RuntimeError, "executable Python"):
                    self.process()
            for timeout in ("0", "nan", "inf", "86401"):
                with patch.dict(os.environ, {worker.TIMEOUT_ENV: timeout}), \
                        self.assertRaisesRegex(ValueError, "between 0 and 86400"):
                    self.process()
            launch.assert_not_called()

    def test_repeat_jobs_have_distinct_outputs_and_preserve_preexisting_files(self):
        existing = self.source.with_stem("输入_se")
        existing.write_bytes(b"another job")
        with patch.object(worker.subprocess, "run", side_effect=self.launch):
            first = self.process()
            second = self.process()
        self.assertTrue(set(first[1]).isdisjoint(second[1]))
        self.assertTrue(all(Path(path).is_file() for path in first[1] + second[1]))
        self.assertEqual(existing.read_bytes(), b"another job")

    def test_real_child_imports_clearvoice_without_loading_web_application(self):
        (self.root / "clearvoice.py").write_text('''
import sys
from pathlib import Path
class ClearVoice:
    def __init__(self, **options):
        assert "fastapi_webui_v2_impl" not in sys.modules
    def __call__(self, *, input_path, online_write):
        return Path(input_path).read_bytes() + b"processed"
    def write(self, output, *, output_path):
        Path(output_path).write_bytes(output)
''', encoding="utf-8")
        with patch.dict(os.environ, {"PYTHONPATH": str(ROOT)}):
            result = self.process()
        self.assertEqual(Path(result[0]).read_bytes(), b"input audioprocessedprocessed")

    def test_real_timeout_kills_and_reaps_child(self):
        (self.root / "clearvoice.py").write_text('''
import time
class ClearVoice:
    def __init__(self, **options):
        time.sleep(60)
''', encoding="utf-8")
        children = []
        original = subprocess.Popen

        def start(*args, **options):
            child = original(*args, **options)
            children.append(child)
            return child

        with patch.dict(os.environ, {"PYTHONPATH": str(ROOT), worker.TIMEOUT_ENV: "0.3"}), \
                patch.object(worker.subprocess, "Popen", side_effect=start), \
                self.assertRaisesRegex(RuntimeError, "exceeded"):
            self.process()
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertEqual(list(self.root.glob("*.wav")), [self.source])


class ClearVoiceRoutingTests(unittest.TestCase):
    def test_default_direct_model_cache_remains_unchanged(self):
        model = Mock()
        factory = Mock(return_value=model)
        module = SimpleNamespace(is_configured=Mock(return_value=False),
                                 is_available=Mock(return_value=True), process=Mock())
        namespace = {"Optional": Optional, "Tuple": tuple, "List": list, "print": Mock(),
                     "ClearVoice": factory, "clearvoice_worker": module,
                     "AVAILABLE_ENHANCEMENT_MODELS": ["default"], "DEFAULT_ENHANCEMENT_MODEL": "default",
                     "_enhancement_model": None, "_super_res_model": None,
                     "_current_enhancement_model_name": None, "os": os,
                     "_append_suffix_to_path": lambda path, suffix: str(Path(path).with_stem(Path(path).stem + suffix))}
        direct = load_definition(ROOT / "fastapi_webui_v2_impl.py", "_apply_clearvoice_processing_sync", namespace)
        expected = ("input_se.wav", ["input_se.wav"], "input_se.wav")
        self.assertEqual(direct("input.wav", True, False), expected)
        self.assertEqual(direct("input.wav", True, False), expected)
        factory.assert_called_once_with(task="speech_enhancement", model_names=["default"])
        module.process.assert_not_called()

    def test_isolated_path_bypasses_optional_main_import_and_cached_models(self):
        expected = ("input_se.wav", ["input_se.wav"], "input_se.wav")
        module = SimpleNamespace(is_configured=Mock(return_value=True), process=Mock(return_value=expected))
        namespace = {"Optional": Optional, "Tuple": tuple, "List": list,
                     "ClearVoice": None, "clearvoice_worker": module,
                     "AVAILABLE_ENHANCEMENT_MODELS": ["default"], "DEFAULT_ENHANCEMENT_MODEL": "default"}
        routed = load_definition(ROOT / "fastapi_webui_v2_impl.py", "_apply_clearvoice_processing_sync", namespace)
        self.assertEqual(routed("input.wav", True, False, "invalid"), expected)
        module.process.assert_called_once_with("input.wav", True, False, enhancement_model_name="default")

    def test_configured_parallel_chunks_keep_user_concurrency_without_spawned_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            barrier = threading.Barrier(2)
            forbidden_pool = Mock(side_effect=AssertionError("must not spawn web application processes"))

            def run_chunk(job):
                barrier.wait(timeout=5)
                output = Path(job.chunk_path).with_stem(Path(job.chunk_path).stem + "_isolated_se")
                output.write_bytes(b"enhanced")
                return SimpleNamespace(chunk_idx=job.chunk_idx, final_path=str(output),
                                       enhancement_path=str(output), generated_paths=[str(output)])

            namespace = {
                "Optional": Optional, "Tuple": tuple, "List": list, "Set": set, "print": Mock(),
                "ClearVoiceParallelConfig": SimpleNamespace, "ClearVoiceParallelChunkJob": SimpleNamespace,
                "ClearVoiceParallelChunkResult": SimpleNamespace,
                "clearvoice_worker": SimpleNamespace(is_configured=lambda: True),
                "ThreadPoolExecutor": ThreadPoolExecutor, "ProcessPoolExecutor": forbidden_pool,
                "as_completed": as_completed, "os": os,
                "tempfile": SimpleNamespace(mkstemp=lambda **options: tempfile.mkstemp(dir=root, **options)),
                "time": SimpleNamespace(perf_counter=lambda: 0), "_ffmpeg_available": lambda: True,
                "_plan_clearvoice_parallel_chunks": lambda *args: [(0, 1000), (1000, 2000)],
                "_run_clearvoice_chunk_job": run_chunk,
                "_append_suffix_to_path": lambda path, suffix: str(Path(path).with_stem(Path(path).stem + suffix)),
                "_ffmpeg_extract_segment": lambda src, out, *args, **kwargs: Path(out).write_bytes(b"chunk"),
                "_ffmpeg_concat_files": lambda sources, out, **kwargs: Path(out).write_bytes(b"merged"),
                "_safe_remove_file": lambda path: Path(path).unlink(missing_ok=True),
            }
            parallel = load_definition(ROOT / "fastapi_webui_v2_impl.py", "_apply_clearvoice_parallel_sync", namespace)
            result = parallel("input.wav", 2000, True, False,
                              SimpleNamespace(enabled=True, max_workers=2, chunk_ms=1000))
            self.assertEqual(Path(result[0]).read_bytes(), b"merged")
            self.assertEqual(Path(result[2]).read_bytes(), b"merged")
            forbidden_pool.assert_not_called()
            for path in result[1]:
                Path(path).unlink(missing_ok=True)

            def fail_second_chunk(job):
                completed = run_chunk(job)
                if job.chunk_idx == 1:
                    Path(completed.final_path).unlink()
                    raise RuntimeError("one chunk failed")
                return completed

            namespace["_run_clearvoice_chunk_job"] = fail_second_chunk
            with self.assertRaisesRegex(RuntimeError, "one chunk failed"):
                parallel("input.wav", 2000, True, False,
                         SimpleNamespace(enabled=True, max_workers=2, chunk_ms=1000))
            self.assertEqual(list(root.iterdir()), [])

    def test_availability_without_configuration_follows_local_installation(self):
        with patch.dict(os.environ, {worker.PYTHON_ENV: ""}):
            self.assertTrue(worker.is_available(local_available=True))
            self.assertFalse(worker.is_available(local_available=False))

if __name__ == "__main__":
    unittest.main()
