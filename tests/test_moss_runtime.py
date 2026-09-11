import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from indextts_web.services.translation.moss_runtime import MossRuntime


class FakeModel:
    def __init__(self):
        self.device = "cuda:0"
        self.moves = []

    def to(self, device):
        self.device = device
        self.moves.append(device)
        return self


class MossRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.models = []
        self.loads = 0

        def loader():
            model = FakeModel()
            self.models.append(weakref.ref(model))
            self.loads += 1
            return model, SimpleNamespace(), "cuda:0", "bfloat16"

        self.release = Mock()
        self.runtime = MossRuntime(loader=loader, release_cache=self.release)

    def test_lazy_load_sleep_wake_unload_and_reload(self):
        self.assertEqual(self.runtime.status()["state"], "unloaded")
        self.runtime.change("sleep")
        self.assertEqual(self.loads, 0)
        self.assertEqual(self.runtime.generate(lambda m, *_: m.device), "cuda:0")
        self.assertEqual(self.loads, 1)
        self.runtime.change("sleep")
        self.runtime.change("sleep")
        self.assertEqual(self.runtime.status()["state"], "sleeping")
        self.assertEqual(self.models[0]().moves, ["cpu"])
        self.release.assert_called_once()
        self.runtime.change("wake")
        self.assertEqual(self.models[0]().moves, ["cpu", "cuda:0"])
        self.assertEqual(self.loads, 1)
        self.runtime.change("sleep")
        self.assertEqual(self.runtime.generate(lambda m, *_: m.device), "cuda:0")
        self.runtime.change("unload")
        self.assertIsNone(self.models[0]())
        self.assertEqual(self.runtime.status()["state"], "unloaded")
        self.runtime.generate(lambda *_: "transcript")
        self.assertEqual(self.loads, 2)
        self.runtime.change("unload")
        self.runtime.change("wake")
        self.assertEqual(self.loads, 3)

    def test_sleeping_model_can_be_fully_unloaded(self):
        self.runtime.change("wake")
        self.runtime.change("sleep")
        self.runtime.change("unload")
        self.assertIsNone(self.models[0]())
        self.assertEqual(self.runtime.status()["state"], "unloaded")

    def test_failed_automatic_wake_discards_partial_runtime(self):
        self.runtime.change("wake")
        self.runtime.change("sleep")
        self.models[0]().to = Mock(side_effect=RuntimeError("out of memory"))
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.runtime.generate(lambda *_: self.fail("must not generate"))
        self.assertEqual(self.runtime.status()["state"], "unloaded")
        self.assertFalse(self.runtime.status()["busy"])
        self.runtime.change("wake")
        self.assertEqual(self.loads, 2)

    def test_management_waits_for_inference_and_status_stays_responsive(self):
        for action in ("sleep", "unload"):
            with self.subTest(action=action), ThreadPoolExecutor(max_workers=2) as executor:
                started, finish, changing = threading.Event(), threading.Event(), threading.Event()

                def inference(model, *_, started=started, finish=finish):
                    started.set()
                    if not finish.wait(timeout=5):
                        raise TimeoutError("test did not release inference")
                    self.assertEqual(model.device, "cuda:0")
                    return "transcript"

                def change(changing=changing, action=action):
                    changing.set()
                    return self.runtime.change(action)

                job = executor.submit(self.runtime.generate, inference)
                try:
                    self.assertTrue(started.wait(timeout=2))
                    management = executor.submit(change)
                    self.assertTrue(changing.wait(timeout=2))
                    self.assertTrue(self.runtime.status()["busy"])
                    self.assertFalse(management.done())
                finally:
                    finish.set()
                self.assertEqual(job.result(timeout=2), "transcript")
                self.assertEqual(management.result(timeout=2)["state"],
                                 "sleeping" if action == "sleep" else "unloaded")
                self.assertFalse(self.runtime.status()["busy"])
