import ast
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from indextts_web.infrastructure.vllm_memory import wake_allocations
from tests.test_modal_gpu_profiles import ROOT


class WakeAllocationTests(unittest.TestCase):
    def setUp(self):
        self.entries = {
            ptr: SimpleNamespace(handle=(0, 100, ptr, ptr + 1), tag=tag,
                                 cpu_backup_tensor=SimpleNamespace(data_ptr=lambda: 1000, numel=lambda: 100,
                                                                  element_size=lambda: 1))
            for ptr, tag in ((10, "weights"), (20, "weights"), (30, "kv_cache"))
        }
        self.allocator = SimpleNamespace(pointer_to_data=self.entries)
        self.live = set()
        self.fail_at = None
        self.create = Mock(side_effect=self.allocate)
        self.release = Mock(side_effect=lambda handle: self.live.remove(handle[2]))
        self.copy = Mock()

    def allocate(self, handle):
        if handle[2] == self.fail_at:
            raise RuntimeError("CUDA Error: out of memory")
        if handle[2] in self.live:
            self.fail("attempted to remap a live allocation")
        self.live.add(handle[2])

    def wake(self, *, free=1000, tags=None):
        return wake_allocations(self.allocator, tags=tags, free_bytes=lambda: free, create=self.create,
                                release=self.release, copy=self.copy, headroom=20)

    def test_insufficient_memory_does_not_enter_allocator(self):
        with self.assertRaisesRegex(RuntimeError, "remains safely asleep"):
            self.wake(free=310)
        self.create.assert_not_called()
        self.assertTrue(all(entry.cpu_backup_tensor is not None for entry in self.entries.values()))

    def test_partial_oom_rolls_back_and_later_wake_succeeds(self):
        self.fail_at = 20
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.wake()
        self.assertEqual(self.live, set())
        self.assertTrue(all(entry.cpu_backup_tensor is not None for entry in self.entries.values()))
        self.fail_at = None
        self.wake()
        self.assertEqual(self.live, {10, 20, 30})
        self.assertTrue(all(entry.cpu_backup_tensor is None for entry in self.entries.values()))

    def test_copy_failure_preserves_backups_and_rolls_back_all_mappings(self):
        self.copy.side_effect = [None, RuntimeError("copy failed")]
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            self.wake()
        self.assertEqual(self.live, set())
        self.assertTrue(all(entry.cpu_backup_tensor is not None for entry in self.entries.values()))
        self.copy.side_effect = None
        self.wake()

    def test_tagged_wake_preserves_unselected_allocations(self):
        self.wake(tags=["weights"])
        self.assertEqual(self.live, {10, 20})
        self.assertIsNotNone(self.entries[30].cpu_backup_tensor)
        self.wake(tags=["kv_cache"])
        self.assertEqual(self.live, {10, 20, 30})

    def test_worker_adapter_restores_upstream_method_after_failure_and_retry(self):
        allocator = self.allocator
        original = Mock()
        allocator.wake_up = original

        class UpstreamWorker:
            def wake_up(self, tags):
                allocator.wake_up(tags)
                return "buffers restored"

        ns = {"Worker": UpstreamWorker, "wake_allocations": wake_allocations,
              "torch": SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (10**9, 10**9))),
              "cumem": SimpleNamespace(CuMemAllocator=SimpleNamespace(get_instance=lambda: allocator),
                                       create_and_map=self.create, unmap_and_release=self.release,
                                       libcudart=SimpleNamespace(cudaMemcpy=self.copy))}
        tree = ast.parse((ROOT / "indextts_web/infrastructure/vllm_worker.py").read_text())
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<worker>", "exec"), ns)
        worker = ns["RecoverableSleepWorker"]()
        self.fail_at = 20
        with self.assertRaises(RuntimeError):
            worker.wake_up()
        self.assertIs(allocator.wake_up, original)
        self.assertEqual(self.live, set())
        self.fail_at = None
        self.assertEqual(worker.wake_up(), "buffers restored")
        self.assertIs(allocator.wake_up, original)
