import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from indextts_web.services.tts.factory import build_backend_registry


class BackendLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.confucius = SimpleNamespace(
            status=AsyncMock(return_value={"ready": True, "backend": "confucius"}),
            shutdown=AsyncMock(),
        )
        self.index25 = SimpleNamespace(
            status=AsyncMock(return_value={"ready": True, "backend": "index25"}),
            shutdown=AsyncMock(),
        )
        self.legacy = SimpleNamespace(
            SETTINGS=SimpleNamespace(tts_backend="index"),
            tts_manager=SimpleNamespace(
                is_ready=lambda: True,
                vllm_status=lambda: {"indextts_vllm_sleeping": False},
            ),
            confucius_backend_manager=self.confucius,
            indextts25_backend_manager=self.index25,
        )
        self.registry = build_backend_registry(self.legacy)

    async def test_registered_backends_report_their_manager_status(self):
        self.assertEqual(
            await self.registry.get("index").status(),
            {"ready": True, "indextts_vllm_sleeping": False},
        )
        for name, manager in (("confucius", self.confucius), ("index25", self.index25)):
            with self.subTest(backend=name):
                self.assertEqual(
                    await self.registry.get(name).status(),
                    {"ready": True, "backend": name},
                )
                manager.status.assert_awaited_once_with()

    async def test_registry_shutdown_reaches_both_external_managers(self):
        await self.registry.shutdown()
        self.confucius.shutdown.assert_awaited_once_with()
        self.index25.shutdown.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
