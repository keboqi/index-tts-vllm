import runpy
import sys
import unittest
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import APIRouter
from fastapi.testclient import TestClient

from indextts_web.api import build_routers
from indextts_web.app import create_app
from indextts_web.config import load_settings
from indextts_web.route_groups import route_group
from tests.support import ROOT
from tests.test_route_contract import EXPECTED_ROUTES


class AppAssemblyTests(unittest.TestCase):
    def source(self):
        source = ModuleType("test_production")
        source.SETTINGS = load_settings([], environ={})
        source.CONCURRENCY = Mock()
        source.tts_manager = SimpleNamespace(is_ready=lambda: False)
        source.app = APIRouter()
        source.events = []

        @asynccontextmanager
        async def lifespan(app):
            source.events.append("startup")
            source.runtime = app.state.runtime
            yield
            source.events.append("shutdown")

        source.lifespan = lifespan
        return source

    def test_all_public_routes_are_assembled_with_feature_tags(self):
        source = self.source()

        async def endpoint():
            return {"status": "ok"}

        for method, path in sorted(EXPECTED_ROUTES):
            source.app.add_api_route(path, endpoint, methods=[method])
        app = create_app(legacy=source)
        schema = app.openapi()
        actual = {(method.upper(), path) for path, operations in schema["paths"].items() for method in operations}
        self.assertEqual(actual, EXPECTED_ROUTES | {("GET", "/health")})
        for method, path in EXPECTED_ROUTES:
            self.assertEqual(schema["paths"][path][method.lower()]["tags"], [route_group(path)])

    def test_lifespan_health_and_endpoint_metadata_are_preserved(self):
        source = self.source()

        @source.app.post("/speak", status_code=202, summary="Generate speech", response_model=dict[str, str])
        async def speak():
            return {"result": "queued"}

        app = create_app(legacy=source)
        self.assertEqual(source.events, [])
        with TestClient(app) as client:
            self.assertIs(source.runtime, app.state.runtime)
            self.assertIs(source.runtime.backends, source.TTS_BACKEND_REGISTRY)
            self.assertEqual(source.events, ["startup"])
            response = client.post("/speak")
            self.assertEqual((response.status_code, response.json()), (202, {"result": "queued"}))
            self.assertEqual(client.get("/health").status_code, 503)
            source.tts_manager.is_ready = lambda: True
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["ready"])
            self.assertEqual(set(health.json()["tts_backends"]), {"index", "index25", "confucius"})
        self.assertEqual(source.events, ["startup", "shutdown"])
        operation = app.openapi()["paths"]["/speak"]["post"]
        self.assertEqual(operation["summary"], "Generate speech")
        self.assertIn("202", operation["responses"])

    def test_unclassified_and_duplicate_routes_fail_assembly(self):
        async def endpoint():
            return None

        routes = APIRouter()
        routes.add_api_route("/api/unclassified", endpoint, methods=["GET"])
        with self.assertRaisesRegex(RuntimeError, "unclassified"):
            build_routers(routes.routes)
        routes = APIRouter()
        routes.add_api_route("/speak", endpoint, methods=["POST"])
        routes.add_api_route("/speak", endpoint, methods=["POST"])
        with self.assertRaisesRegex(RuntimeError, "duplicate.*POST /speak"):
            build_routers(routes.routes)

    def test_public_launcher_exports_and_invokes_application_entrypoint(self):
        entrypoint = ModuleType("indextts_web.main")
        entrypoint.app = object()
        entrypoint.main = Mock()
        with patch.dict(sys.modules, {"indextts_web.main": entrypoint}):
            exports = runpy.run_path(str(ROOT / "fastapi_webui_v2.py"))
            self.assertIs(exports["app"], entrypoint.app)
            self.assertIs(exports["main"], entrypoint.main)
            entrypoint.main.assert_not_called()
            runpy.run_path(str(ROOT / "fastapi_webui_v2.py"), run_name="__main__")
            entrypoint.main.assert_called_once_with()
