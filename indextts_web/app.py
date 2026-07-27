"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import build_routers
from .api.health import router as health_router
from .runtime import RuntimeContainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_ROOT = PROJECT_ROOT / "static"


def load_legacy_module() -> ModuleType:
    """Load the current production implementation only during app assembly."""
    return import_module("legacy_fastapi_webui_v2")


def create_app(*, legacy: ModuleType | None = None) -> FastAPI:
    production = load_legacy_module() if legacy is None else legacy
    runtime = RuntimeContainer.from_legacy(production)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime
        async with production.lifespan(app):
            yield

    application = FastAPI(
        title="IndexTTS vLLM v2 FastAPI WebUI",
        description=(
            "Ultra-fast TTS with vLLM, speaker presets, external backends, "
            "and advanced translate/edit workflows"
        ),
        lifespan=lifespan,
    )
    if STATIC_ROOT.exists():
        application.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
    application.include_router(health_router, tags=["health"])
    for _tag, router in build_routers(production.app):
        application.include_router(router)
    application.state.runtime = runtime
    return application
