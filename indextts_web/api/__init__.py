"""Feature router inventory."""

from __future__ import annotations

from typing import Any

from fastapi.routing import APIRoute

from . import internal, models, speakers, stable_audio, translation, tts, ui, utilities, video
from ._selection import selected_paths

ROUTER_BUILDERS = (
    ("ui", ui.build_router),
    ("internal", internal.build_router),
    ("utilities", utilities.build_router),
    ("models", models.build_router),
    ("stable-audio", stable_audio.build_router),
    ("video", video.build_router),
    ("translation", translation.build_router),
    ("speakers", speakers.build_router),
    ("tts", tts.build_router),
)


def build_routers(source_app: Any):
    routers = []
    assigned: dict[str, str] = {}
    for tag, builder in ROUTER_BUILDERS:
        router = builder(source_app)
        for path in selected_paths(router):
            previous = assigned.setdefault(path, tag)
            if previous != tag:
                raise RuntimeError(f"route {path!r} belongs to both {previous!r} and {tag!r}")
        routers.append((tag, router))

    source_paths = {
        route.path
        for route in source_app.routes
        if isinstance(route, APIRoute)
    }
    missing = source_paths - assigned.keys()
    if missing:
        raise RuntimeError(f"unclassified compatibility routes: {sorted(missing)}")
    return routers

