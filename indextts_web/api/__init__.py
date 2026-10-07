"""Group production endpoints using one route inventory."""

from __future__ import annotations

from collections.abc import Iterable

from fastapi import APIRouter
from fastapi.routing import APIRoute

from ..route_groups import route_group


def build_routers(routes: Iterable[APIRoute]) -> list[tuple[str, APIRouter]]:
    routers: dict[str, APIRouter] = {}
    assigned: set[tuple[str, str]] = set()
    for route in routes:
        if not isinstance(route, APIRoute):
            continue
        tag = route_group(route.path)
        if tag is None:
            raise RuntimeError(f"unclassified production route: {route.path!r}")
        for method in route.methods:
            key = (method, route.path)
            if key in assigned:
                raise RuntimeError(f"duplicate production route: {method} {route.path}")
            assigned.add(key)
        if tag not in routers:
            routers[tag] = APIRouter()
        routers[tag].routes.append(route)
    return list(routers.items())
