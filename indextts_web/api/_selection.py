"""Route-selection helpers for compatibility-preserving extraction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from fastapi import APIRouter
from fastapi.routing import APIRoute

RoutePredicate = Callable[[str], bool]


def select_routes(source_app: Any, predicate: RoutePredicate) -> APIRouter:
    router = APIRouter()
    router.routes.extend(
        route
        for route in source_app.routes
        if isinstance(route, APIRoute) and predicate(route.path)
    )
    return router


def selected_paths(router: APIRouter) -> set[str]:
    return {route.path for route in router.routes if isinstance(route, APIRoute)}


def starts_with(*prefixes: str) -> RoutePredicate:
    return lambda path: path.startswith(prefixes)


def equals(*paths: str) -> RoutePredicate:
    allowed = frozenset(paths)
    return lambda path: path in allowed


def any_of(predicates: Iterable[RoutePredicate]) -> RoutePredicate:
    items = tuple(predicates)
    return lambda path: any(predicate(path) for predicate in items)

