from typing import Any

from ..route_groups import route_group
from ._selection import select_routes


def build_router(source_app: Any):
    return select_routes(source_app, lambda path: route_group(path) == "ui")
