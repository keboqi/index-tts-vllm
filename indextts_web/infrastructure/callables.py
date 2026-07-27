"""Compatibility helpers for optional third-party callables."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any


def filter_supported_keyword_arguments(
    factory: Callable[..., Any],
    keyword_arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return keyword arguments accepted by ``factory`` and names that were dropped.

    Optional integrations evolve independently of the WebUI. Introspection lets
    us retain newer tuning flags while remaining compatible with older releases
    that do not expose those constructor parameters yet.
    """

    values = dict(keyword_arguments)
    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return values, ()

    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return values, ()

    supported_names = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    accepted = {name: value for name, value in values.items() if name in supported_names}
    dropped = tuple(name for name in values if name not in supported_names)
    return accepted, dropped
