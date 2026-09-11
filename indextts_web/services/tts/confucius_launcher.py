"""Forward GPU options through Confucius versions with a narrower public CLI.

The upstream runtime already accepts EngineArgs via engine_kwargs. This
process-local adapter leaves its checkout and sampling implementation intact.
"""

from __future__ import annotations

import functools
import runpy
from pathlib import Path
from typing import Any

from ...gpu_profiles import runtime_gpu_profile


def install_engine_options(runtime_class: type, options: dict[str, Any]) -> None:
    original = runtime_class.__init__

    @functools.wraps(original)
    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        # Explicit upstream kwargs take precedence over the capacity profile.
        engine_kwargs = {**options, **(kwargs.get("engine_kwargs") or {})}
        # The public CLI's memory override is already forwarded positionally or
        # by keyword; do not override it again inside engine_kwargs.
        engine_kwargs.pop("gpu_memory_utilization", None)
        kwargs["engine_kwargs"] = engine_kwargs
        original(self, *args, **kwargs)

    runtime_class.__init__ = initialize


def main() -> None:
    from confuciustts.llm.vllm_runtime import Text2SemanticVLLM

    profile = runtime_gpu_profile()
    install_engine_options(Text2SemanticVLLM, profile.confucius.kwargs())
    entrypoint = Path.cwd() / "fastapi_app.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Confucius entry point missing: {entrypoint}")
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
