"""Executable entry point."""

from __future__ import annotations

from typing import Any

from .app import create_app

app = create_app()


def main() -> None:
    import uvicorn

    settings: Any = app.state.runtime.settings
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        workers=1,
        limit_concurrency=100,
        backlog=2048,
        timeout_keep_alive=300,
        h11_max_incomplete_event_size=16_777_216,
        access_log=True,
    )


if __name__ == "__main__":
    main()

