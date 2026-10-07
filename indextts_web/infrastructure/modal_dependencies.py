"""Install Modal's main dependencies from the deploying checkout."""

from __future__ import annotations

import shlex
from pathlib import Path, PurePosixPath
from typing import Any

MAIN_DEPENDENCY_FILES = (
    "requirements.txt",
    "requirements-core.txt",
    "requirements-modal.txt",
    "constraints-main.txt",
)


def install_main_dependencies(
    image: Any,
    source_root: Path | str,
    remote_root: str = "/app/index-tts-vllm",
) -> Any:
    """Copy local manifests before installation so their contents key the cache."""
    source = Path(source_root).resolve()
    paths = [(source / name, PurePosixPath(remote_root) / name) for name in MAIN_DEPENDENCY_FILES]
    missing = [str(local) for local, _remote in paths if not local.is_file()]
    if missing:
        raise FileNotFoundError(f"Modal dependency manifests missing: {', '.join(missing)}")

    for local, remote in paths:
        image = image.add_local_file(local, str(remote), copy=True)
    requirements = PurePosixPath(remote_root) / "requirements-modal.txt"
    return image.run_commands(f"python -m pip install -r {shlex.quote(str(requirements))}")
