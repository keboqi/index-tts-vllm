"""Injectable subprocess boundary used by media and managed backends."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ProcessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class ProcessRunner(Protocol):
    def run(
        self,
        args: Sequence[str],
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult: ...


class SubprocessRunner:
    def run(
        self,
        args: Sequence[str],
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        command = [str(item) for item in args]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=None if env is None else dict(env),
            check=False,
        )
        return ProcessResult(tuple(command), completed.returncode, completed.stdout, completed.stderr)
