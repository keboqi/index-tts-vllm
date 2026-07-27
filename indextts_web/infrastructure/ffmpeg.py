"""Small, testable FFmpeg command builder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .processes import ProcessResult, ProcessRunner


@dataclass(frozen=True, slots=True)
class Ffmpeg:
    runner: ProcessRunner
    executable: str = "ffmpeg"
    threads: int = 0

    def command(self, *arguments: str, overwrite: bool = True) -> list[str]:
        command = [self.executable, "-hide_banner", "-loglevel", "error"]
        command.append("-y" if overwrite else "-n")
        if self.threads >= 0:
            command.extend(["-threads", str(self.threads)])
        command.extend(str(value) for value in arguments)
        return command

    def run(self, arguments: Sequence[str], *, timeout: float = 600) -> ProcessResult:
        return self.runner.run(self.command(*arguments), timeout=timeout)

    def transcode_audio(self, source: Path, target: Path, *, codec_args: Sequence[str] = ()) -> ProcessResult:
        return self.run(["-i", str(source), "-vn", *codec_args, str(target)])
