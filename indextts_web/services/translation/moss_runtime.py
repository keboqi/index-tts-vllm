"""Thread-safe lifecycle for the dedicated MOSS Transformers service."""

from __future__ import annotations

import threading
from typing import Any


class MossRuntime:
    """Serialize inference and memory changes; status never waits for inference."""

    def __init__(self, *, loader, release_cache) -> None:
        self._loader = loader
        self._release_cache = release_cache
        self._lock = threading.Lock()
        self._runtime = None
        self._state = "unloaded"
        self._busy = False

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "busy": self._busy,
            "capabilities": ["sleep", "wake", "unload"],
        }

    def _ensure_awake(self) -> None:
        try:
            if self._runtime is None:
                self._runtime = self._loader()
            elif self._state == "sleeping":
                self._runtime[0].to(self._runtime[2])
            self._state = "loaded"
        except Exception:
            self._runtime = None
            self._state = "unloaded"
            self._release_cache()
            raise

    def generate(self, inference, *args, **kwargs):
        with self._lock:
            self._busy = True
            try:
                self._ensure_awake()
                # No model references escape the lock to race an unload.
                return inference(*self._runtime, *args, **kwargs)
            finally:
                self._busy = False

    def change(self, action: str) -> dict[str, Any]:
        if action not in {"sleep", "wake", "unload"}:
            raise ValueError(f"Unknown MOSS action: {action}")
        with self._lock:
            self._busy = True
            try:
                if action == "wake":
                    self._ensure_awake()
                elif action == "unload":
                    self._runtime = None
                    self._state = "unloaded"
                    self._release_cache()
                elif self._runtime is not None and self._state != "sleeping":
                    self._runtime[0].to("cpu")
                    self._state = "sleeping"
                    self._release_cache()
            except Exception:
                # A failed device transfer may leave weights split across devices.
                # Drop the runtime so a later request can reload it consistently.
                self._runtime = None
                self._state = "unloaded"
                self._release_cache()
                raise
            finally:
                self._busy = False
            return self.status()
