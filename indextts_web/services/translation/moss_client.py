"""Model Manager client for the dedicated MOSS service used by Modal."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any


class MossModelClient:
    def __init__(self, *, url: str, enabled: bool, api_key: str = "") -> None:
        self.url = url.rstrip("/")
        self.enabled = enabled
        self.api_key = api_key

    @classmethod
    def from_env(cls):
        return cls(
            url=os.getenv("MOSS_TRANSCRIBE_SGLANG_URL", "http://127.0.0.1:8003"),
            enabled=os.getenv("MOSS_TRANSCRIBE_BACKEND", "auto").strip().lower() in {"http", "server", "sglang"},
            api_key=os.getenv("MOSS_TRANSCRIBE_API_KEY", ""),
        )

    def _request(self, action: str | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        path = f"/model/{action}" if action else "/model/status"
        request = urllib.request.Request(
            self.url + path, headers=headers, method="POST" if action else "GET",
        )
        # An action can wait for an active transcription or a checkpoint reload.
        with urllib.request.urlopen(request, timeout=1800 if action else 2) as response:
            result = json.load(response)
        if not isinstance(result, dict) or result.get("service") != "moss-transcribe":
            raise RuntimeError("The MOSS backend does not expose model lifecycle controls")
        return result

    async def inventory(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            status = await asyncio.to_thread(self._request)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # External OpenAI-compatible servers need not implement our API.
                return []
            status = {"state": "unavailable", "error": str(exc)}
        except (OSError, ValueError, RuntimeError) as exc:
            status = {"state": "unavailable", "error": str(exc)}
        state = status.get("state", "unavailable")
        actions = {
            "loaded": ["sleep", "unload"], "sleeping": ["wake", "unload"], "unloaded": ["wake"],
        }.get(state, [])
        return [{
            "key": "moss_transcribe", "name": "MOSS Transcribe + Diarize", "kind": "Transcription",
            "state": state, "busy": bool(status.get("busy")),
            "actions": [action for action in actions if action in status.get("capabilities", [])],
            "error": status.get("error", ""),
        }]

    async def change(self, action: str) -> dict[str, Any]:
        if not self.enabled or action not in {"sleep", "wake", "unload"}:
            raise ValueError("MOSS lifecycle control is not configured for this action")
        return await asyncio.to_thread(self._request, action)
