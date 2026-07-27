"""Translation session persistence abstraction."""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class TranslationSession:
    payload: dict[str, Any]
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            **copy.deepcopy(self.payload),
        }


class SessionRepository(Protocol):
    async def put(self, session: TranslationSession) -> None: ...

    async def get(self, session_id: str) -> TranslationSession | None: ...

    async def delete(self, session_id: str) -> bool: ...

    async def cleanup(self, *, now: float | None = None) -> int: ...


class InMemorySessionRepository:
    def __init__(self, *, ttl_seconds: float = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, TranslationSession] = {}
        self._lock = asyncio.Lock()

    async def put(self, session: TranslationSession) -> None:
        session.updated_at = time.time()
        async with self._lock:
            self._items[session.session_id] = session

    async def get(self, session_id: str) -> TranslationSession | None:
        async with self._lock:
            session = self._items.get(session_id)
            return copy.deepcopy(session) if session is not None else None

    async def patch(self, session_id: str, values: Mapping[str, Any]) -> TranslationSession | None:
        async with self._lock:
            session = self._items.get(session_id)
            if session is None:
                return None
            session.payload.update(copy.deepcopy(dict(values)))
            session.updated_at = time.time()
            return copy.deepcopy(session)

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._items.pop(session_id, None) is not None

    async def cleanup(self, *, now: float | None = None) -> int:
        cutoff = (time.time() if now is None else now) - self.ttl_seconds
        async with self._lock:
            expired = [key for key, item in self._items.items() if item.updated_at < cutoff]
            for key in expired:
                del self._items[key]
            return len(expired)
