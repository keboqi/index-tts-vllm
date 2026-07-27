import unittest

from indextts_web.services.translation.sessions import InMemorySessionRepository, TranslationSession


class SessionRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_trip_is_isolated_from_callers(self):
        repository = InMemorySessionRepository()
        session = TranslationSession({"segments": [{"text": "hello"}]})
        await repository.put(session)
        loaded = await repository.get(session.session_id)
        loaded.payload["segments"][0]["text"] = "changed"
        loaded_again = await repository.get(session.session_id)
        self.assertEqual(loaded_again.payload["segments"][0]["text"], "hello")

    async def test_cleanup_uses_ttl(self):
        repository = InMemorySessionRepository(ttl_seconds=10)
        session = TranslationSession({"value": 1}, created_at=1, updated_at=1)
        await repository.put(session)
        session_in_store = repository._items[session.session_id]
        session_in_store.updated_at = 1
        self.assertEqual(await repository.cleanup(now=20), 1)
        self.assertIsNone(await repository.get(session.session_id))


if __name__ == "__main__":
    unittest.main()

