"""
Shared pytest fixtures.

Rule: Unit tests must NEVER call real network services.
All external dependencies (Qdrant, Redis, OpenAI) are mocked here.

How fixture sharing works in pytest:
  When multiple fixtures in the same test function use the same lower-level fixture
  (e.g., both 'test_client' and 'mock_redis' depend on mock_redis),
  pytest (with function scope) provides the SAME instance to both.
  This means setting mock_redis.get.return_value in a test affects the same
  object that the app is using inside test_client. This is intentional.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.chat_history import InMemoryChatMessageHistory

from app.main import app


# ── Chat history stub ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fake_chat_history(monkeypatch):
    """
    Replace the Redis-backed chat history with an in-memory one for every test.

    The history layer (RedisChatMessageHistory) opens its own connection straight
    to settings.redis_url — it does NOT go through the mocked app.state.redis — so
    without this stub the session endpoints would try to reach a real Redis.
    """
    store: dict[str, InMemoryChatMessageHistory] = {}

    def _factory(*args, **kwargs):
        session_id = kwargs.get("session_id") or (args[0] if args else "default")
        return store.setdefault(session_id, InMemoryChatMessageHistory())

    monkeypatch.setattr(
        "app.core.memory.history.RedisChatMessageHistory", _factory
    )


# ── Qdrant mock ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_qdrant():
    """Mock QdrantClient — no real Qdrant connection needed."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    client.create_collection.return_value = None
    client.create_payload_index.return_value = None
    client.scroll.return_value = ([], None)
    return client


# ── Redis mock ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """Mock async Redis client — no real Redis connection needed."""
    client = AsyncMock()
    client.get.return_value = None          # Default: cache miss
    client.setex.return_value = True
    client.delete.return_value = 0
    client.keys.return_value = []
    client.exists.return_value = 0          # Default: session absent → first turn
    # Rate limiter: first hit of the window, well under any limit (AsyncMock would
    # otherwise return a truthy MagicMock and spuriously trip the > limit check).
    client.incr.return_value = 1
    client.expire.return_value = True
    client.ttl.return_value = 60
    return client


@pytest.fixture
def mock_redis_with_cache(mock_redis):
    """Redis mock pre-loaded with a cached response for any key."""
    payload = {
        "answer": "Cached: The paper proposes a new attention mechanism.",
        "sources": [],
        "session_id": "test-session-cached",
        "latency_ms": 7,
    }
    mock_redis.get.return_value = json.dumps(payload)
    return mock_redis


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture
async def test_client(mock_qdrant, mock_redis):
    """
    Async HTTP test client for FastAPI integration tests.

    Uses httpx.AsyncClient with ASGITransport — no real network calls.
    Injects mocked Qdrant and Redis into app.state before the client is returned.
    """
    app.state.qdrant = mock_qdrant
    app.state.redis = mock_redis
    # Async Qdrant client used by async routes (e.g. list_documents). Needs awaitable
    # methods, so it's an AsyncMock — the sync mock_qdrant can't be awaited.
    qdrant_async = AsyncMock()
    qdrant_async.scroll.return_value = ([], None)
    qdrant_async.get_collections.return_value = MagicMock(collections=[])
    app.state.qdrant_async = qdrant_async

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
