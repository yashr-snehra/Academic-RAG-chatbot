import json
from unittest.mock import AsyncMock

import pytest

from app.core.memory.cache import (
    _make_cache_key,
    cache_response,
    get_cached_response,
    invalidate_all_cache,
)


# ── Cache key tests (synchronous — no mock needed) ────────────────────────────

def test_key_is_deterministic():
    k1 = _make_cache_key("What is BERT?", None)
    k2 = _make_cache_key("What is BERT?", None)
    assert k1 == k2


def test_key_normalizes_case():
    k1 = _make_cache_key("What is BERT?", None)
    k2 = _make_cache_key("WHAT IS BERT?", None)
    assert k1 == k2


def test_key_strips_whitespace():
    k1 = _make_cache_key("What is BERT?", None)
    k2 = _make_cache_key("  What is BERT?  ", None)
    assert k1 == k2


def test_key_doc_id_order_independent():
    k1 = _make_cache_key("What is BERT?", ["paper_a", "paper_b"])
    k2 = _make_cache_key("What is BERT?", ["paper_b", "paper_a"])
    assert k1 == k2, "Same doc_ids in different order must produce the same key"


def test_different_questions_different_keys():
    k1 = _make_cache_key("What is BERT?", None)
    k2 = _make_cache_key("What is GPT-4?", None)
    assert k1 != k2


def test_key_starts_with_prefix():
    k = _make_cache_key("test", None)
    assert k.startswith("rag_cache:")


# ── Async cache operation tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    redis = AsyncMock()
    redis.get.return_value = None
    result = await get_cached_response(redis, "What is BERT?", None)
    assert result is None


@pytest.mark.asyncio
async def test_cache_hit_returns_parsed_response():
    payload = {"answer": "BERT is a transformer model.", "sources": [], "latency_ms": 100}
    redis = AsyncMock()
    redis.get.return_value = json.dumps(payload)
    result = await get_cached_response(redis, "What is BERT?", None)
    assert result == payload


@pytest.mark.asyncio
async def test_cache_response_calls_setex():
    redis = AsyncMock()
    response = {"answer": "Test answer.", "sources": []}
    await cache_response(redis, "What is BERT?", None, response)
    redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_cache_response_stores_correct_json():
    redis = AsyncMock()
    response = {"answer": "Test answer.", "sources": []}
    await cache_response(redis, "What is BERT?", None, response)

    _, args, kwargs = redis.setex.mock_calls[0]
    stored_value = args[2] if len(args) > 2 else kwargs.get("value")
    assert json.loads(stored_value) == response


@pytest.mark.asyncio
async def test_invalidate_all_cache_deletes_keys():
    redis = AsyncMock()
    redis.keys.return_value = ["rag_cache:abc", "rag_cache:def"]
    redis.delete.return_value = 2
    count = await invalidate_all_cache(redis)
    assert count == 2
    redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_invalidate_empty_cache_returns_zero():
    redis = AsyncMock()
    redis.keys.return_value = []
    count = await invalidate_all_cache(redis)
    assert count == 0
    redis.delete.assert_not_called()
