"""
Response Cache — Phase 5

Caches RAG chain responses in Redis so identical (or near-identical) questions
are served in milliseconds instead of seconds.

Cache key design:
  MD5(normalized_question + sorted_doc_ids)

  Normalization: lowercase + strip whitespace
  Sorting doc_ids: ensures "paper_a, paper_b" and "paper_b, paper_a" hit the same key

  MD5 is used here for speed, not cryptographic security — it's a cache key, not a password.

Cache hit flow:
  Request → check Redis → found → return cached JSON (< 10ms)

Cache miss flow:
  Request → check Redis → not found → invoke RAG chain → store in Redis → return
"""

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis

from app.config import settings


def _make_cache_key(question: str, doc_ids: list[str] | None) -> str:
    """
    Build a deterministic, normalized cache key for a question + doc_ids pair.

    >>> _make_cache_key("What is BERT?", ["paper_a", "paper_b"])
    'rag_cache:abc123...'
    """
    normalized = question.lower().strip()
    ids_str = ",".join(sorted(doc_ids)) if doc_ids else "all_documents"
    raw = f"{normalized}:{ids_str}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"rag_cache:{digest}"


async def get_cached_response(
    redis: aioredis.Redis,
    question: str,
    doc_ids: list[str] | None,
) -> dict[str, Any] | None:
    """
    Look up a cached response in Redis.

    Returns:
        Parsed response dict on cache hit, None on cache miss.
    """
    key = _make_cache_key(question, doc_ids)
    cached = await redis.get(key)
    if cached is None:
        return None
    return json.loads(cached)


async def cache_response(
    redis: aioredis.Redis,
    question: str,
    doc_ids: list[str] | None,
    response: dict[str, Any],
) -> None:
    """
    Store a response in Redis with a TTL.

    Args:
        redis: Async Redis client.
        question: The original question.
        doc_ids: Document filter used (or None for all docs).
        response: The response dict to cache (must be JSON-serializable).
    """
    key = _make_cache_key(question, doc_ids)
    await redis.setex(
        name=key,
        time=settings.cache_ttl_seconds,
        value=json.dumps(response),
    )


async def invalidate_all_cache(redis: aioredis.Redis) -> int:
    """
    Delete all RAG response cache entries.

    Call this after ingesting new documents so cached answers don't
    reflect stale knowledge. Returns the number of keys deleted.
    """
    keys = await redis.keys("rag_cache:*")
    if not keys:
        return 0
    return await redis.delete(*keys)
