"""
Persistent document metadata store (Redis).

Listing documents by scrolling every point in Qdrant is O(collection size) and
gets slow as the corpus grows. Instead we keep a tiny per-document record in
Redis, written once at ingest time and read directly when listing.

Redis layout:
    doc_meta:{name}  -> hash { name, total_chunks, pages, ingested_at }

Write path is sync (called from the background ingest task, which runs in a
threadpool); read path is async (called from the FastAPI route). They use
different clients on purpose.
"""

from datetime import datetime, timezone

import redis  # sync client — the async one lives on app.state for routes

from app.config import settings

_PREFIX = "doc_meta:"

# One shared sync client, created on first use. redis-py keeps an internal
# connection pool, so reusing this avoids leaking a fresh client (and its socket)
# on every ingest the way a per-call redis.from_url() did.
_sync_client: "redis.Redis | None" = None


def _get_sync_client() -> "redis.Redis":
    global _sync_client
    if _sync_client is None:
        _sync_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _sync_client


def record_document(
    name: str,
    total_chunks: int,
    pages: int,
    client: "redis.Redis | None" = None,
) -> None:
    """Record (or overwrite) one document's metadata. Sync — safe to call from a
    background ingest task. Pass `client` in tests; otherwise the shared module
    client (a pooled redis.from_url) is reused.
    """
    client = client or _get_sync_client()
    client.hset(
        f"{_PREFIX}{name}",
        mapping={
            "name": name,
            "total_chunks": int(total_chunks),
            "pages": int(pages),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        },
    )


async def get_documents(redis_async) -> list[dict]:
    """Return all recorded documents (async read). Empty list if none recorded —
    callers can fall back to a Qdrant scan for documents ingested before this
    store existed."""
    keys = await redis_async.keys(f"{_PREFIX}*")
    docs: list[dict] = []
    for key in keys:
        h = await redis_async.hgetall(key)
        if not h:
            continue
        docs.append(
            {
                "name": h.get("name", key.removeprefix(_PREFIX)),
                "total_chunks": int(h.get("total_chunks", 0)),
                "pages": int(h["pages"]) if h.get("pages") else None,
                "ingested_at": h.get("ingested_at"),
            }
        )
    docs.sort(key=lambda d: d["name"])
    return docs
