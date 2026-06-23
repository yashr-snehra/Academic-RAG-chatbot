"""
FastAPI dependency injectors.

These functions are passed to route handlers via Depends().
They pull the shared clients from app.state (initialized in lifespan).

Usage in a route:
    @router.post("/chat")
    async def chat(qdrant=Depends(get_qdrant), redis=Depends(get_redis)):
        ...
"""

from fastapi import Request
from qdrant_client import AsyncQdrantClient, QdrantClient
import redis.asyncio as aioredis


def get_qdrant(request: Request) -> QdrantClient:
    """Return the shared (sync) QdrantClient from app state.

    Used for retrieval (LangChain needs the sync client; it offloads the call to
    a threadpool under ainvoke) and for background ingestion tasks.
    """
    return request.app.state.qdrant


def get_qdrant_async(request: Request) -> AsyncQdrantClient:
    """Return the shared AsyncQdrantClient — use this from async routes that call
    Qdrant directly (readiness, document listing) so the event loop never blocks."""
    return request.app.state.qdrant_async


def get_redis(request: Request) -> aioredis.Redis:
    """Return the shared async Redis client from app state."""
    return request.app.state.redis
