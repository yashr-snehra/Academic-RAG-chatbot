"""
Chat Session History — Phase 3 & 5

Uses Redis to persist conversation history across API requests.
Each session_id maps to its own Redis key with a configurable TTL.

How it integrates with LangChain:
  RunnableWithMessageHistory wraps the RAG chain and automatically:
    - Loads previous messages from Redis before each invocation
    - Appends the new human message and AI response to Redis after

Why Redis instead of in-memory?
  - Survives server restarts
  - Works across multiple API workers (uvicorn --workers N)
  - TTL handles automatic cleanup of old sessions
"""

# The langchain_community implementation is the one pinned here
# (langchain-community 0.3.31) and is NOT deprecated in that line. Upstream is
# migrating this class to the standalone `langchain-redis` package, but that move
# also changes the constructor (url -> redis_url, different key handling), so it's
# a deliberate dependency + code change, not a drop-in import swap. We stay on the
# community class until we choose to migrate; revisit when bumping langchain.
from langchain_community.chat_message_histories import RedisChatMessageHistory
import redis.asyncio as aioredis

from app.config import settings

# Key format used by RedisChatMessageHistory is "{key_prefix}{session_id}".
# Kept as a constant so the first-turn EXISTS check can't drift from it.
_KEY_PREFIX = "academic_chat:"


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """
    Return the Redis-backed message history for a conversation session.

    Called by RunnableWithMessageHistory on every request with the session_id
    from the API request body.

    Args:
        session_id: UUID string identifying the conversation.

    Returns:
        RedisChatMessageHistory instance linked to this session.
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.redis_url,
        ttl=settings.session_ttl_seconds,
        key_prefix=_KEY_PREFIX,  # Redis key format: "academic_chat:{session_id}"
    )


async def session_exists(redis_async: "aioredis.Redis", session_id: str) -> bool:
    """Return True if this session already has stored messages.

    A cheap O(1) ``EXISTS`` on the session's Redis key, used for the first-turn
    check so callers don't load the entire message list just to test emptiness.
    """
    return bool(await redis_async.exists(f"{_KEY_PREFIX}{session_id}"))


def clear_session(session_id: str) -> None:
    """
    Delete all messages for a session (start fresh conversation).

    Args:
        session_id: UUID string of the session to clear.
    """
    history = get_session_history(session_id)
    history.clear()
