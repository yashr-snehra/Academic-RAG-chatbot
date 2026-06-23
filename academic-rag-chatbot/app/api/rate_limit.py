"""
Lightweight Redis-backed rate limiting.

A fixed-window counter keyed on client IP + route. It reuses the shared async
Redis client already on app.state, so it adds no new dependency and works across
uvicorn workers (the counter lives in Redis, not per-process memory).

Auth is a separate, larger concern (these endpoints are currently unauthenticated);
this caps the abuse/DoS surface on the expensive endpoints in the meantime.

ponytail: fixed-window counter — a client can burst up to ~2x the limit right at a
window boundary. Swap for a sliding-window/token-bucket if that precision matters.
"""

from fastapi import Depends, HTTPException, Request

from app.api.deps import get_redis


def rate_limit(max_requests: int, window_seconds: int):
    """Build a dependency that returns 429 once a client (by IP) exceeds
    ``max_requests`` within ``window_seconds`` on the calling route.

    If Redis is unreachable the request is allowed through — rate limiting is a
    guardrail, not a correctness gate, and must not take the API down with it.
    """

    async def _dependency(request: Request, redis=Depends(get_redis)) -> None:
        client_ip = request.client.host if request.client else "unknown"
        # Use the route template (e.g. /chat/{session_id}) so per-param URLs share
        # one bucket instead of each getting its own.
        route = request.scope.get("route")
        scope_id = getattr(route, "path", request.url.path)
        key = f"ratelimit:{scope_id}:{client_ip}"

        try:
            count = await redis.incr(key)
            if count == 1:
                # First hit of a new window — start the expiry clock.
                await redis.expire(key, window_seconds)
        except Exception:  # noqa: BLE001 - never let a Redis blip 500 the endpoint
            return

        if count > max_requests:
            ttl = await redis.ttl(key)
            retry_after = max(int(ttl), 1)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )

    return _dependency
