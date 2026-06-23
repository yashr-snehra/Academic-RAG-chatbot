import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.config import settings
from app.api.routes import chat, documents, evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("app")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize shared resources on startup, clean them up on shutdown.

    Using lifespan() instead of deprecated on_startup/on_shutdown events.
    Resources stored on app.state are accessible in routes via Request.app.state.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    # Sync client for LangChain retrieval/ingestion; async client for direct calls
    # from async routes (readiness, document listing) so the event loop never blocks.
    app.state.qdrant = QdrantClient(url=settings.qdrant_url)
    app.state.qdrant_async = AsyncQdrantClient(url=settings.qdrant_url)
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    logger.info("Qdrant connected -> %s", settings.qdrant_url)
    logger.info("Redis connected  -> %s", settings.redis_url)
    logger.info("Collection       -> %s", settings.qdrant_collection)

    yield  # Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await app.state.redis.aclose()
    await app.state.qdrant_async.close()
    logger.info("Connections closed.")


app = FastAPI(
    title="Academic RAG Chatbot API",
    version="1.0.0",
    description=(
        "Enterprise-grade Retrieval-Augmented Generation (RAG) chatbot for academic literature. "
        "Upload PDFs, ask questions, receive grounded answers with page-level citations."
    ),
    lifespan=lifespan,
)

# Wildcard origins and credentials cannot be combined (the browser rejects it and
# the CORS spec forbids it), so credentials are enabled only for an explicit origin
# list. Configure real origins via CORS_ALLOW_ORIGINS in production.
_cors_origins = settings.cors_allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all route modules
app.include_router(chat.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(evaluation.router, prefix="/api/v1")

# NOTE: Prometheus /metrics (prometheus-fastapi-instrumentator) was removed —
# instrumentator 7.x/8.x is incompatible with Starlette >=0.52 (required by
# FastAPI 0.137): its per-request route-name lookup raises on included routers and
# 500s every /api/v1/* call. Re-add a Starlette-1.x-compatible metrics layer if
# observability is needed.


@app.get("/", include_in_schema=False)
async def chat_ui():
    """Serve the chat web UI. API docs remain available at /docs."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/v1/health", tags=["health"], summary="Health check (liveness)")
async def health():
    """Returns service status. Use this to verify the API process is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/v1/health/ready", tags=["health"], summary="Readiness check")
async def readiness():
    """
    Deep check that the dependencies the chatbot needs are actually reachable:
    Redis (cache + history), Qdrant (vectors), and Ollama (LLM + embeddings).
    Returns 503 if any are down, so a load balancer can hold traffic until ready.
    """
    checks: dict[str, str] = {}

    try:
        await app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:  # noqa: BLE001 - report any failure verbatim
        checks["redis"] = f"error: {e}"

    try:
        await app.state.qdrant_async.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["qdrant"] = f"error: {e}"

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
        checks["ollama"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["ollama"] = f"error: {e}"

    ready = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"ready": ready, "checks": checks},
    )
