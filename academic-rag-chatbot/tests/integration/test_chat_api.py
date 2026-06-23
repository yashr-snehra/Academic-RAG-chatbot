"""
Integration tests for the FastAPI routes.

These tests use httpx.AsyncClient with ASGITransport — no real network calls.
External services (Qdrant, Redis) are mocked via conftest.py fixtures.

Tests verify:
  - HTTP status codes and response schemas
  - Cache logic (hit vs miss)
  - Input validation (Pydantic rejections)
  - Correct behavior when Redis returns cached data
"""

import json

import pytest


# ── Health endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200(test_client):
    resp = await test_client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Chat endpoint — input validation ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_requires_question(test_client):
    """Missing 'question' field should return 422 Unprocessable Entity."""
    resp = await test_client.post("/api/v1/chat", json={"session_id": "test-001"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_rejects_empty_question(test_client):
    """Empty question string should fail Pydantic min_length=1 validation."""
    resp = await test_client.post("/api/v1/chat", json={
        "question": "",
        "session_id": "test-001",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_requires_session_id(test_client):
    """Missing 'session_id' should return 422."""
    resp = await test_client.post("/api/v1/chat", json={
        "question": "What is attention?",
    })
    assert resp.status_code == 422


# ── Chat endpoint — cache behavior ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_serves_cached_response(test_client, mock_redis):
    """When Redis has a cached entry, it must be returned with cached=True."""
    cached = {
        "answer": "This answer was cached.",
        "sources": [],
        "session_id": "test-002",
        "latency_ms": 5,
    }
    mock_redis.get.return_value = json.dumps(cached)

    resp = await test_client.post("/api/v1/chat", json={
        "question": "What is attention?",
        "session_id": "test-002",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is True
    assert body["answer"] == "This answer was cached."


# ── Documents endpoint ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_documents_returns_200(test_client):
    """GET /documents should return 200 even when collection is empty."""
    resp = await test_client.get("/api/v1/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert "documents" in body
    assert "total" in body


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_type(test_client):
    """Unsupported file types (e.g. .docx) should be rejected; PDF/txt/md are allowed."""
    resp = await test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("essay.docx", b"fake content", "application/vnd.openxmlformats")},
    )
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_root_serves_chat_ui(test_client):
    """GET / serves the chat web UI."""
    resp = await test_client.get("/")
    assert resp.status_code == 200
    assert "Academic RAG Chatbot" in resp.text


@pytest.mark.asyncio
async def test_health_liveness(test_client):
    resp = await test_client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness_reports_all_checks(test_client):
    """Readiness probe returns a status per dependency (redis/qdrant/ollama)."""
    resp = await test_client.get("/api/v1/health/ready")
    body = resp.json()
    assert set(body["checks"].keys()) == {"redis", "qdrant", "ollama"}
    assert resp.status_code in (200, 503)
    assert isinstance(body["ready"], bool)


@pytest.mark.asyncio
async def test_chat_stream_serves_cached(test_client, mock_redis_with_cache):
    """A cache hit on the streaming endpoint emits a token frame then a done frame."""
    resp = await test_client.post(
        "/api/v1/chat/stream",
        json={"question": "What is the contribution?", "session_id": "stream-cached"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text.replace(" ", "")
    assert '"type":"token"' in text
    assert '"type":"done"' in text


@pytest.mark.asyncio
async def test_upload_accepts_valid_pdf(test_client):
    """A valid PDF upload should be queued and return 200."""
    resp = await test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
    assert body["filename"] == "paper.pdf"


# ── Session management ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_session_returns_200(test_client):
    resp = await test_client.delete("/api/v1/chat/some-session-id")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"
