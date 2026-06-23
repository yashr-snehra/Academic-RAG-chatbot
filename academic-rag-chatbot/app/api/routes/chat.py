"""
Chat Router — Phase 4

Endpoints:
  POST   /api/v1/chat                       — Ask a question
  GET    /api/v1/chat/{session_id}/history  — Get conversation history
  DELETE /api/v1/chat/{session_id}          — Clear a session
"""

import asyncio
import json
import logging
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import StreamingResponse
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.api.deps import get_qdrant, get_redis
from app.api.rate_limit import rate_limit
from app.core.generation.chain import build_rag_chain, dynamic_num_predict
from app.core.generation.citations import extract_citations, retrieval_confidence
from app.core.memory.cache import cache_response, get_cached_response
from app.core.memory.history import clear_session, get_session_history, session_exists
from app.core.retrieval.retriever import get_retriever
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

logger = logging.getLogger("app.chat")


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question about uploaded documents",
    dependencies=[Depends(rate_limit(30, 60))],  # 30 questions / minute / IP
)
async def chat(
    request: ChatRequest,
    qdrant=Depends(get_qdrant),
    redis=Depends(get_redis),
):
    """
    Main chat endpoint. Full request lifecycle:

      1. Check Redis cache → return immediately on hit (< 10ms)
      2. Build retriever (with optional document_ids filter)
      3. Build RAG chain (history-aware retriever + stuff documents chain)
      4. Invoke chain with question + session history from Redis
      5. Extract source citations from retrieved documents
      6. Cache the response and return

    The session_id is used to load/save conversation history in Redis,
    enabling natural multi-turn conversations.
    """
    start = time.monotonic()

    # The shared response cache is keyed on question text + doc_ids only, so it is
    # ONLY safe for standalone (first-turn) questions. A follow-up like "what else?"
    # is contextualized against this session's history; caching it by raw text would
    # serve one session's answer to another. So we cache strictly on the first turn.
    # First-turn check is a cheap async EXISTS on the session key — loading the full
    # message list just to test emptiness is O(history) on every turn. The sync
    # RedisChatMessageHistory .add_* calls below still hit Redis on the calling
    # thread, so those stay offloaded to a worker thread under concurrency.
    history = get_session_history(request.session_id)
    is_first_turn = not await session_exists(redis, request.session_id)

    # ── 1. Cache check (first turn only) ──────────────────────────────────────
    if is_first_turn:
        cached = await get_cached_response(redis, request.question, request.document_ids)
        if cached:
            # Record the turn so the conversation stays coherent even on a cache hit
            # (otherwise the next question's history would be missing this exchange).
            await asyncio.to_thread(history.add_user_message, request.question)
            await asyncio.to_thread(history.add_ai_message, cached["answer"])
            cached["cached"] = True
            cached["latency_ms"] = int((time.monotonic() - start) * 1000)
            return ChatResponse(**cached)

    # ── 2 & 3. Build retriever + chain ────────────────────────────────────────
    # Size the generation budget to the question so simple asks finish faster.
    retriever = get_retriever(qdrant, request.document_ids)
    rag_chain = build_rag_chain(retriever, num_predict=dynamic_num_predict(request.question))

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # ── 4. Invoke chain ───────────────────────────────────────────────────────
    # ainvoke() keeps the event loop free while the LLM/Qdrant calls are in flight.
    # Plain .invoke() here would block every other request for seconds.
    try:
        response = await chain_with_history.ainvoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}},
        )
    except (httpx.HTTPError, ConnectionError, OSError) as e:
        # Ollama/Qdrant unreachable -> clear 503 instead of a generic 500.
        raise HTTPException(status_code=503, detail=f"LLM backend unavailable: {e}") from e

    # ── 5. Extract citations ──────────────────────────────────────────────────
    sources = extract_citations(response.get("context", []))
    latency_ms = int((time.monotonic() - start) * 1000)

    result_dict = {
        "answer": response["answer"],
        "sources": [s.model_dump() for s in sources],
        "session_id": request.session_id,
        "latency_ms": latency_ms,
        "retrieval_confidence": retrieval_confidence(sources),
    }

    # ── 6. Cache (first turn only) and return ─────────────────────────────────
    # Only cache grounded answers: an empty/ "I don't know" result with no sources
    # would otherwise poison the cache for the full TTL.
    if is_first_turn and sources:
        await cache_response(redis, request.question, request.document_ids, result_dict)

    logger.info("chat latency=%dms sources=%d q=%r", latency_ms, len(sources), request.question[:80])
    return ChatResponse(**result_dict, cached=False)


def _sse(obj: dict) -> str:
    """Format a dict as one Server-Sent Events frame."""
    return f"data: {json.dumps(obj)}\n\n"


@router.post(
    "/stream",
    summary="Ask a question (answer streamed token-by-token via SSE)",
    dependencies=[Depends(rate_limit(30, 60))],  # 30 questions / minute / IP
)
async def chat_stream(
    request: ChatRequest,
    qdrant=Depends(get_qdrant),
    redis=Depends(get_redis),
):
    """
    Same contract as POST /chat, but streams the answer as it is generated
    (media type text/event-stream). Frames:
      {"type":"token","content":"..."}        — one per generated token
      {"type":"done","sources":[...],"cached":bool,"latency_ms":int,"session_id":"..."}
      {"type":"error","message":"..."}        — on failure

    The non-streaming /chat endpoint remains for clients that want a single JSON body.
    """
    start = time.monotonic()

    async def event_gen():
        try:
            # Cache is first-turn-only — see the /chat endpoint for why follow-ups
            # must not be served from the question-keyed cache.
            history = get_session_history(request.session_id)
            is_first_turn = not await session_exists(redis, request.session_id)

            # Cache hit → send the whole answer in one frame, then done.
            if is_first_turn:
                cached = await get_cached_response(redis, request.question, request.document_ids)
                if cached:
                    await asyncio.to_thread(history.add_user_message, request.question)
                    await asyncio.to_thread(history.add_ai_message, cached["answer"])
                    yield _sse({"type": "token", "content": cached["answer"]})
                    yield _sse({
                        "type": "done", "sources": cached["sources"], "cached": True,
                        "latency_ms": int((time.monotonic() - start) * 1000),
                        "session_id": request.session_id,
                        "retrieval_confidence": cached.get("retrieval_confidence"),
                    })
                    return

            retriever = get_retriever(qdrant, request.document_ids)
            rag_chain = build_rag_chain(
                retriever, num_predict=dynamic_num_predict(request.question)
            )
            chain_with_history = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input", history_messages_key="chat_history",
                output_messages_key="answer",
            )

            answer_parts: list[str] = []
            context_docs = []
            async for chunk in chain_with_history.astream(
                {"input": request.question},
                config={"configurable": {"session_id": request.session_id}},
            ):
                if chunk.get("context"):
                    context_docs = chunk["context"]
                token = chunk.get("answer")
                if token:
                    answer_parts.append(token)
                    yield _sse({"type": "token", "content": token})

            answer = "".join(answer_parts)
            citations = extract_citations(context_docs)
            sources = [s.model_dump() for s in citations]
            confidence = retrieval_confidence(citations)
            latency_ms = int((time.monotonic() - start) * 1000)
            # Only cache grounded answers (see /chat) — skip when no sources.
            if is_first_turn and sources:
                await cache_response(
                    redis, request.question, request.document_ids,
                    {"answer": answer, "sources": sources,
                     "session_id": request.session_id, "latency_ms": latency_ms,
                     "retrieval_confidence": confidence},
                )
            yield _sse({
                "type": "done", "sources": sources, "cached": False,
                "latency_ms": latency_ms, "session_id": request.session_id,
                "retrieval_confidence": confidence,
            })
        except Exception as e:  # surface failures to the UI instead of a dead stream
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/{session_id}/history",
    summary="Get conversation history for a session",
)
async def get_history(
    session_id: str = Path(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
):
    """
    Returns all messages stored for the given session_id.
    Useful for rebuilding conversation context in a frontend.
    """
    history = get_session_history(session_id)
    messages = await asyncio.to_thread(lambda: history.messages)

    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": [
            {"role": m.type, "content": m.content}
            for m in messages
        ],
    }


@router.delete(
    "/{session_id}",
    summary="Clear a conversation session",
)
async def delete_session(
    session_id: str = Path(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
):
    """
    Delete all message history for a session.
    After calling this, subsequent questions from the same session_id
    start without any conversation context.
    """
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
