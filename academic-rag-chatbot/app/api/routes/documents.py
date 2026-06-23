"""
Documents Router — Phase 1 & 4

Endpoints:
  POST /api/v1/documents/upload  — Upload a PDF for async ingestion
  GET  /api/v1/documents         — List all ingested documents
"""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile

from app.api.deps import get_qdrant, get_qdrant_async, get_redis
from app.api.rate_limit import rate_limit
from app.config import settings
from app.core.ingestion.chunker import chunk_documents
from app.core.ingestion.embedder import ensure_collection, store_chunks
from app.core.ingestion.loader import SUPPORTED_SUFFIXES, load_document
from app.core.memory.cache import invalidate_all_cache
from app.core.memory.doc_store import get_documents, record_document
from app.models.schemas import DocumentInfo, DocumentListResponse, UploadResponse

logger = logging.getLogger("app.ingest")

router = APIRouter(prefix="/documents", tags=["documents"])


# ── Background task ───────────────────────────────────────────────────────────

def _ingest_pdf(tmp_path: str, original_name: str, qdrant_client) -> None:
    """
    Background task: load → chunk → embed → store → record metadata.

    Runs asynchronously after the upload endpoint returns 200.
    For a 50-page paper, this typically takes 30-60 seconds.

    `original_name` is the uploaded filename stem. The loader derives source_file
    from the path, which here is a random temp file — so we override it with the
    real name, otherwise documents would be stored (and cited) under a temp name.
    """
    try:
        docs = load_document(tmp_path)
        for doc in docs:
            doc.metadata["source_file"] = original_name
        chunks = chunk_documents(docs)
        ensure_collection(qdrant_client)
        count = store_chunks(chunks, qdrant_client)
        record_document(name=original_name, total_chunks=count, pages=len(docs))
        logger.info("Ingest completed: %s -> %d pages, %d chunks", original_name, len(docs), count)
    except Exception as e:
        logger.error("Ingest FAILED: %s -> %s", original_name, e)
    finally:
        # Always remove the temp upload — otherwise it leaks into the OS temp dir
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload an academic PDF for ingestion",
    dependencies=[Depends(rate_limit(10, 60))],  # 10 uploads / minute / IP
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    qdrant=Depends(get_qdrant),
    redis=Depends(get_redis),
):
    """
    Upload a PDF. The file is accepted immediately and processed in the background.
    Once ingestion completes, the document becomes searchable via the /chat endpoint.

    Notes:
      - Only PDF files are accepted
      - Maximum file size: 50MB
      - After ingestion, all cached responses are invalidated
    """
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Accepted: {sorted(SUPPORTED_SUFFIXES)}",
        )

    max_bytes = 50 * 1024 * 1024
    # file.size is the client-declared size — a fast early reject, but it can be
    # None (no Content-Length) or a lie, so we also enforce the cap during the read.
    if file.size and file.size > max_bytes:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")

    # Write to temp file — background task needs a persistent path, not an async stream.
    # mkstemp() creates the file atomically (mktemp() is deprecated and race-prone).
    # Read in 1 MB chunks via the async UploadFile API so a large upload never blocks
    # the event loop (shutil.copyfileobj would read the whole stream synchronously).
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    written = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413, detail="File too large. Maximum size is 50MB."
                    )
                f.write(chunk)
    except BaseException:
        # Don't leave a partial temp file behind on reject/disconnect.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    # Queue ingestion and invalidate cache in background
    original_name = Path(file.filename).stem
    background_tasks.add_task(_ingest_pdf, tmp_path, original_name, qdrant)
    background_tasks.add_task(invalidate_all_cache, redis)

    return UploadResponse(
        status="queued",
        filename=file.filename,
        message="PDF queued for processing. Check /documents to see when it appears.",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_documents(
    qdrant=Depends(get_qdrant_async),
    redis=Depends(get_redis),
):
    """
    Returns all ingested documents with chunk counts.

    Reads the persistent metadata store (Redis) first — O(#documents). Falls back
    to scanning the Qdrant collection only when the store is empty (e.g. documents
    ingested by the batch script before the store existed).
    """
    recorded = await get_documents(redis)
    if recorded:
        documents = [
            DocumentInfo(
                name=d["name"],
                total_chunks=d["total_chunks"],
                pages=d.get("pages"),
                ingested_at=d.get("ingested_at"),
            )
            for d in recorded
        ]
        return DocumentListResponse(documents=documents, total=len(documents))

    try:
        # Page through the whole collection on the scroll cursor — a single
        # fixed `limit` would silently truncate once the corpus exceeds it.
        doc_counts: dict[str, int] = {}
        offset = None
        while True:
            points, offset = await qdrant.scroll(
                collection_name=settings.qdrant_collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                name = point.payload.get("source_file", "unknown")
                doc_counts[name] = doc_counts.get(name, 0) + 1
            if offset is None:
                break

        documents = [
            DocumentInfo(name=name, total_chunks=count)
            for name, count in sorted(doc_counts.items())
        ]

        return DocumentListResponse(documents=documents, total=len(documents))

    except Exception:
        return DocumentListResponse(documents=[], total=0)
