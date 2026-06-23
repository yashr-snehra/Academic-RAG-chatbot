"""
Pydantic models for all API request and response bodies.

These are the contracts between your API and its consumers.
FastAPI uses these for automatic validation, serialization, and OpenAPI docs.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The academic question to ask the chatbot",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
        description=(
            "Conversation identifier (a UUID is recommended). Letters, digits, "
            "hyphens and underscores only, up to 128 chars. Reuse the same ID for "
            "follow-up questions."
        ),
    )
    document_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "Restrict search to specific document names (without .pdf extension). "
            "Pass null or omit to search across all ingested documents."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What evaluation metrics were used in this paper?",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "document_ids": None,
            }
        }
    }


# ── Response models ───────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    source_file: str = Field(..., description="PDF filename without extension")
    page_number: int = Field(..., description="Page number in the source document")
    chunk_index: int = Field(..., description="Sequential index of this chunk in the document")
    snippet: str = Field(..., description="First 200 characters of the retrieved chunk (for preview)")
    score: Optional[float] = Field(
        default=None,
        description=(
            "Retrieval relevance score (higher = more relevant). Scale depends on "
            "retrieval mode: cosine similarity (~0-1) for dense search, or a fused "
            "RRF score (much smaller, not 0-1) when hybrid search is enabled. "
            "Null if unscored."
        ),
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Grounded answer with inline citations in [Source: X, Page Y] format")
    sources: list[SourceCitation] = Field(..., description="All source documents used to generate the answer")
    session_id: str
    cached: bool = Field(default=False, description="True if this response was served from Redis cache")
    latency_ms: int = Field(..., description="Total response time in milliseconds")
    retrieval_confidence: Optional[float] = Field(
        default=None,
        description=(
            "Best retrieval score among cited sources. In dense mode this is a "
            "cosine similarity (~0-1) where low values suggest a weak/ungrounded "
            "answer; in hybrid mode it is a fused RRF score on a different scale, "
            "so compare it only against other hybrid-mode results. Null if unscored."
        ),
    )


class UploadResponse(BaseModel):
    status: str
    filename: str
    message: str


class DocumentInfo(BaseModel):
    name: str = Field(..., description="Document name (source_file value)")
    total_chunks: int = Field(..., description="Number of chunks stored in Qdrant")
    pages: Optional[int] = Field(default=None, description="Page count (None if unknown)")
    ingested_at: Optional[str] = Field(
        default=None, description="ISO-8601 UTC timestamp of ingestion (None if unknown)"
    )


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int
