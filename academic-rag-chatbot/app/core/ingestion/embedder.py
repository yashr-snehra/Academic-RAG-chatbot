"""
Embedder — Phase 1

Handles:
  1. Creating the Qdrant collection with proper vector config and payload indexes
  2. Embedding chunks via the local Ollama model (nomic-embed-text)
  3. Batch-upserting embedded chunks into Qdrant

Why payload indexes?
  Without them, every filtered Qdrant search scans the full collection.
  With indexes on source_file and page_number, filters are O(log n) — fast even at scale.
"""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """
    Return a local Ollama embeddings model instance (cached singleton).

    The vector dimension is fixed by the model (nomic-embed-text → 768), so we do
    NOT pass a dimensions arg here — that was OpenAI-specific. settings.embedding_dimensions
    only drives the Qdrant collection size and must match the model's output.

    Cached because get_retriever() runs per request: rebuilding the client object
    each time is wasted work (the embedding call itself still hits Ollama fresh).
    """
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


@lru_cache(maxsize=1)
def _get_sparse_embedding():
    """Cached BM25 sparse embedder for hybrid mode. Loading it per request would
    re-init the FastEmbed model every time — slow. Lazy import: only needed when
    hybrid is enabled."""
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name=settings.sparse_model)


# Named vectors used only in hybrid mode (dense-only mode keeps Qdrant's default
# unnamed vector, so existing collections are untouched when hybrid is off).
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def get_vectorstore(client: QdrantClient) -> QdrantVectorStore:
    """
    Build the QdrantVectorStore the whole app shares (ingestion + retrieval), so
    dense-only and hybrid modes are configured in exactly one place.

    Hybrid mode adds a BM25 sparse embedding and fuses dense + sparse results.
    FastEmbedSparse is imported lazily — `fastembed` is only needed when
    settings.hybrid_enabled is True.
    """
    if settings.hybrid_enabled:
        from langchain_qdrant import RetrievalMode

        return QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=get_embeddings(),
            sparse_embedding=_get_sparse_embedding(),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )


def ensure_collection(client: QdrantClient) -> None:
    """
    Create the Qdrant collection and payload indexes if they don't exist yet.

    Safe to call multiple times — checks before creating (idempotent).
    Call this before any upsert operation.
    """
    existing_names = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection in existing_names:
        return  # Already set up, nothing to do

    if settings.hybrid_enabled:
        # Hybrid needs NAMED vectors: a dense vector + a sparse (BM25) vector.
        # This schema is incompatible with the dense-only one, hence the re-ingest.
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )
    else:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )

    # Payload indexes for fast filtered search (e.g., "search only in paper_X")
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="source_file",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="page_number",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    print(f"[qdrant] Created collection '{settings.qdrant_collection}' with payload indexes.")


def store_chunks(chunks: list[Document], client: QdrantClient) -> int:
    """
    Embed chunks and upsert them into Qdrant.

    LangChain's QdrantVectorStore.add_documents() handles batching internally
    (default 64 documents per call to the embeddings backend). Returns the number stored.

    Args:
        chunks: List of Document objects (from chunk_documents).
        client: Active QdrantClient instance.

    Returns:
        Number of chunks stored.
    """
    if not chunks:
        return 0

    vectorstore = get_vectorstore(client)  # dense-only or hybrid per settings
    vectorstore.add_documents(chunks)  # Batching and UUID generation handled internally
    return len(chunks)
