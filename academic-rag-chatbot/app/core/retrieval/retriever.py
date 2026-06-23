"""
Retriever — Phase 2

Wraps the Qdrant collection in a LangChain VectorStoreRetriever.
This is what the RAG chain calls to fetch relevant academic context.

Key concepts:
  - search_type="similarity": cosine similarity between query embedding and stored embeddings
  - k: number of chunks to retrieve per query (tunable in .env as RETRIEVAL_TOP_K)
  - Filter: optional Qdrant payload filter to restrict search to specific documents

When to use document_ids filter:
  A user may upload multiple papers. If they ask "what does paper_A say about X?",
  pass document_ids=["paper_A"] to restrict retrieval to that paper only.
  Without a filter, search spans all ingested documents (good for cross-paper queries).
"""

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from app.config import settings
from app.core.ingestion.embedder import get_vectorstore
from app.core.retrieval.reranker import rerank


def get_retriever(
    client: QdrantClient,
    document_ids: list[str] | None = None,
) -> Runnable:
    """
    Build a Qdrant-backed retriever that also surfaces the similarity score.

    The standard ``as_retriever()`` drops the score, so retrieval confidence
    can't be shown to the user. Here we use ``similarity_search_with_score`` and
    stash the score in each Document's metadata (``score``), where it flows
    through the RAG chain to the citation extractor. Returns a Runnable so it
    still plugs into create_history_aware_retriever and supports ``.invoke(query)``.

    Search mode (dense vs hybrid dense+BM25) is decided by get_vectorstore().
    When reranking is enabled, a wider candidate set is fetched and a cross-encoder
    reorders it down to retrieval_top_k.

    Args:
        client: Active QdrantClient instance.
        document_ids: Optional source_file names to restrict search to.
                      If None, searches across all ingested documents.
    """
    qdrant_filter = None
    if document_ids:
        qdrant_filter = Filter(
            must=[FieldCondition(key="source_file", match=MatchAny(any=document_ids))]
        )

    vectorstore = get_vectorstore(client)

    # Over-fetch before reranking; otherwise fetch exactly what we return.
    fetch_k = (
        settings.retrieval_top_k * settings.rerank_fetch_multiplier
        if settings.rerank_enabled
        else settings.retrieval_top_k
    )

    def _retrieve(query: str) -> list[Document]:
        scored = vectorstore.similarity_search_with_score(
            query, k=fetch_k, filter=qdrant_filter
        )
        docs: list[Document] = []
        # score is a dense cosine similarity (~0-1) in dense mode, or a fused RRF
        # score (smaller, not 0-1) in hybrid mode — higher = more relevant either way.
        for doc, score in scored:
            doc.metadata["score"] = round(float(score), 4)
            docs.append(doc)

        if settings.rerank_enabled:
            docs = rerank(query, docs, settings.retrieval_top_k)
        return docs

    return RunnableLambda(_retrieve)
