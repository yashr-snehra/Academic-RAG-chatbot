"""
Cross-encoder reranking.

Dense retrieval is fast but approximate: it scores query/chunk similarity from
two independently-computed embeddings. A cross-encoder reads the query and chunk
*together*, so it judges relevance far more accurately — but it's too slow to run
over the whole collection. The standard pattern: dense-retrieve a wide candidate
set (top_k * multiplier), then rerank just those down to top_k.

Enabled via settings.rerank_enabled. The model (sentence-transformers CrossEncoder)
is imported and loaded lazily so the app runs without the dependency when reranking
is off.
"""

from functools import lru_cache
from typing import Callable, Sequence

from langchain_core.documents import Document

from app.config import settings


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load and cache the cross-encoder. Lazy import: sentence-transformers is only
    required when reranking is actually enabled."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model)


def _default_scorer(query: str, docs: Sequence[Document]) -> list[float]:
    model = _get_cross_encoder()
    pairs = [(query, d.page_content) for d in docs]
    return [float(s) for s in model.predict(pairs)]


def rerank(
    query: str,
    docs: Sequence[Document],
    top_k: int,
    scorer: Callable[[str, Sequence[Document]], list[float]] | None = None,
) -> list[Document]:
    """Reorder `docs` by cross-encoder relevance to `query`; return the top_k.

    Each returned Document gets metadata["rerank_score"]. `scorer` is injectable
    so the ordering logic is testable without loading a model.
    """
    if not docs:
        return []
    scorer = scorer or _default_scorer
    scores = scorer(query, docs)

    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    out: list[Document] = []
    for doc, score in ranked[:top_k]:
        doc.metadata["rerank_score"] = round(float(score), 4)
        out.append(doc)
    return out
