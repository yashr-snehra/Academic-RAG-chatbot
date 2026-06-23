"""
Citation Extractor — Phase 3

Transforms the list of retrieved LangChain Documents into clean SourceCitation
objects for the API response.

Deduplication logic:
  When multiple overlapping chunks from the same page are retrieved (which happens
  with chunk_overlap > 0), we only want one citation entry per page.
  Deduplication key: (source_file, page_number).
"""

from app.models.schemas import SourceCitation


def extract_citations(source_docs: list) -> list[SourceCitation]:
    """
    Extract unique page-level citations from a list of retrieved Documents.

    Args:
        source_docs: List of LangChain Document objects from the RAG chain's "context" key.

    Returns:
        List of SourceCitation objects, deduplicated by (source_file, page_number)
        and sorted by source_file then page_number.
    """
    citations: list[SourceCitation] = []
    seen: set[tuple[str, int]] = set()

    for doc in source_docs:
        source_file = doc.metadata.get("source_file", "unknown")
        page_number = doc.metadata.get("page_number", 0)
        key = (source_file, page_number)

        if key in seen:
            continue  # Already have a citation for this page
        seen.add(key)

        citations.append(
            SourceCitation(
                source_file=source_file,
                page_number=page_number,
                chunk_index=doc.metadata.get("chunk_index", 0),
                snippet=doc.page_content[:200].strip(),
                score=doc.metadata.get("score"),  # None if retriever didn't supply one
            )
        )

    # Sort for deterministic ordering in API responses
    citations.sort(key=lambda c: (c.source_file, c.page_number))
    return citations


def retrieval_confidence(citations: list[SourceCitation]) -> float | None:
    """Overall confidence = best similarity score across cited sources (None if unscored)."""
    scores = [c.score for c in citations if c.score is not None]
    return max(scores) if scores else None
