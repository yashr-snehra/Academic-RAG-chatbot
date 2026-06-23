from langchain_core.documents import Document

from app.core.generation.citations import extract_citations, retrieval_confidence


def _doc(
    source: str, page: int, chunk: int = 0, content: str = "", score: float | None = None
) -> Document:
    meta = {"source_file": source, "page_number": page, "chunk_index": chunk}
    if score is not None:
        meta["score"] = score
    return Document(page_content=content or f"Content from {source} page {page}.", metadata=meta)


def test_returns_empty_for_no_docs():
    assert extract_citations([]) == []


def test_single_doc_produces_one_citation():
    result = extract_citations([_doc("paper_a", page=3)])
    assert len(result) == 1
    assert result[0].source_file == "paper_a"
    assert result[0].page_number == 3


def test_deduplicates_same_page_different_chunks():
    """Two chunks from the same page should produce only one citation."""
    docs = [
        _doc("paper_a", page=3, chunk=0),
        _doc("paper_a", page=3, chunk=1),  # Same page — should be deduplicated
    ]
    result = extract_citations(docs)
    assert len(result) == 1


def test_different_pages_each_get_citation():
    docs = [
        _doc("paper_a", page=1),
        _doc("paper_a", page=7),
        _doc("paper_a", page=12),
    ]
    result = extract_citations(docs)
    assert len(result) == 3
    assert [c.page_number for c in result] == [1, 7, 12]


def test_sorted_by_source_then_page():
    docs = [
        _doc("paper_c", page=2),
        _doc("paper_a", page=5),
        _doc("paper_a", page=1),
        _doc("paper_b", page=3),
    ]
    result = extract_citations(docs)
    assert [(c.source_file, c.page_number) for c in result] == [
        ("paper_a", 1),
        ("paper_a", 5),
        ("paper_b", 3),
        ("paper_c", 2),
    ]


def test_snippet_truncated_at_200_chars():
    long_content = "X" * 500
    doc = Document(
        page_content=long_content,
        metadata={"source_file": "paper", "page_number": 1, "chunk_index": 0},
    )
    result = extract_citations([doc])
    assert len(result[0].snippet) == 200


def test_unknown_source_falls_back_gracefully():
    doc = Document(page_content="Some content.", metadata={})  # No metadata at all
    result = extract_citations([doc])
    assert len(result) == 1
    assert result[0].source_file == "unknown"
    assert result[0].page_number == 0


def test_score_carried_into_citation():
    result = extract_citations([_doc("paper_a", page=1, score=0.83)])
    assert result[0].score == 0.83


def test_score_is_none_when_retriever_omits_it():
    assert extract_citations([_doc("paper_a", page=1)])[0].score is None


def test_retrieval_confidence_is_max_score():
    citations = extract_citations([
        _doc("paper_a", page=1, score=0.6),
        _doc("paper_a", page=2, score=0.91),
        _doc("paper_b", page=1, score=0.7),
    ])
    assert retrieval_confidence(citations) == 0.91


def test_retrieval_confidence_none_when_unscored():
    assert retrieval_confidence(extract_citations([_doc("paper_a", page=1)])) is None
