from langchain_core.documents import Document

from app.core.retrieval.reranker import rerank


def _docs(*texts: str) -> list[Document]:
    return [Document(page_content=t, metadata={"source_file": "p", "page_number": i})
            for i, t in enumerate(texts)]


def _scorer_by_length(query, docs):
    # Stand-in cross-encoder: "more relevant" = longer chunk. Deterministic, no model.
    return [float(len(d.page_content)) for d in docs]


def test_reorders_by_score_and_truncates_to_top_k():
    docs = _docs("a", "aaaa", "aa")  # lengths 1, 4, 2
    result = rerank("q", docs, top_k=2, scorer=_scorer_by_length)
    assert [d.page_content for d in result] == ["aaaa", "aa"]


def test_attaches_rerank_score():
    result = rerank("q", _docs("aaa"), top_k=1, scorer=_scorer_by_length)
    assert result[0].metadata["rerank_score"] == 3.0


def test_empty_input_returns_empty():
    assert rerank("q", [], top_k=5, scorer=_scorer_by_length) == []


def test_top_k_larger_than_candidates_returns_all():
    result = rerank("q", _docs("a", "bb"), top_k=10, scorer=_scorer_by_length)
    assert len(result) == 2
