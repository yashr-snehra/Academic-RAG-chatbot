from langchain_core.documents import Document

from app.core.ingestion.chunker import chunk_documents


def test_long_document_is_split():
    """A document longer than chunk_size must be split into multiple chunks."""
    docs = [Document(page_content="word " * 1000, metadata={"source_file": "test.pdf", "page_number": 1})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1, "5000-char document should produce multiple chunks"


def test_source_metadata_preserved_after_split():
    """Original metadata must be present on every chunk."""
    docs = [Document(page_content="A " * 2000, metadata={"source_file": "paper.pdf", "page_number": 5})]
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert chunk.metadata.get("source_file") == "paper.pdf"
        assert chunk.metadata.get("page_number") == 5


def test_chunk_index_is_sequential():
    """chunk_index must be a sequential integer starting from 0."""
    docs = [Document(page_content="B " * 3000, metadata={})]
    chunks = chunk_documents(docs)
    indices = [c.metadata.get("chunk_index") for c in chunks]
    assert indices == list(range(len(chunks))), "Chunk indices must be 0, 1, 2, ..."


def test_short_document_not_split():
    """A document shorter than chunk_size should remain as one chunk."""
    short_text = "This is a concise research abstract."
    docs = [Document(page_content=short_text, metadata={})]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].page_content == short_text


def test_multiple_pages_indexed_globally():
    """chunk_index must be globally sequential across all input documents."""
    docs = [
        Document(page_content="Page 1 text. " * 200, metadata={"page_number": 1}),
        Document(page_content="Page 2 text. " * 200, metadata={"page_number": 2}),
    ]
    chunks = chunk_documents(docs)
    indices = [c.metadata.get("chunk_index") for c in chunks]
    assert sorted(indices) == list(range(len(chunks)))


def test_empty_list_returns_empty():
    """Empty input should return an empty list without errors."""
    result = chunk_documents([])
    assert result == []
