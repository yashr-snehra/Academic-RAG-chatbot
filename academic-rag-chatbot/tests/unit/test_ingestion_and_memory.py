"""Tests for structure-aware chunking, multi-format loading, and bounded history."""

from langchain_core.documents import Document

from app.config import settings
from app.core.generation.chain import trim_history
from app.core.ingestion.chunker import chunk_documents
from app.core.ingestion.loader import load_document, load_text


# ── bounded conversation memory ───────────────────────────────────────────────

def test_trim_history_caps_to_max():
    msgs = list(range(50))
    out = trim_history(msgs)
    assert len(out) == settings.max_history_messages
    assert out == msgs[-settings.max_history_messages:]


def test_trim_history_keeps_short_history():
    msgs = [1, 2, 3]
    assert trim_history(msgs) == msgs


# ── structure-aware chunking ──────────────────────────────────────────────────

def test_chunker_tags_sections():
    page = "1. Introduction\nWe study X.\n2. Methods\nWe use Y.\n3. Results\nWe found Z."
    docs = [Document(page_content=page, metadata={"source_file": "p", "page_number": 1})]
    chunks = chunk_documents(docs)
    sections = {c.metadata.get("section") for c in chunks}
    assert any(s and "Introduction" in s for s in sections)
    assert any(s and "Methods" in s for s in sections)


def test_chunker_does_not_cross_sections():
    page = "1. Methods\n" + ("m " * 50) + "\n2. Results\n" + ("r " * 50)
    docs = [Document(page_content=page, metadata={})]
    chunks = chunk_documents(docs)
    for c in chunks:
        assert not ("m m m" in c.page_content and "r r r" in c.page_content)


def test_chunker_still_handles_headingless_pages():
    docs = [Document(page_content="word " * 1000, metadata={"source_file": "x", "page_number": 1})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


# ── multi-format loading ──────────────────────────────────────────────────────

def test_load_text_file(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Title\nSome notes about the experiment.", encoding="utf-8")
    docs = load_text(f)
    assert len(docs) == 1
    assert docs[0].metadata["source_file"] == "notes"
    assert "experiment" in docs[0].page_content


def test_load_document_dispatches_text(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("plain content", encoding="utf-8")
    assert load_document(f)[0].page_content == "plain content"


def test_load_document_rejects_unknown(tmp_path):
    f = tmp_path / "a.docx"
    f.write_text("x", encoding="utf-8")
    try:
        load_document(f)
        assert False, "expected ValueError for unsupported type"
    except ValueError:
        pass
