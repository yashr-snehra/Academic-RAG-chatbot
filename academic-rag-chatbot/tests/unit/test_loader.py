"""
Unit tests for the PDF loader.

Builds a small PDF on the fly with PyMuPDF (fitz) so the test is hermetic —
no fixture files on disk, no network.
"""

from pathlib import Path

import fitz

from app.core.ingestion.loader import load_pdf


def _make_pdf(path: Path, pages: list[str]) -> None:
    """Write a PDF where each string in `pages` becomes one page (empty string = blank page)."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_loads_one_document_per_nonempty_page(tmp_path):
    pdf = tmp_path / "sample_paper.pdf"
    _make_pdf(pdf, ["Introduction text.", "Methods text.", "Results text."])

    docs = load_pdf(pdf)

    assert len(docs) == 3
    assert [d.metadata["page_number"] for d in docs] == [1, 2, 3]


def test_blank_pages_are_skipped(tmp_path):
    pdf = tmp_path / "with_blank.pdf"
    _make_pdf(pdf, ["Page one.", "", "Page three."])

    docs = load_pdf(pdf)

    # The blank middle page must be dropped, but page numbers stay true to the source.
    assert len(docs) == 2
    assert [d.metadata["page_number"] for d in docs] == [1, 3]


def test_metadata_uses_filename_stem_and_total_pages(tmp_path):
    pdf = tmp_path / "resnet_paper.pdf"
    _make_pdf(pdf, ["Only page."])

    docs = load_pdf(pdf)

    assert docs[0].metadata["source_file"] == "resnet_paper"   # no .pdf extension
    assert docs[0].metadata["total_pages"] == 1
    assert "Only page." in docs[0].page_content


def test_accepts_string_path(tmp_path):
    pdf = tmp_path / "as_string.pdf"
    _make_pdf(pdf, ["Hello."])

    docs = load_pdf(str(pdf))  # str, not Path

    assert len(docs) == 1
