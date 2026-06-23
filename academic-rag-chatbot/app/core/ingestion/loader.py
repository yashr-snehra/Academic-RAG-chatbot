"""
Document Loader — Phase 1

Loads source documents into LangChain Documents carrying citation metadata
(source_file, page_number, total_pages).

Supported formats:
  .pdf            — PyMuPDF (fitz): handles multi-column academic layouts, fast
  .txt .md .markdown — plain text (treated as a single page)

Use load_document() — it dispatches on the file extension.
"""

from pathlib import Path

import fitz  # pip package: pymupdf
from langchain_core.documents import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = {".pdf"} | TEXT_SUFFIXES


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a PDF and return one Document per non-empty page, with citation metadata.
    """
    path = Path(file_path)
    doc = fitz.open(str(path))
    documents: list[Document] = []
    source_name = path.stem

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # "text" mode preserves reading order
        if not text.strip():
            continue
        cleaned_text = "\n".join(line for line in text.splitlines() if line.strip())
        documents.append(
            Document(
                page_content=cleaned_text,
                metadata={
                    "source_file": source_name,
                    "page_number": page_num,
                    "total_pages": len(doc),
                },
            )
        )

    doc.close()
    return documents


def load_text(file_path: str | Path) -> list[Document]:
    """
    Load a plain-text or markdown file as a single-page Document.
    """
    path = Path(file_path)
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return []
    return [
        Document(
            page_content=content,
            metadata={"source_file": path.stem, "page_number": 1, "total_pages": 1},
        )
    ]


def load_document(file_path: str | Path) -> list[Document]:
    """
    Load any supported document, dispatching on its extension.

    Raises:
        ValueError: if the extension is not supported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in TEXT_SUFFIXES:
        return load_text(path)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
    )
