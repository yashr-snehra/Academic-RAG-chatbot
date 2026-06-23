"""
Document Chunker — Phase 1 (structure-aware)

Splits full-page Documents into smaller chunks, but first splits each page on
detected section boundaries (e.g. "3 Methods", "RESULTS", "Introduction") so a
chunk never straddles two sections and each chunk is tagged with its section.
This keeps retrieved context topically coherent.

Falls back to plain recursive splitting when a page has no detectable headings,
so behaviour is unchanged for prose pages.

Tuning:
  chunk_size / chunk_overlap come from settings (.env). Smaller chunks = more
  topically focused retrieval; larger = more context per chunk.
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

# Lines that look like an academic section heading.
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{0,60}"          # "3 Methods", "3.1 Setup"
    r"|[A-Z][A-Z0-9 \-]{3,60}"                        # ALL-CAPS heading
    r"|(?:Abstract|Introduction|Background|Related Work|Method|Methods|Methodology|"
    r"Experiment|Experiments|Result|Results|Discussion|Conclusion|Conclusions|"
    r"References|Acknowledgements|Acknowledgments|Appendix|Evaluation|Limitations)"
    r"[^\n]{0,40}"
    r")\s*$"
)


def _looks_like_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line)) and len(line.split()) <= 8


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split one page into [(section_title, section_text), ...]."""
    sections: list[tuple[str, list[str]]] = []
    title = ""
    buf: list[str] = []
    for line in text.split("\n"):
        if _looks_like_heading(line):
            if buf:
                sections.append((title, buf))
            title = line.strip()
            buf = [line]
        else:
            buf.append(line)
    if buf:
        sections.append((title, buf))
    return [(t, "\n".join(lines)) for t, lines in sections]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split full-page Documents into section-aware chunks.

    Original metadata (source_file, page_number) is preserved; a 'section' field
    is added when a heading was detected, and a per-document 'chunk_index' is added
    (0-based within this batch). Each upload is one document, so chunk_index resets
    per document and is NOT unique across documents — pair it with source_file for a
    global key.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[Document] = []
    for doc in documents:
        for section_title, section_text in _split_into_sections(doc.page_content):
            section_doc = Document(page_content=section_text, metadata=dict(doc.metadata))
            for chunk in splitter.split_documents([section_doc]):
                if section_title:
                    chunk.metadata["section"] = section_title
                chunks.append(chunk)

    # Per-document sequential index: 0-based within this batch (one upload = one
    # document). Not unique across documents — combine with source_file if you need
    # a globally unique chunk key.
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    return chunks
