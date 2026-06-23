"""
Batch PDF Ingestion Script

Ingests all PDF files in a directory (or a single PDF) into Qdrant.
Use this for initial data loading or re-ingestion after changing chunk parameters.

Usage:
    poetry run python scripts/ingest.py                    # Ingests data/pdfs/
    poetry run python scripts/ingest.py data/pdfs/         # Same as above
    poetry run python scripts/ingest.py path/to/paper.pdf  # Single file

After running, verify in Qdrant dashboard: http://localhost:6333/dashboard
"""

import sys
import time
from pathlib import Path

# Add project root to Python path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from app.config import settings
from app.core.ingestion.chunker import chunk_documents
from app.core.ingestion.embedder import ensure_collection, store_chunks
from app.core.ingestion.loader import load_pdf


def ingest_file(pdf_path: Path, client: QdrantClient) -> dict:
    """Ingest a single PDF. Returns a summary dict."""
    start = time.monotonic()
    print(f"\n  Loading:  {pdf_path.name}")

    docs = load_pdf(pdf_path)
    if not docs:
        print(f"  WARNING: No text extracted from {pdf_path.name}. Skipping.")
        return {"file": pdf_path.name, "pages": 0, "chunks": 0, "status": "skipped"}

    print(f"  Pages:    {len(docs)}")

    chunks = chunk_documents(docs)
    print(f"  Chunks:   {len(chunks)}")

    count = store_chunks(chunks, client)
    elapsed = time.monotonic() - start

    print(f"  Stored:   {count} vectors in Qdrant")
    print(f"  Time:     {elapsed:.1f}s")

    return {"file": pdf_path.name, "pages": len(docs), "chunks": count, "status": "ok"}


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/pdfs")

    if target.is_file():
        if not target.suffix.lower() == ".pdf":
            print(f"Error: {target} is not a PDF file.")
            sys.exit(1)
        pdfs = [target]
    elif target.is_dir():
        pdfs = sorted(target.glob("*.pdf"))
    else:
        print(f"Error: {target} does not exist.")
        sys.exit(1)

    if not pdfs:
        print(f"No PDF files found in {target}/")
        print("Drop some academic PDFs into data/pdfs/ and try again.")
        sys.exit(0)

    print(f"Connecting to Qdrant at {settings.qdrant_url}...")
    client = QdrantClient(url=settings.qdrant_url)
    ensure_collection(client)

    print(f"\nIngesting {len(pdfs)} PDF(s) into collection '{settings.qdrant_collection}'...")
    print("=" * 60)

    results = []
    for pdf in pdfs:
        result = ingest_file(pdf, client)
        results.append(result)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    total_chunks = 0
    for r in results:
        status_icon = "✓" if r["status"] == "ok" else "!"
        print(f"  {status_icon} {r['file']:<40} {r['chunks']:>5} chunks")
        total_chunks += r["chunks"]

    print(f"\nTotal chunks stored: {total_chunks}")
    print(f"Collection: {settings.qdrant_collection}")
    print(f"\nVerify at: http://localhost:6333/dashboard")
    print("Next step: run 'python scripts/test_retrieval.py' to validate quality.")


if __name__ == "__main__":
    main()
