"""
Retrieval Quality Validation Script

Run this BEFORE building the RAG chain to verify that Qdrant returns
semantically relevant chunks for your academic queries.

If retrieval is bad, the chain will be bad — fix retrieval first.

Usage:
    poetry run python scripts/test_retrieval.py

What to look for in the output:
  GOOD: Results are from relevant sections of the paper, on topic with the query
  BAD:  Results are from unrelated sections, or chunks are too short/too generic

If results are bad, try:
  - Reducing CHUNK_SIZE in .env (e.g., 400-500) → re-ingest
  - Increasing CHUNK_OVERLAP in .env (e.g., 150) → re-ingest
  - Increasing RETRIEVAL_TOP_K in .env (e.g., 8-10)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from app.config import settings
from app.core.retrieval.retriever import get_retriever

# ── Customize these queries to match your uploaded papers ────────────────────
TEST_QUERIES = [
    "What is the main contribution of this paper?",
    "What evaluation metrics were used to assess performance?",
    "What dataset was used for training and testing?",
    "What are the limitations of the proposed approach?",
    "How does the proposed method compare to existing baselines?",
]


def main() -> None:
    print(f"Connecting to Qdrant at {settings.qdrant_url}...")
    client = QdrantClient(url=settings.qdrant_url)

    # Check that the collection exists and has data
    try:
        collections = [c.name for c in client.get_collections().collections]
        if settings.qdrant_collection not in collections:
            print(f"\nERROR: Collection '{settings.qdrant_collection}' does not exist.")
            print("Run 'python scripts/ingest.py' first to ingest your PDFs.")
            sys.exit(1)

        count_result = client.count(collection_name=settings.qdrant_collection)
        if count_result.count == 0:
            print(f"\nERROR: Collection '{settings.qdrant_collection}' is empty.")
            print("Run 'python scripts/ingest.py' to ingest your PDFs first.")
            sys.exit(1)

        print(f"Collection '{settings.qdrant_collection}' has {count_result.count} points.\n")

    except Exception as e:
        print(f"ERROR connecting to Qdrant: {e}")
        sys.exit(1)

    retriever = get_retriever(client)

    for query in TEST_QUERIES:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = retriever.invoke(query)

        if not results:
            print("  !! No results returned. Is the collection populated?")
            continue

        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            source = meta.get("source_file", "unknown")
            page = meta.get("page_number", "?")
            chunk = meta.get("chunk_index", "?")

            print(f"\n  [{i}] {source}  |  Page {page}  |  Chunk #{chunk}")
            print(f"  {'-' * 60}")
            # Show first 350 chars of the chunk
            preview = doc.page_content[:350].replace("\n", " ")
            print(f"  {preview}...")

    print("\n\n" + "=" * 70)
    print("RETRIEVAL CHECK COMPLETE")
    print("=" * 70)
    print("\nIf the results look relevant → proceed to Phase 3 (build the RAG chain)")
    print("If the results look off-topic → adjust chunking params and re-ingest")
    print(f"\nCurrent settings: chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}, k={settings.retrieval_top_k}")


if __name__ == "__main__":
    main()
