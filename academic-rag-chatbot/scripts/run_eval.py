"""
Ragas Evaluation CLI Script

Runs the full automated quality evaluation pipeline and prints results.

Usage:
    poetry run python scripts/run_eval.py

Prerequisites:
  1. At least one PDF ingested into Qdrant
  2. evaluation/datasets/test_qa.json populated with real Q&A pairs
  3. Ollama running with the configured models pulled (llama3.1, nomic-embed-text)

Output:
  - Console: per-metric scores with pass/fail against targets
  - File: evaluation/results.csv (full per-question scores for analysis)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.runnables.history import RunnableWithMessageHistory
from qdrant_client import QdrantClient

from app.config import settings
from app.core.generation.chain import build_rag_chain
from app.core.memory.history import get_session_history
from app.core.retrieval.retriever import get_retriever
from evaluation.pipeline import run_evaluation


def main() -> None:
    dataset_path = Path("evaluation/datasets/test_qa.json")

    if not dataset_path.exists():
        print(f"ERROR: Test dataset not found at {dataset_path}")
        print("\nCreate evaluation/datasets/test_qa.json with this structure:")
        print('  [{"question": "...", "ground_truth": "..."}, ...]')
        print("\nSee the example file — replace the placeholder answers with real ones from your papers.")
        sys.exit(1)

    import json
    with open(dataset_path) as f:
        cases = json.load(f)

    placeholder_count = sum(1 for c in cases if c.get("ground_truth", "").startswith("REPLACE"))
    if placeholder_count > 0:
        print(f"WARNING: {placeholder_count} test case(s) still have placeholder ground_truth values.")
        print("These will skew Context Recall scores. Update test_qa.json with real answers.")
        print()

    print(f"Connecting to Qdrant at {settings.qdrant_url}...")
    client = QdrantClient(url=settings.qdrant_url)
    retriever = get_retriever(client)
    rag_chain = build_rag_chain(retriever)

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    summary = run_evaluation(chain_with_history, dataset_path)

    # Exit with non-zero code if any metric misses target (useful for CI pipelines)
    targets = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.75,
    }

    all_passing = all(summary[m] >= t for m, t in targets.items())
    sys.exit(0 if all_passing else 1)


if __name__ == "__main__":
    main()
