"""
Ragas Evaluation Pipeline — Phase 6

Runs automated quality evaluation using four metrics:

  Faithfulness (0-1):
    Are all claims in the answer supported by the retrieved context?
    Low score = hallucination. Target: >= 0.85
    Fix: Strengthen system prompt. Add "ONLY use the context" rules.

  Answer Relevancy (0-1):
    Is the generated answer actually relevant to the asked question?
    Low score = off-topic answers. Target: >= 0.80
    Fix: Improve contextualize prompt. Increase retrieval k.

  Context Precision (0-1):
    What fraction of retrieved chunks were actually useful for answering?
    Low score = noisy retrieval. Target: >= 0.75
    Fix: Reduce chunk_size. Add metadata filtering.

  Context Recall (0-1):
    Did we retrieve all the context needed to produce a complete answer?
    (Requires ground_truth in test dataset.) Target: >= 0.75
    Fix: Increase retrieval_top_k. Increase chunk_overlap.

Improvement loop:
  1. Run evaluation → get baseline scores
  2. Identify lowest metric
  3. Apply the corresponding fix from above
  4. Re-ingest PDFs if you changed chunking parameters
  5. Re-run evaluation → compare against baseline
  6. Repeat until all metrics >= target thresholds
"""

import json
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from app.config import settings


def load_test_dataset(path: str | Path) -> list[dict]:
    """Load Q&A test pairs from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data


def run_evaluation(
    chain_with_history,
    test_dataset_path: str | Path = "evaluation/datasets/test_qa.json",
) -> dict:
    """
    Run Ragas evaluation against the test dataset.

    Args:
        chain_with_history: A RunnableWithMessageHistory wrapping the RAG chain.
        test_dataset_path: Path to the JSON file with test Q&A pairs.

    Returns:
        Dict with mean scores for each metric and the sample count.
    """
    test_cases = load_test_dataset(test_dataset_path)
    n = len(test_cases)
    print(f"\n[ragas] Starting evaluation on {n} test cases...\n")

    questions: list[str] = []
    ground_truths: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []

    for i, case in enumerate(test_cases, 1):
        q = case["question"]
        print(f"  [{i:02d}/{n:02d}] {q[:70]}...")

        # Fresh session per question to prevent cross-contamination of history
        result = chain_with_history.invoke(
            {"input": q},
            config={"configurable": {"session_id": f"ragas-eval-{i}"}},
        )

        questions.append(q)
        ground_truths.append(case["ground_truth"])
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in result.get("context", [])])

    # Build a Ragas 0.2 EvaluationDataset. Note the 0.2 column names:
    #   user_input / response / retrieved_contexts / reference
    # (these replaced the old question / answer / contexts / ground_truth.)
    samples = [
        {
            "user_input": q,
            "response": a,
            "retrieved_contexts": ctx,
            "reference": gt,
        }
        for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths)
    ]
    dataset = EvaluationDataset.from_list(samples)

    # Ragas itself needs an LLM + embeddings to score the metrics — run those
    # locally through Ollama too, so no external API is ever called.
    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
    )

    print("\n[ragas] Computing metrics (runs locally via Ollama, may take a while)...")

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = result.to_pandas()
    output_path = "evaluation/results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[ragas] Full results saved to {output_path}")

    # Metric column names vary slightly across ragas versions (e.g. context precision
    # may be 'llm_context_precision_with_reference'), so match by substring.
    def _mean_for(token: str) -> float:
        for col in df.columns:
            if token in col:
                return round(float(df[col].mean()), 3)
        return 0.0

    summary = {
        "faithfulness":       _mean_for("faithfulness"),
        "answer_relevancy":   _mean_for("answer_relevancy"),
        "context_precision":  _mean_for("context_precision"),
        "context_recall":     _mean_for("context_recall"),
        "n_samples": n,
    }

    targets = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.75,
    }

    print("\n[ragas] Summary:")
    for metric, score in summary.items():
        if metric == "n_samples":
            continue
        target = targets[metric]
        status = "✓" if score >= target else "✗"
        print(f"  {status} {metric:<22} {score:.3f}  (target: >= {target})")

    return summary
