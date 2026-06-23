"""
Performance benchmark for the live RAG API.

Sends a set of academic questions to a running server, each TWICE:
  - 1st call  → cache miss  (full pipeline: retrieve + 2 LLM calls)
  - 2nd call  → cache hit   (served from Redis)

Prints per-query timings plus p50/p95/avg aggregates and the cache speedup.

Usage:
    # start the API first:  uvicorn app.main:app --port 8000
    python scripts/benchmark.py
    python scripts/benchmark.py --base-url http://localhost:8000 --runs 5
"""

import argparse
import statistics
import sys
import time
import uuid

import httpx

QUESTIONS = [
    "What is the main contribution of this paper?",
    "What evaluation metrics were used?",
    "What datasets were used in the experiments?",
    "What are the limitations of the proposed approach?",
    "How does the method compare to prior work?",
]


def _pct(values: list[float], p: float) -> float:
    """Simple percentile (nearest-rank) — avoids a numpy dependency."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * len(s) + 0.5) - 1))
    return s[k]


def _ask(client: httpx.Client, base_url: str, question: str, session_id: str) -> dict:
    start = time.perf_counter()
    resp = client.post(
        f"{base_url}/api/v1/chat",
        json={"question": question, "session_id": session_id},
        timeout=600,
    )
    wall_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    body = resp.json()
    return {
        "wall_ms": wall_ms,
        "server_ms": body.get("latency_ms"),
        "cached": body.get("cached"),
        "n_sources": len(body.get("sources", [])),
        "answer_chars": len(body.get("answer", "")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--runs", type=int, default=len(QUESTIONS),
                    help="How many distinct questions to send (max = built-in list).")
    args = ap.parse_args()

    questions = QUESTIONS[: max(1, min(args.runs, len(QUESTIONS)))]
    session_id = f"bench-{uuid.uuid4().hex[:8]}"

    with httpx.Client() as client:
        # Health check
        try:
            t0 = time.perf_counter()
            client.get(f"{args.base_url}/api/v1/health", timeout=10).raise_for_status()
            health_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"ERROR: API not reachable at {args.base_url} ({e})")
            print("Start it first:  uvicorn app.main:app --port 8000")
            sys.exit(1)

        print(f"\nBenchmarking {args.base_url}  ·  {len(questions)} questions  ·  session {session_id}")
        print(f"Health check: {health_ms:.1f} ms\n")
        print(f"{'#':>2}  {'question':<46}  {'uncached':>10}  {'cached':>9}  {'sources':>7}")
        print("-" * 84)

        uncached, cached = [], []
        for i, q in enumerate(questions, 1):
            miss = _ask(client, args.base_url, q, session_id)   # cache miss
            hit = _ask(client, args.base_url, q, session_id)    # cache hit
            uncached.append(miss["wall_ms"])
            cached.append(hit["wall_ms"])
            flag = "" if hit["cached"] else "  (!! not cached)"
            print(f"{i:>2}  {q[:46]:<46}  {miss['wall_ms']:>8.0f}ms  {hit['wall_ms']:>7.0f}ms  "
                  f"{miss['n_sources']:>7}{flag}")

        def summary(label: str, vals: list[float]) -> None:
            print(f"  {label:<10}  avg {statistics.mean(vals):>8.0f} ms   "
                  f"p50 {_pct(vals, 50):>8.0f} ms   p95 {_pct(vals, 95):>8.0f} ms   "
                  f"min {min(vals):>7.0f} ms   max {max(vals):>7.0f} ms")

        speedup = statistics.mean(uncached) / statistics.mean(cached) if cached else 0
        print("\n" + "=" * 84)
        print("PERFORMANCE SUMMARY")
        print("=" * 84)
        summary("uncached", uncached)
        summary("cached", cached)
        print(f"\n  cache speedup: {speedup:.0f}x faster on a hit "
              f"({statistics.mean(uncached):.0f} ms -> {statistics.mean(cached):.0f} ms)")


if __name__ == "__main__":
    main()
