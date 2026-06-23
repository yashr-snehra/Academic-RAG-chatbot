"""
Evaluation Router — Phase 6

Endpoint:
  POST /api/v1/evaluate — Trigger the Ragas evaluation pipeline

The actual evaluation runs in a background task and writes results to
evaluation/results.csv. For production, you'd stream progress via WebSocket
or return a job_id that clients poll. This simple version suffices for development.
"""

import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.rate_limit import rate_limit

router = APIRouter(prefix="/evaluate", tags=["evaluation"])

# app/api/routes/evaluation.py -> parents[3] is the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _run_evaluation_task() -> None:
    """
    Background task: run the full Ragas evaluation pipeline.

    Reads from:   evaluation/datasets/test_qa.json
    Writes to:    evaluation/results.csv

    Requires Ollama running (with the configured models pulled) and Qdrant to be populated.
    """
    script = _PROJECT_ROOT / "scripts" / "run_eval.py"
    try:
        print("[eval] Starting Ragas evaluation pipeline...")
        # sys.executable -> the same (venv) interpreter running the API, not
        # whatever "python" happens to be on PATH. Absolute script path + explicit
        # cwd so the script's relative data paths resolve regardless of caller CWD.
        subprocess.run(
            [sys.executable, str(script)],
            check=True,
            cwd=str(_PROJECT_ROOT),
        )
        print("[eval] Evaluation complete. Results in evaluation/results.csv")
    except Exception as e:
        print(f"[eval] Evaluation failed: {e}")


@router.post(
    "",
    summary="Trigger Ragas evaluation pipeline",
    dependencies=[Depends(rate_limit(2, 300))],  # 2 runs / 5 min / IP (heavy job)
)
async def trigger_evaluation(background_tasks: BackgroundTasks):
    """
    Start a Ragas evaluation run in the background.

    Prerequisites:
      1. At least one PDF must be ingested into Qdrant
      2. evaluation/datasets/test_qa.json must exist with your Q&A test pairs

    Results are written to evaluation/results.csv when complete.
    Check server logs for progress updates.
    """
    background_tasks.add_task(_run_evaluation_task)
    return {
        "status": "evaluation_started",
        "message": (
            "Ragas evaluation running in background. "
            "Check evaluation/results.csv when complete (may take several minutes)."
        ),
    }
