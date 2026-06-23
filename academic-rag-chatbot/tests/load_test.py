"""
Locust Load Test — Phase 7

Simulates concurrent academic chatbot users to validate performance targets.

Usage:
    # Start the API first: uvicorn app.main:app --port 8000
    locust -f tests/load_test.py --host=http://localhost:8000

Then open: http://localhost:8089
Set: 50 users, spawn rate 5/second → Start swarming

Target metrics (check in Locust web UI):
    Failure rate:               0%
    P95 cached response:        < 100ms
    P95 uncached response:      < 4,000ms
    Requests/second at 50 users: > 10 RPS

Task distribution:
    5x - Common question (likely cache hit after first request)
    2x - Unique question (always a cache miss, full RAG pipeline)
    1x - Health check
    1x - List documents
"""

import uuid

from locust import HttpUser, between, task


class AcademicRagUser(HttpUser):
    wait_time = between(1, 3)   # Simulates user reading/thinking between requests

    @task(5)
    def ask_common_question(self):
        """
        Simulates the cache-hit scenario.
        Many students asking the same common question about a paper.
        After the first request, Redis serves this instantly.
        """
        self.client.post(
            "/api/v1/chat",
            json={
                "question": "What is the main contribution of this paper?",
                "session_id": "load-test-shared-session",
            },
            name="/api/v1/chat [cached]",
        )

    @task(2)
    def ask_unique_question(self):
        """
        Simulates a cache-miss scenario.
        Unique question per request — always triggers the full RAG pipeline.
        """
        unique_suffix = uuid.uuid4().hex[:6]
        self.client.post(
            "/api/v1/chat",
            json={
                "question": f"Explain the experimental methodology described in section {unique_suffix}",
                "session_id": str(uuid.uuid4()),
            },
            name="/api/v1/chat [uncached]",
        )

    @task(1)
    def health_check(self):
        self.client.get("/api/v1/health", name="/api/v1/health")

    @task(1)
    def list_documents(self):
        self.client.get("/api/v1/documents", name="/api/v1/documents")
