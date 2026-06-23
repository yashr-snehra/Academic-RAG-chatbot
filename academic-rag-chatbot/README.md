# Academic RAG Chatbot

Enterprise-grade Retrieval-Augmented Generation (RAG) chatbot for academic literature.
Upload PDFs, ask questions, receive grounded answers with page-level citations.

## Tech Stack

| Tool | Role |
|---|---|
| **LangChain** | RAG chain orchestration + conversational memory |
| **Qdrant** | Vector database for semantic chunk retrieval |
| **FastAPI** | Async REST API layer |
| **Redis** | Response cache + chat session persistence |
| **Ollama** | Local LLM + embeddings (llama3.2:3b / nomic-embed-text) — no API key |
| **Ragas** | Automated quality evaluation (faithfulness, relevancy) |

## Prerequisites

- Python 3.11+ (3.13 recommended)
- Docker Desktop
- [Ollama](https://ollama.com/download) — runs the LLM + embeddings locally (no API key, no cost):
  ```bash
  ollama pull llama3.2:3b       # chat model
  ollama pull nomic-embed-text  # embedding model (768-dim)
  ```
- Poetry is optional — pip + `requirements.txt` works too.

## Quick Start

### One command (Windows)

With Ollama installed and running, just run:

```powershell
./run.ps1
```

This creates the virtualenv, installs dependencies, starts Qdrant + Redis (launching
Docker Desktop if needed), pulls any missing models, starts the API, and opens the
chat UI at http://localhost:8000. Then ingest a paper: `./.venv/Scripts/python.exe scripts/ingest.py path/to/paper.pdf`.

### Manual setup

```bash
# 1. Install dependencies (pick one)
pip install -r requirements.txt      # pip
poetry install                       # or Poetry

# 2. Configure environment
cp .env.example .env
# Defaults work out of the box — no API key needed. Just make sure Ollama is
# running with llama3.2:3b and nomic-embed-text pulled.

# 3. Start infrastructure (Qdrant + Redis)
docker compose -f docker/docker-compose.yml up -d

# 4. Start the API (chat UI at http://localhost:8000, docs at /docs)
uvicorn app.main:app --reload --port 8000
```

## Faster inference on Intel Arc GPU (optional)

Standard Ollama runs on **CPU** on Intel machines. To offload to an Intel Arc GPU
(including the integrated Arc on Lunar Lake laptops, e.g. Arc 140V), use Intel's
**IPEX-LLM Ollama portable build** — same `localhost:11434`, no app or code changes.

1. (Optional) Remove standard Ollama so it doesn't hold the port: `winget uninstall Ollama.Ollama`.
   Your pulled models in `~/.ollama` are kept and reused.
2. Download the newest `ollama-ipex-llm-*-win.zip` from the
   [ipex-llm release](https://github.com/ipex-llm/ipex-llm/releases/tag/v2.3.0-nightly)
   (note the org is `ipex-llm/ipex-llm`). Unzip to `C:\ipex-ollama`.
3. Make sure your Intel Arc GPU driver is current.
4. Run the project: **`./run.ps1`** starts the portable build automatically (it sets
   `OLLAMA_NUM_GPU=999`, flash attention, and an 8-bit KV cache). The first GPU start
   compiles SYCL kernels and can take ~2 min; later starts are fast.

To start the GPU server by hand instead: `C:\ipex-ollama\start-ollama.bat`.
Verify it's on the GPU: the serve log shows `using Intel GPU` and `SYCL0` buffers.
Measured on an Arc 140V: ~36 tok/s for llama3.2:3b (vs CPU-bound otherwise).

## Usage

### Ingest academic PDFs

```bash
# Drop PDFs into data/pdfs/ then run:
poetry run python scripts/ingest.py

# Or ingest a single file:
poetry run python scripts/ingest.py path/to/paper.pdf
```

### Validate retrieval quality (do this before testing the chatbot)

```bash
poetry run python scripts/test_retrieval.py
```

### Chat in the browser

Open **http://localhost:8000** — a built-in chat UI with source citations, per-answer
latency, and multi-turn memory. No separate frontend to run.

### Chat via API

```bash
# Ask a question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main contribution of this paper?",
    "session_id": "my-session-uuid"
  }'

# Ask a follow-up (same session_id maintains conversation context)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What evaluation metrics did they use for this?",
    "session_id": "my-session-uuid"
  }'

# Stream the answer token-by-token (Server-Sent Events) — this is what the web UI uses
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What datasets were used?", "session_id": "my-session-uuid"}'
# → data: {"type":"token","content":"..."}    (one frame per generated token)
# → data: {"type":"done","sources":[...],"cached":false,"latency_ms":...}
# (cache hits arrive as a single token frame; errors as {"type":"error","message":...})

# List ingested documents
curl http://localhost:8000/api/v1/documents

# Get conversation history
curl http://localhost:8000/api/v1/chat/my-session-uuid/history
```

### Run quality evaluation (Ragas)

```bash
# 1. Edit evaluation/datasets/test_qa.json with real Q&A pairs from your papers
# 2. Run:
poetry run python scripts/run_eval.py
# 3. View full results:
cat evaluation/results.csv
```

### Run tests

```bash
# All tests with coverage report
poetry run pytest tests/ --cov=app --cov-report=term-missing

# Unit tests only
poetry run pytest tests/unit/ -v

# Integration tests only
poetry run pytest tests/integration/ -v

# Load test (start the API first, then in a separate terminal):
locust -f tests/load_test.py --host=http://localhost:8000
# Open: http://localhost:8089
```

## Quality Targets

| Metric | Target |
|---|---|
| Ragas Faithfulness | ≥ 0.85 |
| Ragas Answer Relevancy | ≥ 0.80 |
| Ragas Context Precision | ≥ 0.75 |
| Ragas Context Recall | ≥ 0.75 |
| P95 cached response | < 100ms |
| P95 uncached response | < 4,000ms |
| Test coverage (mocked CI) | ≥ 75% |

## Project Structure

```
run.ps1                         One-command setup + launch (Windows)
requirements.txt                Pinned deps for pip users

app/
├── main.py                     FastAPI app + lifespan + chat UI route
├── config.py                   Pydantic settings (reads .env)
├── static/index.html           Built-in chat web UI
├── api/
│   ├── deps.py                 Dependency injectors
│   └── routes/
│       ├── chat.py             POST /chat, session management
│       ├── documents.py        Upload + list PDFs
│       └── evaluation.py       Trigger Ragas eval
├── core/
│   ├── ingestion/              PDF load → chunk → embed → store
│   ├── retrieval/              Qdrant-backed LangChain retriever
│   ├── generation/             RAG chain, prompts, adaptive token budget, citations
│   └── memory/                 Redis cache + session history
└── models/schemas.py           Pydantic I/O models

evaluation/                     Ragas pipeline + test datasets
tests/                          Unit, integration, and load tests
scripts/                        CLI tools: ingest, test_retrieval, run_eval, benchmark
docker/                         Docker Compose + Dockerfile
```

## Tuning Guide

**Faithfulness is low (< 0.85):** Strengthen the system prompt. Add explicit "never use outside knowledge" instructions.

**Context Precision is low (< 0.75):** Reduce `CHUNK_SIZE` to 400-500 in `.env` and re-ingest. Smaller chunks are more topically focused.

**Context Recall is low (< 0.75):** Increase `RETRIEVAL_TOP_K` to 8-10, or increase `CHUNK_OVERLAP` to 150 and re-ingest.

**Slow uncached responses:** Local inference speed depends on your hardware.
- **Answer length** auto-adapts per question (`dynamic_num_predict` in `app/core/generation/chain.py`): brief/factual questions cap at `LLM_NUM_PREDICT_BRIEF` tokens, detailed ones at `LLM_NUM_PREDICT_DETAILED`. Tune these in `.env`.
- **GPU acceleration** is the biggest lever. NVIDIA/AMD/Apple GPUs are used automatically. For **Intel Arc** (incl. integrated Arc on Lunar Lake), the stock Ollama runs on CPU — use Intel's [IPEX-LLM Ollama](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/ollama_quickstart.md) build to offload to the iGPU.
- **Keep the model warm** with `LLM_KEEP_ALIVE` (default `30m`) so back-to-back questions skip the cold reload.
- A cache hit returns in single-digit milliseconds regardless of model.
