from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM (local, via Ollama) ───────────────────────────────────────────────
    # No API key needed — everything runs locally through the Ollama server.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"               # chat/generation model
    ollama_embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768              # nomic-embed-text output size;
    #                                              MUST match the Qdrant vector size.
    #                                              Change this if you switch embed models.
    llm_temperature: float = 0.0                 # 0 = most faithful; never raise for RAG
    # Adaptive generation budget — computed continuously per question by
    # dynamic_num_predict(): base + per-word scaling, adjusted by intent and multi-part
    # structure, then clamped to [min, max]. num_predict is a cap, not a target.
    llm_num_predict_min: int = 128               # floor (a one-word answer still fits)
    llm_num_predict_max: int = 1536              # ceiling (longest multi-part answers)
    llm_num_predict_default: int = 512           # fallback when no per-question budget is passed
    #                                              (e.g. eval/scripts that build the chain directly)
    llm_num_predict_per_word: int = 24           # budget added per word of the question
    llm_keep_alive: str = "30m"                  # keep model resident in VRAM/RAM between requests
    llm_request_timeout: int = 120               # seconds to wait on the Ollama server

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Browsers (and the CORS spec) forbid wildcard origins together with
    # credentials, so credentials are auto-disabled while this is "*". Set an
    # explicit origin list (e.g. ["https://app.example.com"]) in production.
    cors_allow_origins: list[str] = ["*"]

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "academic_docs"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600      # 1 hour — how long to cache API responses
    session_ttl_seconds: int = 86400   # 24 hours — how long to keep chat history
    max_history_messages: int = 10     # cap turns fed to the LLM so prompts stay bounded

    # ── Retrieval tuning ──────────────────────────────────────────────────────
    retrieval_top_k: int = 6     # Number of chunks to retrieve per query
    chunk_size: int = 700        # Target chunk size in characters
    chunk_overlap: int = 100     # Overlap between consecutive chunks

    # ── Hybrid search (dense + BM25 sparse) ───────────────────────────────────
    # OFF by default. Turning this ON changes the Qdrant collection schema (named
    # "dense" vector + "sparse" vector), so you MUST recreate the collection and
    # re-ingest. Requires the `fastembed` package (pip install fastembed).
    hybrid_enabled: bool = False
    sparse_model: str = "Qdrant/bm25"   # FastEmbed sparse (BM25) model

    # ── Reranking (cross-encoder) ─────────────────────────────────────────────
    # OFF by default. When ON, retrieval fetches top_k * fetch_multiplier candidates
    # then a cross-encoder reranks them down to top_k. Requires `sentence-transformers`
    # (first run downloads the model, ~80MB).
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_fetch_multiplier: int = 4    # candidates to fetch before reranking = top_k * this

    # ── Context-aware generation budget ───────────────────────────────────────
    # After retrieval, the answer token budget gets extra headroom proportional to
    # how much relevant context was retrieved (a question backed by lots of context
    # may warrant a longer synthesis). Still clamped to llm_num_predict_max.
    context_budget_per_1k_chars: int = 80   # extra tokens per 1k chars of retrieved context
    context_budget_max_bonus: int = 512     # cap on that bonus

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Single shared instance — import this everywhere, never re-instantiate
settings = Settings()
