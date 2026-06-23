# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install the pinned deps into an isolated prefix so the runtime stage stays lean.
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Lean production runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the installed packages (from --prefix=/install) into the runtime image
COPY --from=builder /install /usr/local

# Copy only what the app needs at runtime
COPY app/ ./app/
COPY evaluation/ ./evaluation/
COPY scripts/ ./scripts/

# Run as non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
