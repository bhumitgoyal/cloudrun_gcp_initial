# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the embedding model during build (avoids cold-start download)
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from build stage
COPY --from=builder /install /usr/local

# Copy pre-downloaded model from build stage cache
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

# Copy application code
COPY --chown=appuser:appuser . .

# Fix ownership of the model cache
RUN chown -R appuser:appuser /home/appuser/.cache

USER appuser

# Cloud Run sets PORT automatically; default 8080
ENV PORT=8080
EXPOSE 8080

# Use exec form to receive SIGTERM properly
CMD ["python", "main.py"]
