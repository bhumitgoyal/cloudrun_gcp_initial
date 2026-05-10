# ── Dockerfile (App Engine Flex / local dev) ──────────────────────────────────
# App Engine Standard (python311) deploys directly from source — no Dockerfile
# needed. This file is kept for:
#   1. Local Docker-based development/testing
#   2. Potential future migration to App Engine Flexible
#
# App Engine Standard deployment:
#   gcloud app deploy app.yaml --project=ghc-chatbot --quiet
# ──────────────────────────────────────────────────────────────────────────────

# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt




# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from build stage
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=appuser:appuser . .

USER appuser

# App Engine sets PORT automatically; default 8080
ENV PORT=8080
EXPOSE 8080

# Use exec form to receive SIGTERM properly
CMD ["python", "main.py"]
