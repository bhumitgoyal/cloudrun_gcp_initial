"""
GoHappy Club — WhatsApp Customer Support Chatbot
Hosted on GCP Cloud Run | Vertex AI RAG + Gemini + Firestore + WhatsApp Business API
"""

import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse

from bot.whatsapp import WhatsAppClient
from bot.rag import RAGEngine
from bot.memory import ConversationMemory
from bot.llm import GeminiChat
from bot.rag_cache import RAGCache
from bot.pipeline import MessagePipeline
from bot.evaluator import OutputValidator
from bot.sheets_logger import SheetsAuditLogger

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("gohappy.main")

# ── App lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting GoHappy chatbot service …")
    cache = RAGCache()
    app.state.cache = cache
    app.state.pipeline = MessagePipeline(
        whatsapp=WhatsAppClient(),
        rag=RAGEngine(),
        memory=ConversationMemory(),
        llm=GeminiChat(),
        cache=cache,
        evaluator=OutputValidator(),
        sheets_logger=SheetsAuditLogger(),
    )
    yield
    logger.info("Shutting down …")


app = FastAPI(title="GoHappy Club Support Bot", lifespan=lifespan)


# ── WhatsApp webhook verification (GET) ──────────────────────────────────────
@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Meta requires a one-time GET verification when you register the webhook.
    Respond with the hub.challenge value if the token matches.
    """
    params = dict(request.query_params)
    mode      = params.get("hub.mode")
    token     = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    verify_token = os.environ["WHATSAPP_VERIFY_TOKEN"]

    if mode == "subscribe" and token == verify_token:
        logger.info("Webhook verified successfully.")
        return PlainTextResponse(content=challenge)

    logger.warning("Webhook verification failed — token mismatch.")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


# ── Incoming WhatsApp message (POST) ─────────────────────────────────────────
@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """
    All incoming WhatsApp messages arrive here.
    We respond 200 immediately and handle the message in the background
    so Meta doesn't retry thinking we timed out.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Acknowledge immediately
    background_tasks.add_task(
        app.state.pipeline.handle,
        payload,
    )
    return Response(status_code=200)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "gohappy-support-bot"}


# ── Cache stats ─────────────────────────────────────────────────────────────
@app.get("/cache/stats")
async def cache_stats():
    """Return hit_count, miss_count, hit_rate, and total_cached_entries."""
    return await app.state.cache.get_stats()


@app.post("/cache/invalidate")
async def cache_invalidate():
    """Flush all cached entries (admin use)."""
    await app.state.cache.invalidate_all()
    return {"status": "ok", "message": "Cache invalidated"}


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
