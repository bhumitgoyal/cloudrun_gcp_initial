"""
GoHappy Club — Customer Support Chatbot (WhatsApp + In-App)
Hosted on GCP App Engine | Vertex AI RAG + Gemini + Firestore

Channels:
  - WhatsApp Business API (webhook at /webhook)
  - In-App REST API (POST /api/chat, GET /api/chat/history/{user_id})
"""

import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional, List

from bot.whatsapp import WhatsAppClient
from bot.rag import RAGEngine
from bot.memory import ConversationMemory
from bot.llm import GeminiChat
from bot.rag_cache import RAGCache
from bot.pipeline import MessagePipeline
from bot.evaluator import OutputValidator
from bot.sheets_logger import SheetsAuditLogger
from bot.kb_insights import KBInsightsGenerator
from bot.kb_manager import KnowledgeBaseManager

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
    sheets_logger = SheetsAuditLogger()
    app.state.pipeline = MessagePipeline(
        whatsapp=WhatsAppClient(),
        rag=RAGEngine(),
        memory=ConversationMemory(),
        llm=GeminiChat(),
        cache=cache,
        evaluator=OutputValidator(),
        sheets_logger=sheets_logger,
        kb_insights=KBInsightsGenerator(sheets_logger),
        kb_manager=KnowledgeBaseManager()
    )
    yield
    logger.info("Shutting down …")


app = FastAPI(title="GoHappy Club Support Bot", lifespan=lifespan)

# ── CORS (allow mobile / web app clients) ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models for in-app chat API ──────────────────────────────────────
class ChatRequest(BaseModel):
    user_id: str                          # Stable user identifier (Firebase UID, phone, etc.)
    display_name: str = "App User"        # User's display name
    message: str                          # The chat message text
    message_id: Optional[str] = None      # Optional, for client-side deduplication

class ChatResponse(BaseModel):
    reply: str                            # Bot's reply text
    escalation: bool = False              # True if escalated to human support

class ChatHistoryTurn(BaseModel):
    role: str                             # "user" or "assistant"
    content: str
    ts: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    display_name: str = ""
    turns: List[dict] = []
    summary: str = ""


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


# ══════════════════════════════════════════════════════════════════════════════
#  IN-APP CHAT API
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat", response_model=ChatResponse)
async def app_chat(req: ChatRequest):
    """
    In-app chatbot endpoint.  Send a message and receive the bot's reply
    synchronously in the response body.

    Request body:
        {
            "user_id": "firebase_uid_or_phone",
            "display_name": "Ramesh",
            "message": "What is GoHappy Club?",
            "message_id": "optional-uuid-for-dedup"
        }
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message field cannot be empty")

    result = await app.state.pipeline.handle_app_message(
        user_id=req.user_id,
        display_name=req.display_name,
        text=req.message.strip(),
        message_id=req.message_id,
    )

    if result is None:
        # Message was filtered (junk/duplicate/greeting) — return empty reply
        return ChatResponse(reply="", escalation=False)

    return ChatResponse(reply=result.answer, escalation=result.escalation)


@app.get("/api/chat/history/{user_id}", response_model=ChatHistoryResponse)
async def chat_history(user_id: str):
    """
    Retrieve the recent conversation history for a user.
    Useful for rendering previous messages in the app's chat UI.
    """
    state = await app.state.pipeline.memory.get_state(user_id)
    return ChatHistoryResponse(
        display_name=state.get("display_name", ""),
        turns=state.get("recent_turns", []),
        summary=state.get("summary", ""),
    )


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "gohappy-support-bot"}


# ── App Engine warmup handler ────────────────────────────────────────────────
@app.get("/_ah/warmup")
async def warmup():
    """
    App Engine sends a GET /_ah/warmup request when spinning up a new instance.
    We just log it and return 200.
    """
    logger.info("Warmup request received.")
    return {"status": "ok", "warmup": True}


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


# ── Insights-driven KB update ────────────────────────────────────────────────
@app.post("/insights_update")
async def insights_update():
    """
    One-click KB update from the latest insights.

    Workflow:
      1. Read the last row from the "KB Insights" Google Sheet tab
      2. Feed the insight text to Gemini to update the master Knowledge Base
      3. Sync the updated KB to Vertex AI RAG Corpus

    This automates the manual /insights → read → /update_kb → /approve_kb flow.
    """
    pipeline = app.state.pipeline

    if not pipeline.kb_insights or not pipeline.kb_manager:
        raise HTTPException(
            status_code=500,
            detail="KB Insights or KB Manager not initialised.",
        )

    # ── Step 1: Fetch latest insight from Google Sheets ──────────────────
    try:
        insight_text = await pipeline.kb_insights.get_latest_insight()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to fetch latest insight: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read insights sheet: {e}")

    # ── Step 2: Use Gemini to apply the insight to the master KB ─────────
    try:
        update_instruction = (
            "Based on the following insights from analyzing recent customer conversations, "
            "update the knowledge base to address the identified gaps. "
            "Add any missing facts, policies, or instructions suggested below. "
            "Do NOT remove any existing content unless it directly contradicts the insight.\n\n"
            f"INSIGHTS:\n{insight_text}"
        )
        new_kb = await pipeline.kb_manager.generate_update(update_instruction)
    except Exception as e:
        logger.error("Gemini KB update generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate KB update: {e}")

    # ── Step 3: Save as pending and approve (sync to RAG) ────────────────
    try:
        await pipeline.kb_manager.save_pending_update(new_kb)
        success = await pipeline.kb_manager.approve_and_sync()
    except Exception as e:
        logger.error("RAG sync failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to sync to RAG: {e}")

    if not success:
        raise HTTPException(status_code=500, detail="RAG sync returned failure. Check server logs.")

    return {
        "status": "ok",
        "message": "Knowledge base updated from latest insights and synced to RAG.",
        "insight_length": len(insight_text),
        "new_kb_length": len(new_kb),
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
