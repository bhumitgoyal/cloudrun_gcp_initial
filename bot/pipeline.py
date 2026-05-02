"""
bot/pipeline.py
Orchestrates the full message handling flow:
  WhatsApp in → Filter → Dedup → Firestore → Cache? → RAG → Gemini → Cache store → Firestore → WhatsApp out

Also handles:
  - Message filtering (blocks links, social media forwards, emoji-only, greetings)
  - Deduplication (ignores repeated webhook deliveries for same message ID)
  - Rolling summary compression via Gemini when turn threshold is hit
  - Escalation flag — logs and (optionally) routes to a human queue
"""

import os
import logging
import asyncio
from typing import Optional

from bot.whatsapp        import WhatsAppClient, IncomingMessage
from bot.rag             import RAGEngine
from bot.memory          import ConversationMemory
from bot.llm             import GeminiChat, BotResponse
from bot.moderation      import HinglishModerator
from bot.rag_cache       import RAGCache
from bot.message_filter  import filter_message
from bot.evaluator       import OutputValidator
from bot.sheets_logger   import SheetsAuditLogger
from bot.kb_insights     import KBInsightsGenerator
from bot.kb_manager      import KnowledgeBaseManager

logger = logging.getLogger("gohappy.pipeline")

# Simple in-process dedup cache (last 500 message IDs)
_SEEN_IDS: set = set()
_SEEN_MAX  = 500


class MessagePipeline:

    def __init__(
        self,
        whatsapp:      WhatsAppClient,
        rag:           RAGEngine,
        memory:        ConversationMemory,
        llm:           GeminiChat,
        cache:         RAGCache,
        evaluator:     OutputValidator  = None,
        sheets_logger: SheetsAuditLogger = None,
        kb_insights:   KBInsightsGenerator = None,
        kb_manager:    KnowledgeBaseManager = None,
    ):
        self.wa            = whatsapp
        self.rag           = rag
        self.memory        = memory
        self.llm           = llm
        self.cache         = cache
        self.evaluator     = evaluator
        self.sheets_logger = sheets_logger
        self.kb_insights   = kb_insights
        self.kb_manager    = kb_manager
        self.moderator     = HinglishModerator()

    # ── Entry point called by FastAPI background task ─────────────────────────

    async def handle(self, payload: dict):
        """
        Full pipeline for one incoming webhook payload.
        Any uncaught exception is logged but not re-raised (background task).
        """
        try:
            await self._process(payload)
        except Exception as exc:
            logger.error("Unhandled pipeline error: %s", exc, exc_info=True)

    # ── Main flow ─────────────────────────────────────────────────────────────

    async def _process(self, payload: dict):
        # 1. Parse
        msg: Optional[IncomingMessage] = self.wa.parse_message(payload)
        if msg is None:
            return   # Not an actionable message (status update, etc.)

        admin_phone = os.environ.get("ADMIN_PHONE_NUMBER")

        # 1.5 Admin Override & Insights Logic
        is_hardcoded_admin = msg.from_number in ("919818646823", "+919818646823")
        
        if is_hardcoded_admin and msg.text.strip().lower() == "insights":
            await self.wa.send_text(
                to=msg.from_number,
                body="⏳ Generating KB Insights... This may take a minute.",
                phone_number_id=msg.phone_number_id
            )
            asyncio.create_task(self._run_insights_generator(msg))
            return
            
        if admin_phone and msg.from_number == admin_phone:
            if msg.text.startswith("/resolve "):
                target_phone = msg.text.split(" ")[1].strip()
                await self.memory.set_escalation_status(target_phone, False)
                await self.wa.send_text(
                    to=admin_phone, 
                    body=f"✅ Resolved escalation for {target_phone}. Bot is back in control.",
                    phone_number_id=msg.phone_number_id
                )
            elif msg.text.startswith("/update_kb "):
                await self.wa.send_text(
                    to=admin_phone, 
                    body="⏳ Processing your KB update request with Gemini... Please wait.",
                    phone_number_id=msg.phone_number_id
                )
                asyncio.create_task(self._handle_update_kb(msg))
            elif msg.text.strip().lower() == "/approve_kb":
                await self.wa.send_text(
                    to=admin_phone, 
                    body="⏳ Approving update and syncing to Vertex AI... Please wait.",
                    phone_number_id=msg.phone_number_id
                )
                asyncio.create_task(self._handle_approve_kb(msg))
            else:
                await self.wa.send_text(
                    to=admin_phone,
                    body="Hello Admin! The bot ignores normal messages from you. Use `/resolve <PHONE_NUMBER>` to unpause a user.",
                    phone_number_id=msg.phone_number_id
                )
            return

        # 2. Deduplicate
        if msg.wa_message_id in _SEEN_IDS:
            logger.info("Duplicate message %s — skipping.", msg.wa_message_id)
            return
        _SEEN_IDS.add(msg.wa_message_id)
        if len(_SEEN_IDS) > _SEEN_MAX:
            _SEEN_IDS.pop()

        # Mark message as read
        asyncio.create_task(self.wa.mark_as_read(msg.wa_message_id, msg.phone_number_id))

        # 2.5 Filter out junk messages (links, forwards, emojis, greetings)
        filter_result = filter_message(msg.text)
        if not filter_result.is_actionable:
            logger.info(
                "Filtered out message from %s: %s — %.80s",
                msg.from_number, filter_result.reason, msg.text,
            )
            return

        logger.info("Processing message from %s: %.80s", msg.from_number, msg.text)
        
        # Show typing indicator to user
        asyncio.create_task(self.wa.send_typing_indicator(msg.from_number, msg.phone_number_id))

        # 3. Load conversation state from Firestore
        state = await self.memory.get_state(msg.from_number)

        if state.get("escalated_to_human", False):
            logger.info("Ignoring message from %s, currently escalated to human.", msg.from_number)
            return

        # 3.4 Indic-aware Input Moderation
        mod_result = await self.moderator.analyze_message(msg.text)
        logger.info("Moderation input result: %s", mod_result)
        
        severity = mod_result.get("severity", "none")
        if severity == "targeted_abuse":
            bot_response = BotResponse(
                answer="Your message violates our community guidelines. An admin will review this conversation.",
                escalation=True
            )
            await self.wa.send_text(to=msg.from_number, body=bot_response.answer, phone_number_id=msg.phone_number_id)
            await self._handle_escalation(msg, bot_response, state)
            return
            
        is_frustrated = (severity == "frustration")
        if severity in ["conversational", "frustration"]:
            # Override text to remove filler profanity so RAG/LLM isn't distracted
            msg.text = mod_result.get("stripped_text", msg.text)

        # 3.5 Polish query for better RAG retrieval
        polished_query = await self.llm.rewrite_query(msg.text)
        if not polished_query:
            polished_query = msg.text # Fallback to original text if the LLM completely stripped a greeting
        logger.info("Polished query: %s", polished_query)

        # ── 3.7 Semantic cache check ─────────────────────────────────────────
        # cached = await self.cache.get(polished_query)
        # if cached is not None:
        #     bot_response = BotResponse(
        #         answer     = cached["answer"],
        #         escalation = cached.get("escalation", False),
        #     )
        #     logger.info("Serving cached response for: %.80s", polished_query)
        # else:
        # 4. RAG retrieval (run in thread pool — blocking SDK call)
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, self.rag.query, polished_query
        )
        retrieved_context = self.rag.format_for_prompt(chunks)

        # 5. Build prompt components
        customer_summary    = self.memory.build_customer_summary(state)
        conversation_history = self.memory.format_history_for_prompt(state)

        # 6. Call Gemini
        bot_response: BotResponse = await self.llm.chat(
            customer_summary     = customer_summary,
            conversation_history = conversation_history,
            user_query           = msg.text,
            retrieved_context    = retrieved_context,
            is_frustrated        = is_frustrated,
        )

        # 6.2 Output Moderation (RAG leakage protection)
        out_mod_result = await self.moderator.analyze_message(bot_response.answer)
        out_severity = out_mod_result.get("severity", "none")
        if out_severity in ["targeted_abuse", "frustration"]:
            logger.warning("Blocked unsafe bot output: %s", bot_response.answer)
            bot_response.answer = "I apologize, but I am unable to process this request at the moment. Please let me know if you need help with anything else."


        # 6.5 Cache the response (skips escalation responses automatically)
        # await self.cache.set(polished_query, {
        #     "answer":     bot_response.answer,
        #     "escalation": bot_response.escalation,
        # })

        # 7. Send reply to user
        sent = await self.wa.send_text(
            to             = msg.from_number,
            body           = bot_response.answer,
            phone_number_id= msg.phone_number_id,
        )
        if not sent:
            logger.error("Failed to deliver message to %s", msg.from_number)

        # 7.5 Quality audit (async, doesn't block the reply)
        if self.evaluator and self.sheets_logger:
            asyncio.create_task(
                self._evaluate_and_log(
                    msg, polished_query, bot_response, retrieved_context
                )
            )

        # 8. Persist turn to Firestore
        updated_state = await self.memory.append_turn(
            phone        = msg.from_number,
            display_name = msg.display_name,
            user_text    = msg.text,
            bot_text     = bot_response.answer,
        )

        # 9. Escalation handling
        if bot_response.escalation:
            await self._handle_escalation(msg, bot_response, state)

        # 10. Rolling summary compression (async, doesn't block the reply)
        if self.memory.should_summarise(updated_state):
            asyncio.create_task(
                self._compress_summary(msg.from_number, updated_state)
            )

    # ── Escalation ────────────────────────────────────────────────────────────

    async def _handle_escalation(
        self,
        msg:          IncomingMessage,
        bot_response: BotResponse,
        state:        dict,
    ):
        """
        Called when the bot flags a conversation for human review.
        """
        logger.warning(
            "ESCALATION REQUIRED | phone=%s | name=%s | query=%s | turn=%d",
            msg.from_number,
            msg.display_name,
            msg.text[:120],
            state.get("turn_count", 0),
        )
        
        # 1. Pause the bot for this user
        await self.memory.set_escalation_status(msg.from_number, True)

        # 2. Alert the Admin via WhatsApp
        admin_phone = os.environ.get("ADMIN_PHONE_NUMBER")
        if admin_phone:
            clean_bot_phone = "".join(filter(str.isdigit, msg.bot_phone_number))
            resolve_link = f"https://wa.me/{clean_bot_phone}?text=%2Fresolve%20{msg.from_number}"

            alert_text = (
                f"🚨 *ESCALATION REQUIRED*\n"
                f"User: {msg.display_name} ({msg.from_number})\n\n"
                f"Issue: {msg.text}\n\n"
                f"Bot replied: {bot_response.answer}\n\n"
                f"Reply to them directly from your WhatsApp App. When finished, click the link below to unpause the bot for this user:\n"
                f"{resolve_link}"
            )
            await self.wa.send_text(
                to=admin_phone,
                body=alert_text,
                phone_number_id=msg.phone_number_id
            )

    # ── Rolling summary compression ───────────────────────────────────────────

    async def _compress_summary(self, phone: str, state: dict):
        """
        Ask Gemini to compress recent turns into a tighter rolling summary.
        This keeps the CUSTOMER_SUMMARY block concise as conversations grow.
        """
        history_text = self.memory.format_history_for_prompt(state)
        existing     = state.get("summary", "")

        compression_prompt = f"""
You are summarising a customer service conversation for GoHappy Club.

EXISTING SUMMARY (may be empty):
{existing or "(none)"}

RECENT CONVERSATION:
{history_text}

Write a new, concise summary (3–5 sentences max) covering:
- Who the customer is and what they care about
- Any unresolved issues or follow-up items
- Key facts established (membership plan, preferences, etc.)

Output only the summary text. No preamble. No bullet points.
""".strip()

        try:
            response = await self.llm.summary_model.generate_content_async(compression_prompt)
            new_summary = response.text.strip()
            await self.memory.update_summary(phone, new_summary)
            logger.info("Summary compressed for %s (%d chars)", phone, len(new_summary))
        except Exception as exc:
            logger.error("Summary compression failed for %s: %s", phone, exc)

    # ── Quality audit ────────────────────────────────────────────────────────
    
    async def _evaluate_and_log(
        self,
        msg:               IncomingMessage,
        polished_query:    str,
        bot_response:      BotResponse,
        retrieved_context: str,
    ):
        """
        Runs the Gemini grader on the bot's response and appends to Google Sheets.
        This entire task is fire-and-forget.
        """
        import datetime
        
        logger.info("Starting background quality audit for %s...", msg.from_number)
        
        # 1. Run LLM evaluation
        result = await self.evaluator.evaluate(
            original_query=msg.text,
            polished_query=polished_query,
            bot_answer=bot_response.answer,
            retrieved_context=retrieved_context,
        )
        
        if not result:
            return # Grader failed, nothing to log

        # 2. Append to Sheets
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        await self.sheets_logger.log_to_audit_sheet(
            timestamp=timestamp,
            phone=msg.from_number,
            original_query=msg.text,
            polished_query=polished_query,
            bot_answer=bot_response.answer,
            accuracy=result.accuracy_score,
            hallucination=result.hallucination_check,
            escalation=result.required_escalation,
            empathy=result.empathy_score,
            reasoning=result.reasoning,
            message_id=msg.wa_message_id,
        )

    # ── KB Insights Trigger ──────────────────────────────────────────────────
    async def _run_insights_generator(self, msg: IncomingMessage):
        """
        Background task to generate KB Insights.
        """
        if self.kb_insights:
            result_text = await self.kb_insights.generate_insights()
            await self.wa.send_text(
                to=msg.from_number,
                body=result_text,
                phone_number_id=msg.phone_number_id
            )
        else:
            logger.error("KBInsightsGenerator not initialized.")

    # ── KB Automation Handlers ───────────────────────────────────────────────
    async def _handle_update_kb(self, msg: IncomingMessage):
        if not self.kb_manager:
            logger.error("KnowledgeBaseManager not initialized.")
            return

        admin_input = msg.text[len("/update_kb "):].strip()
        try:
            new_kb = await self.kb_manager.generate_update(admin_input)
            await self.kb_manager.save_pending_update(new_kb)
            
            # Save file locally to upload to whatsapp
            import os
            file_path = "/tmp/GoHappyClub_KnowledgeBase.md"
            with open(file_path, "w") as f:
                f.write(new_kb)

            # Upload to WhatsApp Media
            media_id = await self.wa.upload_media(file_path, mime_type="text/plain")
            if media_id:
                await self.wa.send_document(
                    to=msg.from_number,
                    media_id=media_id,
                    filename="Updated_KB_Draft.md",
                    caption="📄 Here is the updated Knowledge Base draft. Please review it. Reply with `/approve_kb` to confirm and sync to Vertex AI, or send another `/update_kb` command to make further changes."
                )
            else:
                await self.wa.send_text(
                    to=msg.from_number,
                    body="❌ Failed to upload the document to WhatsApp. Check server logs.",
                    phone_number_id=msg.phone_number_id
                )
        except Exception as exc:
            logger.error("Error in KB update: %s", exc, exc_info=True)
            await self.wa.send_text(
                to=msg.from_number,
                body=f"❌ Failed to process KB update: {exc}",
                phone_number_id=msg.phone_number_id
            )

    async def _handle_approve_kb(self, msg: IncomingMessage):
        if not self.kb_manager:
            logger.error("KnowledgeBaseManager not initialized.")
            return
            
        success = await self.kb_manager.approve_and_sync()
        if success:
            await self.wa.send_text(
                to=msg.from_number,
                body="✅ Successfully approved! The new Knowledge Base has been synced to Vertex AI RAG Corpus.",
                phone_number_id=msg.phone_number_id
            )
        else:
            await self.wa.send_text(
                to=msg.from_number,
                body="❌ Failed to sync to Vertex AI RAG Corpus. Please check the server logs.",
                phone_number_id=msg.phone_number_id
            )

