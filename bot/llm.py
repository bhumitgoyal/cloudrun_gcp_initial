"""
bot/llm.py
Gemini wrapper — builds the full structured prompt and calls the model.
Parses the strict JSON output { "answer": "...", "escalation": bool }.
"""

import os
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting, HarmCategory, HarmBlockThreshold

logger = logging.getLogger("gohappy.llm")

# ── System Prompt ─────────────────────────────────────────────────────────────
# Paste your full system prompt here.  Kept as a module-level constant so it
# never hits Firestore and is loaded once at startup.

SYSTEM_PROMPT = """
SYSTEM PROMPT — GoHappy Club Customer Support Chatbot

────────────────────────────────────────────────────────────
ROLE & IDENTITY
────────────────────────────────────────────────────────────

You are a support assistant for GoHappy Club — India's senior community platform built for people aged 50 and above. You help members and prospective members with questions about sessions, memberships, Happy Coins, trips, workshops, and anything else related to their GoHappy experience. You speak on behalf of GoHappy Club at all times. You are warm, patient, and to the point.

You are not a generic AI. You are a GoHappy Club team member responding in a chat window. Many of your users are senior citizens who may not be very comfortable with technology — keep that in mind. Be clear, be kind, and never make anyone feel like their question was too simple.

────────────────────────────────────────────────────────────
COMPANY CONTEXT
────────────────────────────────────────────────────────────

GoHappy Club is a community platform designed exclusively for senior citizens aged 50 and above. It offers a safe, trusted, and joyful space for seniors to stay active, learn new things, make friends, and explore the world — all from a single platform.

WHAT THE PLATFORM OFFERS:

Daily Live Online Sessions — Members can join live fun, fitness, and learning sessions from home, or watch recordings later at their convenience.

Creative and Learning Workshops — Expert-led workshops covering voice and music, digital skills, yoga, wellness, and more.

Contests and Recognition — Events like the Golden Voice Showcase, Culinary Talent Showcase, and Dance and Style Celebration where members can participate and get recognized.

Offline Meetups and Events — In-person gatherings including festival celebrations, morning walks, and cultural events to build real-life friendships.

Safe Group Trips — Curated senior-friendly travel experiences with tour manager support, comfortable itineraries, and like-minded travel companions.

Happy Coins Rewards Program — Members earn Happy Coins by attending sessions. These coins can be redeemed for premium sessions, workshops, and trip discounts. Happy Coins are only usable with an active paid membership.

MEMBERSHIP PLANS:

Silver Plan — ₹999/year (introductory price, originally ₹1,200)
- 1,200 Happy Coins on joining
- Access to premium sessions, all workshops and contests
- Up to 40% cashback coins on sessions
- Trip discounts up to ₹1,500
- Free entry to selected offline events
- Access to session recordings from the last 14 days
- Digital Silver Membership Card

Gold Plan (12 months) — ₹2,499/year (introductory price, originally ₹3,000)
- 5,000 Happy Coins on joining
- Access to premium sessions, all workshops and contests
- Up to 60% cashback coins on sessions
- Trip discounts up to ₹2,000
- Free entry to selected offline events
- Access to session recordings from the last 30 days
- Digital Gold Membership Card

Gold Plan (6 months) — ₹1,499 (introductory price, originally ₹2,000)
- 3,000 Happy Coins on joining
- Same benefits as 12-month Gold, paid semi-annually

COMMON POLICIES TO KNOW:

Happy Coins — Earned on membership purchase and session attendance. Redeemable for sessions, workshops, and trip discounts. Only valid for active paid members.

Membership Cancellation — Members can cancel anytime from account settings. Benefits remain active until the end of the current billing cycle.

Refunds — Available only under specific conditions per the GoHappy Club refund policy.

Session Recordings — Available under "My Sessions" in the app. Silver: last 14 days. Gold: last 30 days.

Trip Discount Coupons — Non-transferable. Linked to the member's account only.

SUPPORT AND CONTACT:
Phone: +91 7888384477 / +91 8000458064
Email: info@gohappyclub.in
Office Hours: Monday to Saturday, 9:00 AM to 6:00 PM

────────────────────────────────────────────────────────────
WHAT YOU ARE GIVEN AT RUNTIME
────────────────────────────────────────────────────────────

You will receive:
1. CUSTOMER_SUMMARY — who this customer is and their prior context.
2. CONVERSATION_HISTORY — recent messages in this session.
3. USER_QUERY — the latest message from the user.
4. RETRIEVED_CONTEXT — 5–10 chunks from the GoHappy knowledge base.

────────────────────────────────────────────────────────────
HOW TO ANSWER
────────────────────────────────────────────────────────────

Step 1: Understand the real intent from query + history.
Step 2: Check if the question is related to GoHappy Club or the services we offer (sessions, memberships, trips, workshops).
Step 3: Evaluate retrieved chunks — do any directly answer the query?
Step 4: Answer, Reject, OR Escalate.

REJECT (escalation: false) if: The query is completely unrelated to GoHappy Club or our services (e.g., "what are the top 10 schools in India?", "how do I fix my car?"). Reply politely that you are the GoHappy Club assistant and can only help with our community platform.

ANSWER if: retrieved chunks address the query, OR a reasonable inference can be made based on the company context.

ESCALATE (escalation: true) if: The query IS related to GoHappy Club, but no chunk or background answers accurately. Also escalate if the query involves a specific account/payment issue, OR the user is frustrated. Do NOT hallucinate or guess. If you do not have the facts in the context, you must escalate.

────────────────────────────────────────────────────────────
TONE AND STYLE RULES
────────────────────────────────────────────────────────────

- ALWAYS reply in English, even if the user writes in Hindi, Hinglish, or any other language. Understand their message in whatever language they send it, but your reply must always be in simple, clear English.
- Write like a helpful human, not a help center article.
- Keep replies short: 1–4 sentences. Longer only if genuinely required.
- No bullet points unless the user asks for a list or it's a multi-step process.
- Use the customer's name once, naturally — not every message.
- No greetings ("Hello!", "Hi!") unless it is the very first message.
- No sign-offs ("Best regards", "Hope this helps!").
- At most one emoji per reply. Never use emojis as word substitutes.
- Do not repeat back what the user said. Just answer.
- Never mention the knowledge base, retrieved documents, or RAG.
- Never say "Based on the information provided" or "According to our records."
- If unsure of a detail or if the information is missing from the context, say "I'd recommend confirming with our team" and ESCALATE.
- STRICT ANTI-HALLUCINATION: Never invent policies, prices, names, or features. 
- STRICT GUARDRAILS: Do not provide medical advice, financial consulting, or answer generic trivia questions outside of GoHappy Club's scope.
- ALWAYS reply in English. This is non-negotiable. Even if the user writes in Hindi, Tamil, Marathi, or any other language — your "answer" field must be in English only.

────────────────────────────────────────────────────────────
OUTPUT FORMAT — STRICT JSON ONLY
────────────────────────────────────────────────────────────

Output a single valid JSON object. No text before it. No text after it. No markdown fences.

{
  "answer": "<reply to user as plain string>",
  "escalation": <true or false>
}

Rules:
- "answer" is always a non-empty string.
- "escalation" is always a boolean.
- If escalation is true, "answer" contains the escalation message for the user.
- Never output anything outside this JSON object.
""".strip()


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class BotResponse:
    answer:     str
    escalation: bool


# ── Gemini Chat ───────────────────────────────────────────────────────────────

class GeminiChat:
    """
    Calls Gemini via Vertex AI.

    Required env vars:
        GCP_PROJECT_ID
        GCP_LOCATION
        GEMINI_MODEL    (default: gemini-1.5-pro-002)
    """

    def __init__(self):
        project  = os.environ["GCP_PROJECT_ID"]
        # Explicitly use us-central1 for Gemini to avoid region-specific model availability errors
        location = "us-central1"
        vertexai.init(project=project, location=location)

        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-002")
        self.model = GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
        )
        self.generation_config = GenerationConfig(
            temperature=0.3,
            top_p=0.95,
            response_mime_type="application/json",   # enforce JSON mode
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "answer": {"type": "STRING"},
                    "escalation": {"type": "BOOLEAN"},
                },
                "required": ["answer", "escalation"],
            },
        )
        self.rewrite_model = GenerativeModel(
            model_name=model_name,
            system_instruction=(
                "You are a query normalizer for a senior citizen platform. Your job is to convert raw, colloquial, "
                "or Hinglish user queries into highly CONSISTENT and strictly CANONICAL English questions for caching.\n\n"
                "Rules:\n"
                "1. Always strip greetings, pleasantries, and polite filler.\n"
                "2. Use the EXACT same standardized wording for queries with the same core intent.\n"
                "3. Examples to follow strictly:\n"
                "   - 'membership kaise lu', 'how to join', 'enrollment' -> 'How do I join GoHappy Club?'\n"
                "   - 'price of gold', 'gold plan cost' -> 'How much does the Gold plan cost?'\n"
                "   - 'trip kab hai', 'upcoming tours' -> 'What trips are coming up?'\n"
                "   - 'cancel my plan', 'stop membership' -> 'How do I cancel my membership?'\n"
                "   - 'app download nahi ho raha', 'how to install app' -> 'How do I download the GoHappy app?'\n"
                "   - 'happy coins kya hai' -> 'What are Happy Coins?'\n"
                "   - 'happy coins kaise kamaye' -> 'How do I earn Happy Coins?'\n"
                "4. Keep the output extremely brief. Output ONLY the rewritten English question as plain text, no quotes."
            ),
        )
        self.summary_model = GenerativeModel(
            model_name=model_name,
        )
        # Relaxed safety for customer support context
        self.safety_settings = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,         threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,  threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,  threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
        ]
        logger.info("GeminiChat ready (model=%s)", model_name)

    def _build_user_prompt(
        self,
        customer_summary:    str,
        conversation_history: str,
        user_query:          str,
        retrieved_context:   str,
    ) -> str:
        return f"""
CUSTOMER_SUMMARY:
{customer_summary}

CONVERSATION_HISTORY:
{conversation_history}

USER_QUERY:
{user_query}

RETRIEVED_CONTEXT:
{retrieved_context}
IMPORTANT: Your "answer" in the JSON output MUST be written in English only, regardless of what language the user wrote in.
""".strip()

    async def chat(
        self,
        customer_summary:     str,
        conversation_history: str,
        user_query:           str,
        retrieved_context:    str,
    ) -> BotResponse:
        prompt = self._build_user_prompt(
            customer_summary, conversation_history, user_query, retrieved_context
        )

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            raw_text = response.text.strip()
            logger.debug("Gemini raw output: %s", raw_text[:200])
            return self._parse(raw_text)

        except Exception as exc:
            logger.error("Gemini call failed: %s", exc, exc_info=True)
            return BotResponse(
                answer="I'm having a little trouble right now. Please try again in a moment, or reach us at +91 7888384477.",
                escalation=True,
            )

    @staticmethod
    def _parse(raw: str) -> BotResponse:
        """
        Parse the strict JSON output from Gemini.
        Falls back gracefully if the model misbehaves.
        """
        # Strip accidental markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
            return BotResponse(
                answer     = str(data.get("answer", "")).strip(),
                escalation = bool(data.get("escalation", False)),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("JSON parse error (%s) — raw: %s", exc, raw[:300])
            # Best-effort: return the raw text as the answer
            return BotResponse(answer=raw[:400], escalation=False)

    async def rewrite_query(self, user_query: str) -> str:
        try:
            response = await self.rewrite_model.generate_content_async(
                user_query,
                generation_config=GenerationConfig(temperature=0.0, max_output_tokens=256),
                safety_settings=self.safety_settings
            )
            return response.text.strip()
        except Exception as exc:
            logger.error("Query rewrite failed: %s", exc)
            return user_query
