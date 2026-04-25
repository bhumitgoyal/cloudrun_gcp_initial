"""
bot/evaluator.py
Post-response quality audit using Gemini 1.5 Pro as a "grader".

Evaluates every bot response on:
  - accuracy_score   (0–100)
  - hallucination    (bool — did the bot fabricate info not in RAG chunks?)
  - should_escalate  (bool — should a human have handled this?)
  - empathy_score    (0–100 — senior-citizen friendliness)
  - reasoning        (short English explanation)

Runs as a fire-and-forget async task so it never delays the user reply.
"""

import os
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)

logger = logging.getLogger("gohappy.evaluator")

# ── Grader System Prompt ──────────────────────────────────────────────────────

GRADER_PROMPT = """
You are a Quality Auditor for GoHappy Club, a customer support chatbot serving
senior citizens aged 50+. Your job is to grade a bot-generated answer.

────────────────────────────────────────────────────────────
INPUTS YOU RECEIVE
────────────────────────────────────────────────────────────
1. ORIGINAL_USER_QUERY  — the raw message the user typed (may be Hinglish).
2. POLISHED_QUERY       — the cleaned English version used for retrieval.
3. RETRIEVED_CONTEXT    — the actual RAG knowledge chunks the bot had access to.
4. BOT_ANSWER           — the reply the bot sent to the user.

────────────────────────────────────────────────────────────
GRADING RUBRIC
────────────────────────────────────────────────────────────

1. accuracy_score (integer 0–100):
   - 90–100: Bot directly and completely answers the user's intent.
   - 70–89:  Mostly correct but missing a minor detail.
   - 50–69:  Partially correct; key information is missing or vague.
   - 0–49:   Wrong, misleading, or completely off-topic.

2. hallucination_check (boolean):
   - true  if the bot stated facts, prices, policies, or names NOT present
     in the RETRIEVED_CONTEXT. Minor phrasing differences are OK.
   - false if the answer is grounded in the retrieved context or the system prompt.

3. required_escalation (boolean):
   - true  if the query involves account-specific issues, payment disputes,
     complaints, frustration, or anything a human should handle.
   - false if the bot handled it appropriately on its own.

4. empathy_score (integer 0–100):
   - Measures warmth, patience, clarity, and appropriateness for a senior
     citizen audience. Penalise jargon, cold tone, overly long replies,
     or dismissive phrasing. Reward simple language and kind tone.

5. reasoning (string, 1–3 sentences):
   - Brief English explanation justifying the scores above.

────────────────────────────────────────────────────────────
OUTPUT FORMAT — STRICT JSON ONLY
────────────────────────────────────────────────────────────
Output a single valid JSON object. No markdown fences. No preamble.

{
  "accuracy_score": <int 0–100>,
  "hallucination_check": <true|false>,
  "required_escalation": <true|false>,
  "empathy_score": <int 0–100>,
  "reasoning": "<string>"
}
""".strip()


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    accuracy_score:      int
    hallucination_check: bool
    required_escalation: bool
    empathy_score:       int
    reasoning:           str


# ── OutputValidator ───────────────────────────────────────────────────────────

class OutputValidator:
    """
    Uses Gemini 1.5 Pro to grade bot responses.

    Required env vars:
        GCP_PROJECT_ID
        EVALUATOR_MODEL  (default: gemini-1.5-pro-002)
    """

    def __init__(self):
        project = os.environ["GCP_PROJECT_ID"]
        location = os.environ.get("GCP_LOCATION", "us-central1")
        vertexai.init(project=project, location=location)

        model_name = os.environ.get("EVALUATOR_MODEL", "gemini-1.5-pro-002")
        self.model = GenerativeModel(
            model_name=model_name,
            system_instruction=GRADER_PROMPT,
        )
        self.generation_config = GenerationConfig(
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "accuracy_score":      {"type": "INTEGER"},
                    "hallucination_check": {"type": "BOOLEAN"},
                    "required_escalation": {"type": "BOOLEAN"},
                    "empathy_score":       {"type": "INTEGER"},
                    "reasoning":           {"type": "STRING"},
                },
                "required": [
                    "accuracy_score",
                    "hallucination_check",
                    "required_escalation",
                    "empathy_score",
                    "reasoning",
                ],
            },
        )
        self.safety_settings = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,         threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,        threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,  threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,  threshold=HarmBlockThreshold.BLOCK_NONE),
        ]
        logger.info("OutputValidator ready (model=%s)", model_name)

    # ── Public API ────────────────────────────────────────────────────────────

    async def evaluate(
        self,
        original_query:    str,
        polished_query:    str,
        bot_answer:        str,
        retrieved_context: str = "",
    ) -> Optional[EvaluationResult]:
        """
        Grade a bot response. Returns EvaluationResult or None on failure.
        This method is designed to be called in a fire-and-forget task —
        it catches all exceptions internally and never raises.
        """
        prompt = self._build_grading_prompt(
            original_query, polished_query, bot_answer, retrieved_context
        )

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            raw_text = response.text.strip()
            logger.debug("Grader raw output: %s", raw_text[:300])
            return self._parse(raw_text)

        except Exception as exc:
            logger.error("Evaluation failed: %s", exc, exc_info=True)
            return None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_grading_prompt(
        self,
        original_query:    str,
        polished_query:    str,
        bot_answer:        str,
        retrieved_context: str,
    ) -> str:
        return f"""
ORIGINAL_USER_QUERY:
{original_query}

POLISHED_QUERY:
{polished_query}

RETRIEVED_CONTEXT:
{retrieved_context or "(no context retrieved)"}

BOT_ANSWER:
{bot_answer}
""".strip()

    @staticmethod
    def _repair_json(text: str) -> str:
        """Best-effort repair of truncated JSON from the grader.

        Common failures:
        - Unterminated string  → close the open quote
        - Missing closing braces/brackets → append them
        """
        # Close an unterminated string literal
        # Count unescaped quotes; if odd, the last string was cut off.
        quotes = re.findall(r'(?<!\\)"', text)
        if len(quotes) % 2 == 1:
            # Trim any trailing partial escape or whitespace, then close
            text = text.rstrip()
            text += '"'

        # Balance braces / brackets
        stack: list[str] = []
        match_map = {'{': '}', '[': ']'}
        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in match_map:
                stack.append(match_map[ch])
            elif ch in match_map.values() and stack and stack[-1] == ch:
                stack.pop()

        # Append missing closers in reverse order
        text += ''.join(reversed(stack))
        return text

    @staticmethod
    def _parse(raw: str) -> EvaluationResult:
        """Parse the strict JSON grading output from Gemini."""
        # Strip accidental markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt repair on truncated / malformed output
            logger.warning(
                "Grader returned malformed JSON, attempting repair. "
                "Raw (first 300 chars): %s",
                cleaned[:300],
            )
            repaired = OutputValidator._repair_json(cleaned)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                logger.error(
                    "JSON repair failed. Repaired text (first 300 chars): %s",
                    repaired[:300],
                )
                # Return a safe default so the pipeline doesn't lose the audit row
                return EvaluationResult(
                    accuracy_score=0,
                    hallucination_check=False,
                    required_escalation=False,
                    empathy_score=0,
                    reasoning="[auto] Grader returned unparseable JSON.",
                )

        return EvaluationResult(
            accuracy_score      = int(data.get("accuracy_score", 0)),
            hallucination_check = bool(data.get("hallucination_check", False)),
            required_escalation = bool(data.get("required_escalation", False)),
            empathy_score       = int(data.get("empathy_score", 0)),
            reasoning           = str(data.get("reasoning", "")),
        )
