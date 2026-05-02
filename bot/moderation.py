import os
import json
import logging
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

logger = logging.getLogger("gohappy.moderation")

MODERATION_PROMPT = """
You are an expert Indic-aware content moderator for a senior citizen community platform in India (audience: 50+ North Indian users).
Your task is to analyze user messages (which may contain Hindi, Hinglish, or English) and classify the intent, taking cultural nuances into account. 

For many 50+ North Indian users, mild profanity is conversational. Words like "bc", "mc", "bsdk", "chutiya", "yaar", "saala" have multiple spellings and are often used as conversational fillers without abusive intent.

CATEGORIES:
1. "conversational": The message contains filler profanity (e.g. "yaar bc class kab hai") but is NOT abusive or frustrated.
2. "frustration": The user is angry or frustrated about the service (e.g., "Mera link nahi chal raha bsdk", "tumhari service bekar hai").
3. "targeted_abuse": The user is explicitly abusing or attacking the bot, the platform, or individuals directly with severe malicious intent (e.g., "gand marao", "tum log chutiye ho", "teri maa ki...").
4. "none": Normal message with no profanity or frustration.

INSTRUCTIONS:
1. Classify the `severity` into one of the four exact strings above.
2. Strip out conversational filler profanity to produce a clean `stripped_text`. Do NOT rewrite the sentence, just remove the bad words. (e.g., "yaar bc kab aayega" -> "yaar kab aayega"). If there's no profanity, output the original text.
3. Provide a brief `intent_summary` (1 sentence).

OUTPUT FORMAT:
Provide your response strictly as a JSON object matching this schema, with no markdown fences:
{
  "severity": "none" | "conversational" | "frustration" | "targeted_abuse",
  "stripped_text": "<clean text>",
  "intent_summary": "<summary>"
}
"""

class HinglishModerator:
    def __init__(self):
        project  = os.environ["GCP_PROJECT_ID"]
        location = os.environ.get("GCP_LOCATION", "us-central1")
        
        # vertexai.init was called centrally or in other modules, but we ensure it here just in case.
        vertexai.init(project=project, location=location)

        # Using gemini-1.5-flash for low-latency moderation
        self.model = GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=MODERATION_PROMPT,
            generation_config=GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )

    async def analyze_message(self, text: str) -> dict:
        """
        Analyzes the text and returns a dict with 'severity', 'stripped_text', and 'intent_summary'.
        Defaults to safe values if the LLM call fails.
        """
        if not text or not text.strip():
            return {
                "severity": "none",
                "stripped_text": text,
                "intent_summary": "Empty message"
            }
            
        try:
            response = await self.model.generate_content_async(text)
            output = response.text.strip()
            
            if output.startswith("```json"):
                output = output[7:]
            if output.endswith("```"):
                output = output[:-3]
                
            result = json.loads(output)
            
            # Ensure valid severity
            valid_severities = {"none", "conversational", "frustration", "targeted_abuse"}
            if result.get("severity") not in valid_severities:
                result["severity"] = "none"
                
            return result
        except Exception as e:
            logger.error(f"Moderation failed: {e}")
            # Fallback to safe pass-through
            return {
                "severity": "none",
                "stripped_text": text,
                "intent_summary": "Moderation error fallback"
            }
