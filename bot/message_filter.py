"""
bot/message_filter.py
Filters out non-actionable messages BEFORE they enter the pipeline.

Senior citizens frequently share:
  - Facebook / Instagram / YouTube / WhatsApp forward links
  - Random URLs from news sites or apps
  - Image-only messages (already handled by whatsapp.py — type != "text")
  - Emoji-only or single-character messages
  - "Good morning" forwards

This module ensures the bot only spends tokens on genuine questions.

Pipeline position:
  parse → FILTER → dedup → load state → rewrite → cache → RAG → Gemini → reply
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("gohappy.filter")


# ── URL / link patterns ──────────────────────────────────────────────────────

# Matches any http/https URL
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+",
    re.IGNORECASE,
)

# Common social media and forwarded link domains
_JUNK_DOMAINS = re.compile(
    r"(?:facebook\.com|fb\.com|fb\.watch|fbcdn\.net"
    r"|instagram\.com|instagr\.am"
    r"|youtube\.com|youtu\.be"
    r"|twitter\.com|x\.com"
    r"|tiktok\.com"
    r"|wa\.me|chat\.whatsapp\.com"
    r"|bit\.ly|goo\.gl|t\.co|tinyurl\.com|shorturl\.at"
    r"|linkedin\.com"
    r"|pinterest\.com"
    r"|dailymotion\.com"
    r"|reddit\.com"
    r"|telegram\.me|t\.me"
    r"|play\.google\.com|apps\.apple\.com)",
    re.IGNORECASE,
)

# Emoji-heavy patterns (messages that are mostly emojis with little text)
_EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F"   # emoticons
    r"\U0001F300-\U0001F5FF"    # symbols & pictographs
    r"\U0001F680-\U0001F6FF"    # transport & map symbols
    r"\U0001F900-\U0001F9FF"    # supplemental symbols
    r"\U0001FA00-\U0001FA6F"    # chess symbols
    r"\U0001FA70-\U0001FAFF"    # symbols extended
    r"\U00002702-\U000027B0"    # dingbats
    r"\U0000FE00-\U0000FE0F"    # variation selectors
    r"\U0000200D"               # zero width joiner
    r"\U00002640-\U00002642"    # gender symbols
    r"\U00002600-\U000026FF"    # misc symbols
    r"]+",
    re.UNICODE,
)


# ── Public API ────────────────────────────────────────────────────────────────

class MessageFilterResult:
    """Result of filtering a message."""
    __slots__ = ("is_actionable", "reason")

    def __init__(self, is_actionable: bool, reason: str = ""):
        self.is_actionable = is_actionable
        self.reason = reason

    def __repr__(self):
        if self.is_actionable:
            return "FilterResult(PASS)"
        return f"FilterResult(BLOCKED: {self.reason})"


def filter_message(text: str) -> MessageFilterResult:
    """
    Determine if a message text is worth processing through the pipeline.

    Returns:
        MessageFilterResult with is_actionable=True for genuine queries,
        is_actionable=False for junk (links, forwards, emoji-only, etc.)
    """
    if not text or not text.strip():
        return MessageFilterResult(False, "empty message")

    stripped = text.strip()

    # ── 1. Pure URL messages (just a link, nothing else) ─────────────────
    urls = _URL_PATTERN.findall(stripped)
    if urls:
        # Remove all URLs and see what's left
        text_without_urls = _URL_PATTERN.sub("", stripped).strip()

        # If nothing meaningful remains after removing URLs → junk
        # Allow through if there's a real question alongside the link
        remaining_alpha = re.sub(r"[^a-zA-Z\u0900-\u097F]", "", text_without_urls)
        if len(remaining_alpha) < 5:
            return MessageFilterResult(False, f"link-only message ({len(urls)} URL(s))")

        # Check if the URLs are from known junk domains
        for url in urls:
            if _JUNK_DOMAINS.search(url):
                # Still junk-domain link — but only block if no real question
                if len(remaining_alpha) < 15:
                    return MessageFilterResult(
                        False,
                        f"social media / forwarded link: {url[:60]}",
                    )

    # ── 2. Emoji-only or near-emoji messages ─────────────────────────────
    text_without_emoji = _EMOJI_PATTERN.sub("", stripped).strip()
    remaining_alpha = re.sub(r"[^a-zA-Z\u0900-\u097F]", "", text_without_emoji)
    if len(remaining_alpha) < 2:
        return MessageFilterResult(False, "emoji-only or non-text message")

    # ── 3. Single character or whitespace-only ───────────────────────────
    if len(stripped) < 2:
        return MessageFilterResult(False, "too short (single character)")

    # ── 4. "Good morning" type forwards (common among seniors) ───────────
    # Strip emojis and punctuation first, then check if only a greeting remains
    text_clean = _EMOJI_PATTERN.sub("", stripped).strip()
    text_clean = re.sub(r"[!.,;:*~\-_]+", "", text_clean).strip()

    _greeting_only = re.match(
        r"^(good\s*morning|good\s*evening|good\s*night|good\s*afternoon|gm|gn|"
        r"happy\s+\w+(\s+\w+)?\s*day|jai\s+shri\s+krishna|jai\s+mata\s+di|"
        r"namaste|namaskar|pranam|radhe\s*radhe|har\s+har\s+mahadev|"
        r"om\s+namah\s+shivaya|ram\s+ram|jai\s+hind|"
        r"shubh\s+prabhat|suprabhat)$",
        text_clean,
        re.IGNORECASE,
    )
    if _greeting_only:
        return MessageFilterResult(False, f"greeting-only: {stripped[:40]}")

    # ── Message passed all filters — it's actionable ─────────────────────
    return MessageFilterResult(True)
