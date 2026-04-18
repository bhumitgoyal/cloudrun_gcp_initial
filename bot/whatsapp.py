"""
bot/whatsapp.py
Handles all communication with the WhatsApp Cloud API (Meta Graph API).
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger("gohappy.whatsapp")

GRAPH_API_BASE = "https://graph.facebook.com/v19.0"


@dataclass
class IncomingMessage:
    wa_message_id: str       # WhatsApp message ID (for deduplication)
    from_number: str         # sender's phone number (E.164, no +)
    display_name: str        # sender's profile name
    phone_number_id: str     # the receiving business phone number ID
    text: str                # message body


class WhatsAppClient:
    """
    Wraps the WhatsApp Cloud API.
    Reads credentials from environment variables at construction time.
    """

    def __init__(self):
        self.access_token     = os.environ["WHATSAPP_ACCESS_TOKEN"]
        self.phone_number_id  = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
        self._client          = httpx.AsyncClient(timeout=15.0)

    # ── Parse incoming webhook payload ────────────────────────────────────────
    def parse_message(self, payload: dict) -> Optional[IncomingMessage]:
        """
        Extract the relevant fields from a raw Meta webhook payload.
        Returns None if the payload contains no actionable text message
        (e.g. status updates, reactions, etc.).
        """
        try:
            entry   = payload["entry"][0]
            changes = entry["changes"][0]
            value   = changes["value"]

            # Skip status update payloads
            if "messages" not in value:
                return None

            message  = value["messages"][0]
            contacts = value.get("contacts", [{}])

            # We only handle plain text messages for now
            if message.get("type") != "text":
                logger.info("Non-text message received (type=%s) — skipping.", message.get("type"))
                return None

            return IncomingMessage(
                wa_message_id   = message["id"],
                from_number     = message["from"],
                display_name    = contacts[0].get("profile", {}).get("name", "Member"),
                phone_number_id = value["metadata"]["phone_number_id"],
                text            = message["text"]["body"].strip(),
            )

        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("Could not parse webhook payload: %s", exc)
            return None

    # ── Send a text reply ─────────────────────────────────────────────────────
    async def send_text(self, to: str, body: str, phone_number_id: Optional[str] = None) -> bool:
        """
        Send a plain-text WhatsApp message.
        `to` is the recipient's phone number in E.164 format (no leading +).
        """
        pid = phone_number_id or self.phone_number_id
        url = f"{GRAPH_API_BASE}/{pid}/messages"

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type":  "application/json",
        }
        data = {
            "messaging_product": "whatsapp",
            "recipient_type":    "individual",
            "to":                to,
            "type":              "text",
            "text":              {"body": body},
        }

        try:
            resp = await self._client.post(url, headers=headers, json=data)
            resp.raise_for_status()
            logger.info("Message sent to %s (status %s)", to, resp.status_code)
            return True
        except httpx.HTTPStatusError as exc:
            logger.error(
                "WhatsApp send failed [%s]: %s",
                exc.response.status_code,
                exc.response.text,
            )
            return False
        except httpx.RequestError as exc:
            logger.error("WhatsApp request error: %s", exc)
            return False

    async def close(self):
        await self._client.aclose()
