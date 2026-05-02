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
    bot_phone_number: str = "" # the business phone number receiving the message


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
                wa_message_id    = message["id"],
                from_number      = message["from"],
                display_name     = contacts[0].get("profile", {}).get("name", "Member"),
                phone_number_id  = value["metadata"]["phone_number_id"],
                text             = message["text"]["body"].strip(),
                bot_phone_number = value["metadata"].get("display_phone_number", ""),
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

    async def mark_as_read(self, message_id: str, phone_number_id: Optional[str] = None) -> bool:
        """Mark an incoming message as read."""
        pid = phone_number_id or self.phone_number_id
        url = f"{GRAPH_API_BASE}/{pid}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type":  "application/json",
        }
        data = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        try:
            resp = await self._client.post(url, headers=headers, json=data)
            resp.raise_for_status()
            logger.debug("Marked message %s as read.", message_id)
            return True
        except Exception as exc:
            logger.warning("Failed to mark message as read: %s", exc)
            return False

    async def send_typing_indicator(self, to: str, phone_number_id: Optional[str] = None) -> bool:
        """Show a typing indicator to the user."""
        pid = phone_number_id or self.phone_number_id
        url = f"{GRAPH_API_BASE}/{pid}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type":  "application/json",
        }
        data = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "sender_action": "typing_on"
        }
        try:
            resp = await self._client.post(url, headers=headers, json=data)
            resp.raise_for_status()
            logger.debug("Sent typing indicator to %s.", to)
            return True
        except Exception as exc:
            logger.warning("Failed to send typing indicator: %s", exc)
            return False

    async def upload_media(self, file_path: str, mime_type: str = "text/plain") -> Optional[str]:
        """Uploads a file to WhatsApp Media API and returns the media ID."""
        url = f"{GRAPH_API_BASE}/{self.phone_number_id}/media"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, "rb") as f:
                files = {
                    "file": (filename, f, mime_type)
                }
                data = {
                    "messaging_product": "whatsapp",
                    "type": mime_type # Wait, WhatsApp expects type in data? Let me double check... no, type is not needed in data, maybe just messaging_product.
                }
                # To be safe, Meta API expects files in multipart/form-data
                resp = await self._client.post(url, headers=headers, data={"messaging_product": "whatsapp"}, files=files)
                resp.raise_for_status()
                media_id = resp.json().get("id")
                logger.info("Uploaded media %s -> id: %s", filename, media_id)
                return media_id
        except httpx.HTTPStatusError as exc:
            logger.error("WhatsApp media upload failed [%s]: %s", exc.response.status_code, exc.response.text)
            return None
        except Exception as exc:
            logger.error("WhatsApp media upload error: %s", exc)
            return None

    async def send_document(self, to: str, media_id: str, filename: str, caption: str = "") -> bool:
        """Sends an uploaded document to the user."""
        url = f"{GRAPH_API_BASE}/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type":  "application/json",
        }
        data = {
            "messaging_product": "whatsapp",
            "recipient_type":    "individual",
            "to":                to,
            "type":              "document",
            "document": {
                "id": media_id,
                "filename": filename,
                "caption": caption
            }
        }
        try:
            resp = await self._client.post(url, headers=headers, json=data)
            resp.raise_for_status()
            logger.info("Document sent to %s (status %s)", to, resp.status_code)
            return True
        except httpx.HTTPStatusError as exc:
            logger.error("WhatsApp document send failed [%s]: %s", exc.response.status_code, exc.response.text)
            return False
        except Exception as exc:
            logger.error("WhatsApp document send error: %s", exc)
            return False

    async def close(self):
        await self._client.aclose()
