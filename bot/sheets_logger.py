"""
bot/sheets_logger.py
Logs evaluation audit results to a Google Spreadsheet.

Uses gspread with Application Default Credentials (same service account
used by Firestore and Vertex AI — zero extra auth configuration).

Features:
  - Auto-creates the spreadsheet if AUDIT_SPREADSHEET_ID is not set
  - Lazy initialisation (gspread client created on first call)
  - Fail-safe: if the Sheets API fails, the user still gets their message
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("gohappy.sheets")


class SheetsAuditLogger:
    """
    Appends evaluation rows to a Google Spreadsheet.

    Env vars:
        AUDIT_SPREADSHEET_ID  — ID of an existing spreadsheet (optional).
                                If empty, a new spreadsheet is auto-created
                                on first use and the ID is logged.
    """

    HEADER_ROW = [
        "Timestamp",
        "User Phone",
        "Original Query",
        "Polished Query",
        "Bot Answer",
        "Accuracy (%)",
        "Hallucination?",
        "Should Escalate?",
        "Empathy Score",
        "Reasoning",
        "Message ID",
    ]

    def __init__(self):
        self._spreadsheet_id: Optional[str] = os.environ.get("AUDIT_SPREADSHEET_ID", "").strip() or None
        self._client = None
        self._sheet = None
        self._initialised = False

    # ── Lazy initialisation ──────────────────────────────────────────────────

    def _ensure_initialised(self):
        """
        Connects to Google Sheets on first call.
        Uses Application Default Credentials (ADC), which on App Engine
        maps to the service account automatically.
        """
        if self._initialised:
            return

        import gspread

        # 1. Prioritize a local secrets.json file if one exists
        if os.path.exists("secrets.json"):
            logger.info("Using local secrets.json for Google Sheets authentication.")
            self._client = gspread.service_account(filename="secrets.json")
        else:
            # 2. Fall back to Application Default Credentials (e.g. on App Engine)
            from google.auth import default
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds, project = default(scopes=scopes)
            self._client = gspread.authorize(creds)

        if self._spreadsheet_id:
            # Open existing spreadsheet
            spreadsheet = self._client.open_by_key(self._spreadsheet_id)
            logger.info("Opened existing audit spreadsheet: %s", self._spreadsheet_id)
        else:
            # Auto-create a new spreadsheet
            title = f"GoHappy Audit Log — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            spreadsheet = self._client.create(title)
            self._spreadsheet_id = spreadsheet.id
            logger.info(
                "Auto-created audit spreadsheet: %s (ID: %s). "
                "Set AUDIT_SPREADSHEET_ID=%s in .env to reuse it.",
                title,
                self._spreadsheet_id,
                self._spreadsheet_id,
            )

            # Share with the project owner if an admin email is available
            admin_email = os.environ.get("AUDIT_SHARE_EMAIL", "").strip()
            if admin_email:
                spreadsheet.share(admin_email, perm_type="user", role="writer")
                logger.info("Shared audit spreadsheet with %s", admin_email)

        self._sheet = spreadsheet.sheet1

        # Write header row if the sheet is empty
        try:
            existing = self._sheet.row_values(1)
            if not existing or existing[0] != self.HEADER_ROW[0]:
                self._sheet.insert_row(self.HEADER_ROW, index=1)
                logger.info("Wrote header row to audit sheet.")
        except Exception:
            # Sheet is brand new / empty
            self._sheet.insert_row(self.HEADER_ROW, index=1)
            logger.info("Wrote header row to audit sheet.")

        self._initialised = True

    # ── Public API ───────────────────────────────────────────────────────────

    async def log_to_audit_sheet(
        self,
        timestamp:     str,
        phone:         str,
        original_query: str,
        polished_query: str,
        bot_answer:    str,
        accuracy:      int,
        hallucination: bool,
        escalation:    bool,
        empathy:       int,
        reasoning:     str,
        message_id:    str,
    ) -> bool:
        """
        Append one audit row. Returns True on success.
        Completely fail-safe — exceptions are caught and logged.
        """
        try:
            self._ensure_initialised()

            row = [
                timestamp,
                phone,
                original_query[:500],      # Truncate to avoid sheet cell limits
                polished_query[:500],
                bot_answer[:1000],
                accuracy,
                str(hallucination),
                str(escalation),
                empathy,
                reasoning[:500],
                message_id,
            ]

            self._sheet.append_row(row, value_input_option="USER_ENTERED")
            logger.info("Audit row logged for %s (accuracy=%d%%)", phone, accuracy)
            return True

        except Exception as exc:
            logger.error(
                "Failed to log audit row for %s: %s", phone, exc, exc_info=True
            )
            return False

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_spreadsheet_id(self) -> Optional[str]:
        """Return the spreadsheet ID (useful for logging / admin endpoint)."""
        return self._spreadsheet_id
