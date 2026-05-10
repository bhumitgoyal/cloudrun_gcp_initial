"""
Test/test_app_chat.py
Integration tests for the in-app chat API endpoints.

Uses FastAPI's TestClient with mocked pipeline components
so we can test the endpoints without GCP credentials.
"""

import os
import sys
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Set env vars before any app imports ──────────────────────────────────────
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_LOCATION", "us-central1")
os.environ.setdefault("WHATSAPP_ACCESS_TOKEN", "test-token")
os.environ.setdefault("WHATSAPP_PHONE_NUMBER_ID", "1234567890")
os.environ.setdefault("WHATSAPP_VERIFY_TOKEN", "test-verify")
os.environ.setdefault("VERTEX_RAG_CORPUS", "projects/test/locations/us-central1/ragCorpora/1")

from fastapi.testclient import TestClient
from bot.llm import BotResponse


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline with handle_app_message mocked."""
    pipeline = MagicMock()
    pipeline.handle_app_message = AsyncMock()
    pipeline.memory = MagicMock()
    pipeline.memory.get_state = AsyncMock()
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """Create a test client with mocked pipeline."""
    from main import app
    app.state.pipeline = mock_pipeline
    # Also mock cache for warmup
    app.state.cache = MagicMock()
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/chat
# ══════════════════════════════════════════════════════════════════════════════

class TestAppChat:

    def test_basic_chat(self, client, mock_pipeline):
        """Normal message should return bot reply."""
        mock_pipeline.handle_app_message.return_value = BotResponse(
            answer="GoHappy Club is India's senior community platform for people aged 50+.",
            escalation=False,
        )

        resp = client.post("/api/chat", json={
            "user_id": "user_123",
            "display_name": "Ramesh",
            "message": "What is GoHappy Club?",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "GoHappy Club is India's senior community platform for people aged 50+."
        assert data["escalation"] is False

        # Verify pipeline was called with correct args
        mock_pipeline.handle_app_message.assert_called_once_with(
            user_id="user_123",
            display_name="Ramesh",
            text="What is GoHappy Club?",
            message_id=None,
        )

    def test_chat_with_message_id(self, client, mock_pipeline):
        """Message with dedup ID should pass it through."""
        mock_pipeline.handle_app_message.return_value = BotResponse(
            answer="Our Gold plan costs ₹999/year.",
            escalation=False,
        )

        resp = client.post("/api/chat", json={
            "user_id": "user_456",
            "message": "Gold plan price?",
            "message_id": "msg-abc-123",
        })

        assert resp.status_code == 200
        mock_pipeline.handle_app_message.assert_called_once_with(
            user_id="user_456",
            display_name="App User",  # default
            text="Gold plan price?",
            message_id="msg-abc-123",
        )

    def test_chat_escalation(self, client, mock_pipeline):
        """Escalation flag should be returned in response."""
        mock_pipeline.handle_app_message.return_value = BotResponse(
            answer="I'll connect you with our support team for this.",
            escalation=True,
        )

        resp = client.post("/api/chat", json={
            "user_id": "user_789",
            "message": "My payment failed and I need a refund!",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["escalation"] is True

    def test_chat_filtered_message(self, client, mock_pipeline):
        """Filtered messages return empty reply."""
        mock_pipeline.handle_app_message.return_value = None

        resp = client.post("/api/chat", json={
            "user_id": "user_123",
            "message": "Good morning",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == ""
        assert data["escalation"] is False

    def test_chat_empty_message_rejected(self, client, mock_pipeline):
        """Empty message should return 400."""
        resp = client.post("/api/chat", json={
            "user_id": "user_123",
            "message": "   ",
        })

        assert resp.status_code == 400

    def test_chat_missing_user_id(self, client, mock_pipeline):
        """Missing user_id should return 422 (validation error)."""
        resp = client.post("/api/chat", json={
            "message": "Hello",
        })

        assert resp.status_code == 422

    def test_chat_missing_message(self, client, mock_pipeline):
        """Missing message field should return 422."""
        resp = client.post("/api/chat", json={
            "user_id": "user_123",
        })

        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
#  GET /api/chat/history/{user_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestChatHistory:

    def test_existing_user_history(self, client, mock_pipeline):
        """Should return conversation history for an existing user."""
        mock_pipeline.memory.get_state.return_value = {
            "display_name": "Ramesh Kumar",
            "recent_turns": [
                {"role": "user", "content": "What is GoHappy?", "ts": "2026-05-10T10:00:00Z"},
                {"role": "assistant", "content": "GoHappy Club is a community for seniors.", "ts": "2026-05-10T10:00:01Z"},
            ],
            "summary": "Ramesh asked about GoHappy Club.",
        }

        resp = client.get("/api/chat/history/user_123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["display_name"] == "Ramesh Kumar"
        assert len(data["turns"]) == 2
        assert data["turns"][0]["role"] == "user"
        assert data["summary"] == "Ramesh asked about GoHappy Club."

    def test_new_user_history(self, client, mock_pipeline):
        """New user should return empty history."""
        mock_pipeline.memory.get_state.return_value = {
            "display_name": "",
            "recent_turns": [],
            "summary": "",
        }

        resp = client.get("/api/chat/history/new_user")

        assert resp.status_code == 200
        data = resp.json()
        assert data["display_name"] == ""
        assert data["turns"] == []
        assert data["summary"] == ""


# ══════════════════════════════════════════════════════════════════════════════
#  Health check (existing)
# ══════════════════════════════════════════════════════════════════════════════

class TestHealth:

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
#  POST /insights_update
# ══════════════════════════════════════════════════════════════════════════════

class TestInsightsUpdate:

    def test_insights_update_success(self, client, mock_pipeline):
        """Full success path: fetch insight → Gemini update → RAG sync."""
        mock_pipeline.kb_insights = MagicMock()
        mock_pipeline.kb_insights.get_latest_insight = AsyncMock(
            return_value="Add session recording policy for Silver members."
        )
        mock_pipeline.kb_manager = MagicMock()
        mock_pipeline.kb_manager.generate_update = AsyncMock(
            return_value="# Updated KB\n\nNew content here..."
        )
        mock_pipeline.kb_manager.save_pending_update = AsyncMock()
        mock_pipeline.kb_manager.approve_and_sync = AsyncMock(return_value=True)

        resp = client.post("/insights_update")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "synced to RAG" in data["message"]
        assert data["insight_length"] > 0
        assert data["new_kb_length"] > 0

        # Verify the chain was called correctly
        mock_pipeline.kb_insights.get_latest_insight.assert_called_once()
        mock_pipeline.kb_manager.generate_update.assert_called_once()
        mock_pipeline.kb_manager.save_pending_update.assert_called_once()
        mock_pipeline.kb_manager.approve_and_sync.assert_called_once()

    def test_insights_update_no_insights(self, client, mock_pipeline):
        """Should return 404 when no insights exist yet."""
        mock_pipeline.kb_insights = MagicMock()
        mock_pipeline.kb_insights.get_latest_insight = AsyncMock(
            side_effect=ValueError("No insight rows found. Run /insights first.")
        )
        mock_pipeline.kb_manager = MagicMock()

        resp = client.post("/insights_update")

        assert resp.status_code == 404
        assert "Run /insights first" in resp.json()["detail"]

    def test_insights_update_gemini_failure(self, client, mock_pipeline):
        """Should return 500 when Gemini fails to generate the update."""
        mock_pipeline.kb_insights = MagicMock()
        mock_pipeline.kb_insights.get_latest_insight = AsyncMock(
            return_value="Some insight text"
        )
        mock_pipeline.kb_manager = MagicMock()
        mock_pipeline.kb_manager.generate_update = AsyncMock(
            side_effect=Exception("Gemini quota exceeded")
        )

        resp = client.post("/insights_update")

        assert resp.status_code == 500
        assert "Failed to generate KB update" in resp.json()["detail"]

    def test_insights_update_rag_sync_failure(self, client, mock_pipeline):
        """Should return 500 when RAG sync fails."""
        mock_pipeline.kb_insights = MagicMock()
        mock_pipeline.kb_insights.get_latest_insight = AsyncMock(
            return_value="Some insight text"
        )
        mock_pipeline.kb_manager = MagicMock()
        mock_pipeline.kb_manager.generate_update = AsyncMock(
            return_value="# Updated KB content"
        )
        mock_pipeline.kb_manager.save_pending_update = AsyncMock()
        mock_pipeline.kb_manager.approve_and_sync = AsyncMock(return_value=False)

        resp = client.post("/insights_update")

        assert resp.status_code == 500
        assert "RAG sync returned failure" in resp.json()["detail"]

    def test_insights_update_not_initialised(self, client, mock_pipeline):
        """Should return 500 when KB modules are not initialised."""
        mock_pipeline.kb_insights = None
        mock_pipeline.kb_manager = None

        resp = client.post("/insights_update")

        assert resp.status_code == 500
        assert "not initialised" in resp.json()["detail"]
