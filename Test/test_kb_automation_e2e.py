import os
import asyncio
import logging
from unittest.mock import patch, MagicMock

from bot.whatsapp import WhatsAppClient, IncomingMessage
from bot.kb_manager import KnowledgeBaseManager
from bot.pipeline import MessagePipeline
from bot.rag import RAGEngine
from bot.memory import ConversationMemory
from bot.llm import GeminiChat
from bot.rag_cache import RAGCache

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_kb_automation")

async def test_whatsapp_media():
    logger.info("--- Testing WhatsApp Media Upload & Send ---")
    wa = WhatsAppClient()
    admin_phone = os.environ.get("ADMIN_PHONE_NUMBER")
    
    if not admin_phone:
        logger.warning("No ADMIN_PHONE_NUMBER set, skipping WhatsApp test.")
        return

    # Create dummy file
    test_file = "/tmp/dummy_test_file.md"
    with open(test_file, "w") as f:
        f.write("# Hello\nThis is a test document.")
        
    media_id = await wa.upload_media(test_file)
    assert media_id, "Media upload failed (no ID returned)"
    logger.info(f"Media uploaded successfully: {media_id}")
    
    sent = await wa.send_document(
        to=admin_phone,
        media_id=media_id,
        filename="TestDocument.md",
        caption="🧪 Automated Test: This is a test document."
    )
    assert sent, "Failed to send document"
    logger.info("Document sent successfully via WhatsApp!")
    await wa.close()

async def test_kb_manager_generation():
    logger.info("--- Testing KB Manager Generation & Saving ---")
    kb_manager = KnowledgeBaseManager()
    
    # Test fetch
    master = await kb_manager.get_master_kb()
    assert len(master) > 100, "Master KB is suspiciously short."
    logger.info(f"Master KB fetched, length: {len(master)}")
    
    # Test generation
    new_kb = await kb_manager.generate_update("Add a test note saying: 'TEST_AUTOMATION_123 is running.'")
    assert "TEST_AUTOMATION_123" in new_kb, "Gemini failed to include the new text."
    logger.info("Gemini successfully generated updated KB.")
    
    # Test saving pending update (this modifies Firestore but it's isolated to kb_pending)
    await kb_manager.save_pending_update(new_kb)
    logger.info("Pending update saved to Firestore.")

async def test_pipeline_handlers():
    logger.info("--- Testing Pipeline Handlers (Mocked Sync) ---")
    # Initialize pipeline
    wa = WhatsAppClient()
    rag_eng = RAGEngine()
    mem = ConversationMemory()
    llm = GeminiChat()
    cache = RAGCache()
    kb_manager = KnowledgeBaseManager()
    
    pipeline = MessagePipeline(
        whatsapp=wa,
        rag=rag_eng,
        memory=mem,
        llm=llm,
        cache=cache,
        kb_manager=kb_manager
    )
    
    admin_phone = os.environ.get("ADMIN_PHONE_NUMBER")
    if not admin_phone:
        return
        
    clean_admin = admin_phone.replace("+", "")
    
    # Mock Vertex AI sync to prevent real deletion/upload
    with patch("bot.kb_manager.rag") as mock_rag:
        mock_rag.list_files.return_value = [MagicMock(name="dummy_file_id")]
        
        # Test 1: Handle /update_kb
        logger.info("Simulating /update_kb command...")
        update_msg = IncomingMessage(
            wa_message_id="test_msg_id_1",
            from_number=clean_admin,
            display_name="Admin",
            phone_number_id=os.environ.get("WHATSAPP_PHONE_NUMBER_ID"),
            text="/update_kb Add a test note: PIPE_TEST_123"
        )
        await pipeline._process({"entry": [{"changes": [{"value": {
            "messages": [{"id": "test_msg_id_1", "from": clean_admin, "type": "text", "text": {"body": "/update_kb Add a test note: PIPE_TEST_123"}}],
            "contacts": [{"profile": {"name": "Admin"}}],
            "metadata": {"phone_number_id": os.environ.get("WHATSAPP_PHONE_NUMBER_ID"), "display_phone_number": "123456789"}
        }}]}]})
        # Note: _process will spawn a background task for `_handle_update_kb`. We should await it directly for testing.
        await pipeline._handle_update_kb(update_msg)
        
        # Test 2: Handle /approve_kb
        logger.info("Simulating /approve_kb command...")
        approve_msg = IncomingMessage(
            wa_message_id="test_msg_id_2",
            from_number=clean_admin,
            display_name="Admin",
            phone_number_id=os.environ.get("WHATSAPP_PHONE_NUMBER_ID"),
            text="/approve_kb"
        )
        await pipeline._handle_approve_kb(approve_msg)
        
        mock_rag.delete_file.assert_called()
        mock_rag.upload_file.assert_called()
        logger.info("Approval handler ran successfully and Vertex RAG mocked sync was triggered!")

    await wa.close()

async def run_all():
    try:
        await test_whatsapp_media()
        await test_kb_manager_generation()
        await test_pipeline_handlers()
        logger.info("\n✅ ALL TESTS PASSED!")
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_all())
