import os
import asyncio
import logging
from bot.kb_manager import KnowledgeBaseManager
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_remove")

async def remove_number():
    manager = KnowledgeBaseManager()
    
    # 1. Update KB
    admin_input = "Please remove the phone number 9292929922 from the contact details. Ensure only the original contact numbers remain."
    logger.info(f"Generating KB update for: {admin_input}")
    new_kb = await manager.generate_update(admin_input)
    
    assert "9292929922" not in new_kb, "Gemini failed to remove the phone number from the KB draft."
    logger.info("Draft generated successfully. Saving pending update...")
    
    await manager.save_pending_update(new_kb)
    
    # 2. Approve KB (Sync to Vertex AI)
    logger.info("Approving update and syncing to Vertex AI RAG Corpus...")
    success = await manager.approve_and_sync()
    if success:
        logger.info("Successfully removed the test phone number from the Knowledge Base!")
    else:
        logger.error("Failed to sync to RAG Corpus.")

if __name__ == "__main__":
    asyncio.run(remove_number())
