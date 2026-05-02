import os
import asyncio
import logging
from bot.kb_manager import KnowledgeBaseManager
from bot.rag import RAGEngine
from bot.llm import GeminiChat
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_real_rag")

async def test_kb_flow():
    manager = KnowledgeBaseManager()
    rag = RAGEngine()
    llm = GeminiChat()
    
    # 1. Update KB
    admin_input = "Please update the customer support contact number to 9292929922. Replace any existing contact numbers with this new one."
    logger.info(f"Generating KB update for: {admin_input}")
    new_kb = await manager.generate_update(admin_input)
    
    assert "9292929922" in new_kb, "Gemini failed to insert the new phone number into the KB draft."
    logger.info("Draft generated successfully. Saving pending update...")
    
    await manager.save_pending_update(new_kb)
    
    # 2. Approve KB (Sync to Vertex AI)
    logger.info("Approving update and syncing to Vertex AI RAG Corpus...")
    success = await manager.approve_and_sync()
    if not success:
        logger.error("Failed to sync to RAG Corpus.")
        return
        
    logger.info("Sync complete. Waiting 10 seconds for RAG index to settle...")
    await asyncio.sleep(10)
    
    # 3. Emulate Customer Query
    customer_query = "What is the customer support contact number?"
    logger.info(f"Querying RAG for: '{customer_query}'")
    
    # Call RAG
    # Note: RAGEngine.query is synchronous but uses blocking network calls
    chunks = await asyncio.get_event_loop().run_in_executor(None, rag.query, customer_query)
    retrieved_context = rag.format_for_prompt(chunks)
    
    logger.info(f"Retrieved Context length: {len(retrieved_context)}")
    if "9292929922" in retrieved_context:
        logger.info("✅ SUCCESS: The new phone number was retrieved from Vertex AI RAG!")
    else:
        logger.error("❌ FAILURE: The new phone number was NOT retrieved from RAG. Context retrieved:")
        print(retrieved_context)
        return
        
    # 4. Ask Gemini to answer the customer
    bot_response = await llm.chat(
        customer_summary="",
        conversation_history="",
        user_query=customer_query,
        retrieved_context=retrieved_context
    )
    
    logger.info(f"Bot Final Answer: {bot_response.answer}")
    if "9292929922" in bot_response.answer:
        logger.info("✅ SUCCESS: The chatbot successfully provided the new phone number to the user!")
    else:
        logger.error("❌ FAILURE: The chatbot did not output the new phone number.")

if __name__ == "__main__":
    asyncio.run(test_kb_flow())
