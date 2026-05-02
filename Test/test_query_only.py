import os
import asyncio
import logging
from bot.rag import RAGEngine
from bot.llm import GeminiChat
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_query")

async def test_query():
    rag = RAGEngine()
    llm = GeminiChat()
    
    customer_query = "What is the customer support contact number?"
    logger.info(f"Querying RAG for: '{customer_query}'")
    
    chunks = await asyncio.get_event_loop().run_in_executor(None, rag.query, customer_query)
    retrieved_context = rag.format_for_prompt(chunks)
    
    bot_response = await llm.chat(
        customer_summary="",
        conversation_history="",
        user_query=customer_query,
        retrieved_context=retrieved_context
    )
    
    logger.info(f"Bot Final Answer: {bot_response.answer}")

if __name__ == "__main__":
    asyncio.run(test_query())
