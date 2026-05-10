import asyncio
from bot.llm import GeminiChat
from dotenv import load_dotenv

async def test():
    load_dotenv()
    llm = GeminiChat()
    response = await llm.chat(
        customer_summary="",
        conversation_history="",
        user_query="how to doewnload app",
        retrieved_context="",
        is_frustrated=False
    )
    print("Bot Answer:", response.answer)

if __name__ == "__main__":
    asyncio.run(test())
