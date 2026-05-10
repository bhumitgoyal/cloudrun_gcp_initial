import os
import asyncio
from dotenv import load_dotenv
from bot.memory import ConversationMemory

async def delete_history():
    load_dotenv()
    memory = ConversationMemory()
    doc_ref = memory._doc_ref("919354914992")
    await doc_ref.delete()
    print("Deleted document for 919354914992")

if __name__ == "__main__":
    asyncio.run(delete_history())
