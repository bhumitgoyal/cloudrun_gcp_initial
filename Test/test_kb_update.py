import os
import asyncio
from bot.kb_manager import KnowledgeBaseManager
from dotenv import load_dotenv

load_dotenv()

async def run_test():
    manager = KnowledgeBaseManager()
    
    # We don't want to actually update the firestore pending document in the real db if possible,
    # but generating the text is safe since we only read from Firestore.
    
    print("Fetching master KB...")
    current = await manager.get_master_kb()
    print("Master KB length:", len(current))
    
    print("Generating update with Gemini...")
    admin_input = "Please add a new policy: Trip cancellations within 24 hours of departure are not eligible for a refund."
    new_kb = await manager.generate_update(admin_input)
    print("New KB length:", len(new_kb))
    print("Does the new KB contain the cancellation policy?", "cancellation" in new_kb.lower() or "refund" in new_kb.lower())
    
if __name__ == "__main__":
    asyncio.run(run_test())
