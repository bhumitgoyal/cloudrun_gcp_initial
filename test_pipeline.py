import os
import asyncio
import logging
from pprint import pprint

# Set environment
os.environ["FIRESTORE_DB"] = "ghcdb-initial"

from bot.rag import RAGEngine
from bot.memory import ConversationMemory
from bot.llm import GeminiChat
from bot.rag_cache import RAGCache
from bot.pipeline import MessagePipeline
from bot.whatsapp import WhatsAppClient, IncomingMessage

logging.basicConfig(level=logging.WARNING)

class MockWhatsAppClient(WhatsAppClient):
    def __init__(self):
        self.phone_number_id = "mock_phone_id"

    def parse_message(self, body: dict):
        if "messages" not in body:
            return None
        msg = body["messages"][0]
        return IncomingMessage(
            wa_message_id=msg["id"],
            from_number=body["from_number"],
            display_name=body["display_name"],
            text=msg["text"]["body"],
            phone_number_id="mock_phone_id"
        )
    
    async def send_text(self, to: str, body: str, phone_number_id: str = None) -> bool:
        print("\n" + "="*50)
        print(f"✅ GOHAPPY BOT REPLIES TO {to}:")
        print(f"💬 {body}")
        print("="*50 + "\n")
        return True

async def run_simulation():
    print("Initialize components...")
    wa = MockWhatsAppClient()
    rag = RAGEngine()
    mem = ConversationMemory()
    llm = GeminiChat()
    cache = RAGCache()
    pipeline = MessagePipeline(wa, rag, mem, llm, cache)

    PHONE = "+919999988888"
    NAME = "Auntie Sharma"

    # Reset testing doc in case it previously existed
    print(f"Clearing old firestore profile for {PHONE}...")
    await mem._doc_ref(PHONE).delete()

    messages = [
        "Hello, what exactly is GoHappy Club?",
        "do you guys have any upcoming trips planned?",
        "how do i download the gohappy club app on my android?",
        "how can i completely delete my account?",
        "okay thanks, that's all I need!"
    ]

    for idx, text in enumerate(messages):
        msg_id = f"mock_msg_{idx}"
        print(f"\n📨 [USER MESSAGE {idx+1}/8] {text}")
        payload = {
            "from_number": PHONE,
            "display_name": NAME,
            "messages": [{"id": msg_id, "text": {"body": text}}]
        }
        
        # Run pipeline
        await pipeline._process(payload)
        
        # Let async summary tasks (if any) finish
        await asyncio.sleep(2)
        
        # Print DB state
        state = await mem.get_state(PHONE)
        print("\n⚙️  FIRESTORE STATE UPDATE:")
        print(f"   Name:       {state.get('display_name')}")
        print(f"   Turn Count: {state.get('turn_count')}")
        print(f"   Active Summary:   {state.get('summary')}")
        recent = state.get('recent_turns', [])
        print(f"   Recent Turns log size: {len(recent)}")
        if state.get('summary'):
            print(f"   [!] Note: A rolling summary compression was triggered!")

if __name__ == "__main__":
    asyncio.run(run_simulation())
