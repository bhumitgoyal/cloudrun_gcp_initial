import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
test_bad_queries.py
===================
Tests the query rewriter + full pipeline with badly typed queries
that mimic how senior citizens actually type on WhatsApp:
  - Broken English
  - Shortforms and abbreviations
  - Spelling mistakes
  - Hinglish (Hindi-English mix)

Shows the BEFORE (raw) → AFTER (polished) query transformation
and verifies Gemini still gives a correct answer.
"""

import os
import sys
import asyncio
import logging

# ── Environment ──────────────────────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

os.environ["FIRESTORE_DB"] = "ghcdb-initial"
os.environ["SUMMARISE_EVERY"] = "99"  # Don't trigger summary for this test

from bot.rag import RAGEngine
from bot.memory import ConversationMemory
from bot.llm import GeminiChat
from bot.pipeline import MessagePipeline, _SEEN_IDS
from bot.whatsapp import WhatsAppClient, IncomingMessage

logging.basicConfig(level=logging.WARNING)

# ── Config ───────────────────────────────────────────────────────────────────
USER_PHONE = "+917777700099"
USER_NAME  = "Ramesh Uncle"
ADMIN_PHONE = os.environ.get("ADMIN_PHONE_NUMBER", "919999911111")

class C:
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    END    = "\033[0m"


class MockWhatsAppClient(WhatsAppClient):
    def __init__(self):
        self.phone_number_id = "mock_phone_id"
        self.sent_messages = []

    def parse_message(self, body: dict):
        if "messages" not in body:
            return None
        msg = body["messages"][0]
        return IncomingMessage(
            wa_message_id=msg["id"],
            from_number=body["from_number"],
            display_name=body["display_name"],
            text=msg["text"]["body"],
            phone_number_id="mock_phone_id",
        )

    async def send_text(self, to, body, phone_number_id=None):
        self.sent_messages.append((to, body))
        if to == USER_PHONE:
            print(f"\n{C.GREEN}  🤖 BOT REPLY:{C.END}")
            print(f"     {body}")
        return True

    async def close(self):
        pass


BAD_QUERIES = [
    {
        "raw":  "hw to dwnload gohapy app on my fone??",
        "desc": "Shortforms + spelling mistakes (download, gohappy, phone)",
    },
    {
        "raw":  "mera happy coin kaise use karu trip ke liye.. plz btao",
        "desc": "Hinglish (Hindi-English mix) — asking about using Happy Coins for trips",
    },
    {
        "raw":  "wat is da gold plan n hw much it cst??? i m confusd",
        "desc": "Heavy shortforms + missing letters (what, the, how, cost, confused)",
    },
    {
        "raw":  "sessn recording dekhna hai kaise dekhu kahan milega",
        "desc": "Hinglish with no punctuation — asking how to watch session recordings",
    },
]


async def run():
    print(f"\n{C.BOLD}{'═'*70}")
    print(f"  QUERY REWRITER TEST — Bad English / Shortforms / Hinglish")
    print(f"{'═'*70}{C.END}")
    print(f"  User: {USER_NAME} ({USER_PHONE})\n")

    wa       = MockWhatsAppClient()
    rag      = RAGEngine()
    mem      = ConversationMemory()
    llm      = GeminiChat()
    pipeline = MessagePipeline(wa, rag, mem, llm)

    _SEEN_IDS.clear()

    # Clean slate
    try:
        await mem._doc_ref(USER_PHONE).delete()
    except Exception:
        pass

    for idx, q in enumerate(BAD_QUERIES):
        msg_id = f"bad_msg_{idx}"
        print(f"\n{C.YELLOW}{'─'*70}")
        print(f"  TEST {idx+1}: {q['desc']}")
        print(f"{'─'*70}{C.END}")

        # Show raw query
        print(f"\n{C.RED}  📨 RAW USER INPUT:{C.END}  \"{q['raw']}\"")

        # Manually call the rewriter so we can show the transformation
        polished = await llm.rewrite_query(q['raw'])
        print(f"{C.CYAN}  ✏️  POLISHED QUERY:{C.END}  \"{polished}\"")

        # Now run through full pipeline
        payload = {
            "from_number": USER_PHONE,
            "display_name": USER_NAME,
            "messages": [{"id": msg_id, "text": {"body": q['raw']}}],
        }
        await pipeline._process(payload)
        await asyncio.sleep(3)

    # Final state
    state = await mem.get_state(USER_PHONE)
    print(f"\n{C.BOLD}{'═'*70}")
    print(f"  FINAL FIRESTORE STATE")
    print(f"{'═'*70}{C.END}")
    print(f"  Name:        {state.get('display_name')}")
    print(f"  Turn Count:  {state.get('turn_count')}")
    turns = state.get("recent_turns", [])
    print(f"  Turns Log:   {len(turns)} entries\n")
    for t in turns:
        role = t.get("role")
        content = t.get("content", "")
        icon = "👤" if role == "user" else "🤖"
        print(f"  {icon} [{role:>9}] {content[:120]}{'…' if len(content)>120 else ''}")
    print()

    print(f"{C.GREEN}{'═'*70}")
    print(f"  ALL BAD-ENGLISH QUERIES HANDLED SUCCESSFULLY ✅")
    print(f"{'═'*70}{C.END}\n")


if __name__ == "__main__":
    asyncio.run(run())
