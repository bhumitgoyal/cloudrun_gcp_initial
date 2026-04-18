"""
test_full_simulation.py
=======================
End-to-end simulation of the GoHappy Club WhatsApp bot pipeline.

Tests the FULL flow:
  1. Fresh user - first contact (Firestore creates new doc)
  2. RAG-powered Q&A (multiple topics)
  3. Deduplication (same message ID sent twice)
  4. Escalation trigger (frustration / account issue)
  5. Bot silence while escalated
  6. Admin /resolve command
  7. Bot resumes for user
  8. Rolling summary compression (triggered after SUMMARISE_EVERY turns)
  9. Final Firestore state dump

Uses a MockWhatsAppClient so no real WhatsApp connection is required.
All RAG, Gemini, and Firestore calls are REAL.
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime, timezone
from pprint import pformat

# ── Environment ──────────────────────────────────────────────────────────────
# Load .env manually (no dotenv dependency needed)
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

os.environ["FIRESTORE_DB"] = "ghcdb-initial"
# Force the summarise threshold low so we can trigger it in this test
os.environ["SUMMARISE_EVERY"] = "4"
os.environ["MAX_RECENT_TURNS"] = "10"

from bot.rag import RAGEngine
from bot.memory import ConversationMemory
from bot.llm import GeminiChat
from bot.pipeline import MessagePipeline, _SEEN_IDS
from bot.whatsapp import WhatsAppClient, IncomingMessage

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# ── Simulation config ────────────────────────────────────────────────────────
USER_PHONE = "+917777700001"
USER_NAME  = "Kamla Devi"
ADMIN_PHONE = os.environ.get("ADMIN_PHONE_NUMBER", "919999911111")

# ── Colours for terminal ─────────────────────────────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    END     = "\033[0m"

def banner(text, color=C.HEADER):
    width = 70
    print(f"\n{color}{'═'*width}")
    print(f"  {text}")
    print(f"{'═'*width}{C.END}\n")

def section(text, color=C.CYAN):
    print(f"\n{color}── {text} ──{C.END}")

# ═══════════════════════════════════════════════════════════════════════════════
#  MOCK WHATSAPP CLIENT — captures all outgoing messages
# ═══════════════════════════════════════════════════════════════════════════════

class MockWhatsAppClient(WhatsAppClient):
    """Intercepts all sends and logs them. No network calls."""

    def __init__(self):
        self.phone_number_id = "mock_phone_id"
        self.sent_messages = []  # list of (to, body) tuples

    def parse_message(self, body: dict):
        """Parse a simplified test payload."""
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

    async def send_text(self, to: str, body: str, phone_number_id: str = None) -> bool:
        self.sent_messages.append((to, body))
        recipient_label = "USER" if to == USER_PHONE else "ADMIN" if to == ADMIN_PHONE else to
        print(f"\n{C.GREEN}{'═'*60}")
        print(f"  ✅ BOT → {recipient_label} ({to}):")
        print(f"  💬 {body[:500]}")
        print(f"{'═'*60}{C.END}")
        return True

    async def close(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_payload(from_number, display_name, text, msg_id):
    """Build a simplified webhook payload for testing."""
    return {
        "from_number": from_number,
        "display_name": display_name,
        "messages": [{"id": msg_id, "text": {"body": text}}],
    }

async def print_firestore_state(mem: ConversationMemory, phone: str, step_label: str):
    """Fetch and print the full Firestore state for a user."""
    state = await mem.get_state(phone)
    section(f"FIRESTORE STATE after: {step_label}")
    print(f"  {C.BOLD}display_name:{C.END}     {state.get('display_name', '(none)')}")
    print(f"  {C.BOLD}turn_count:{C.END}       {state.get('turn_count', 0)}")
    print(f"  {C.BOLD}escalated:{C.END}        {state.get('escalated_to_human', False)}")
    print(f"  {C.BOLD}last_seen:{C.END}        {state.get('last_seen', '(never)')}")

    summary = state.get("summary", "")
    if summary:
        print(f"  {C.YELLOW}{C.BOLD}summary:{C.END}          {summary[:200]}{'…' if len(summary)>200 else ''}")
    else:
        print(f"  {C.BOLD}summary:{C.END}          (empty — not yet compressed)")

    recent = state.get("recent_turns", [])
    print(f"  {C.BOLD}recent_turns:{C.END}    {len(recent)} entries")
    for i, turn in enumerate(recent):
        role = turn.get("role", "?")
        content = turn.get("content", "")[:100]
        icon = "👤" if role == "user" else "🤖"
        print(f"    {icon} [{role:>9}] {content}{'…' if len(turn.get('content',''))>100 else ''}")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

async def run_simulation():
    banner("GoHappy Club — Full Pipeline Simulation", C.BOLD)
    print(f"  User Phone:  {USER_PHONE}")
    print(f"  User Name:   {USER_NAME}")
    print(f"  Admin Phone: {ADMIN_PHONE}")
    print(f"  Firestore DB: {os.environ.get('FIRESTORE_DB')}")
    print(f"  SUMMARISE_EVERY: {os.environ.get('SUMMARISE_EVERY')}")
    print()

    # ── Initialize components ────────────────────────────────────────────────
    section("Initializing pipeline components (RAG, Gemini, Firestore)…")
    wa       = MockWhatsAppClient()
    rag      = RAGEngine()
    mem      = ConversationMemory()
    llm      = GeminiChat()
    pipeline = MessagePipeline(wa, rag, mem, llm)
    print("  ✅ All components initialized.")

    # ── Clean slate ──────────────────────────────────────────────────────────
    section("Clearing old Firestore data for test user…")
    try:
        await mem._doc_ref(USER_PHONE).delete()
        print(f"  🗑️  Deleted any prior data for {USER_PHONE}")
    except Exception:
        print(f"  ℹ️  No prior data found (clean slate)")

    # Clear dedup cache
    _SEEN_IDS.clear()

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 1: First Contact — fresh user, general question
    # ══════════════════════════════════════════════════════════════════════════
    banner("TEST 1: First Contact — \"What is GoHappy Club?\"", C.BLUE)
    payload = make_payload(USER_PHONE, USER_NAME, "Hello, what exactly is GoHappy Club?", "msg_001")
    print(f"  {C.BOLD}📨 USER:{C.END} Hello, what exactly is GoHappy Club?")
    await pipeline._process(payload)
    await asyncio.sleep(3)
    await print_firestore_state(mem, USER_PHONE, "Test 1 — First Contact")

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 2: RAG Query — membership pricing
    # ══════════════════════════════════════════════════════════════════════════
    banner("TEST 2: RAG Query — \"Tell me about membership plans and pricing\"", C.BLUE)
    payload = make_payload(USER_PHONE, USER_NAME, "What are your membership plans? How much do they cost?", "msg_002")
    print(f"  {C.BOLD}📨 USER:{C.END} What are your membership plans? How much do they cost?")
    await pipeline._process(payload)
    await asyncio.sleep(3)
    await print_firestore_state(mem, USER_PHONE, "Test 2 — Membership Plans")

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 3: Follow-up using history context
    # ══════════════════════════════════════════════════════════════════════════
    banner("TEST 3: Follow-up — \"Which one is better for me?\"", C.BLUE)
    payload = make_payload(USER_PHONE, USER_NAME, "Which plan is better for me? I'm 65 years old and love yoga.", "msg_003")
    print(f"  {C.BOLD}📨 USER:{C.END} Which plan is better for me? I'm 65 years old and love yoga.")
    await pipeline._process(payload)
    await asyncio.sleep(3)
    await print_firestore_state(mem, USER_PHONE, "Test 3 — Follow-up with history")

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 4: DEDUPLICATION — same message ID sent again
    # ══════════════════════════════════════════════════════════════════════════
    banner("TEST 4: Deduplication — resending msg_003 (should be skipped)", C.YELLOW)
    sent_count_before = len(wa.sent_messages)
    payload = make_payload(USER_PHONE, USER_NAME, "Which plan is better for me? I'm 65 years old and love yoga.", "msg_003")
    print(f"  {C.BOLD}📨 USER (duplicate):{C.END} Which plan is better for me? ...")
    await pipeline._process(payload)
    sent_count_after = len(wa.sent_messages)
    if sent_count_after == sent_count_before:
        print(f"  {C.GREEN}✅ DEDUP PASSED — No duplicate reply was sent.{C.END}")
    else:
        print(f"  {C.RED}❌ DEDUP FAILED — Bot replied to a duplicate message!{C.END}")

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 5: 4th turn — triggers rolling summary compression!
    # ══════════════════════════════════════════════════════════════════════════
    banner("TEST 5: 4th Turn + Summary Compression — \"How do Happy Coins work?\"", C.BLUE)
    payload = make_payload(USER_PHONE, USER_NAME, "How do Happy Coins work? Can I use them for trips?", "msg_004")
    print(f"  {C.BOLD}📨 USER:{C.END} How do Happy Coins work? Can I use them for trips?")
    print(f"  {C.YELLOW}⚡ This is turn #4 — SUMMARISE_EVERY=4, so summary compression will trigger!{C.END}")
    await pipeline._process(payload)
    # Wait longer for the async summary compression task
    print(f"  {C.CYAN}⏳ Waiting 8 seconds for async summary compression…{C.END}")
    await asyncio.sleep(8)
    state = await print_firestore_state(mem, USER_PHONE, "Test 5 — Summary Compression")
    if state.get("summary"):
        print(f"\n  {C.GREEN}✅ SUMMARY COMPRESSION PASSED — Rolling summary was generated!{C.END}")
    else:
        print(f"\n  {C.YELLOW}⚠️  Summary is still empty — compression may still be running.{C.END}")

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 6: Escalation trigger — frustrated user / account issue
    # ══════════════════════════════════════════════════════════════════════════
    banner("TEST 6: Escalation — \"I paid for Gold but my account still shows free!\"", C.RED)
    payload = make_payload(USER_PHONE, USER_NAME,
        "I paid ₹2499 for Gold membership 3 days ago but my account still shows free plan! This is very frustrating, I want my money back!",
        "msg_005")
    print(f"  {C.BOLD}📨 USER:{C.END} I paid ₹2499 for Gold membership 3 days ago but my account still shows free plan! ...")
    await pipeline._process(payload)
    await asyncio.sleep(3)
    state = await print_firestore_state(mem, USER_PHONE, "Test 6 — Escalation")

    # Check if admin got alerted
    admin_messages = [m for m in wa.sent_messages if m[0] == ADMIN_PHONE]
    if admin_messages:
        print(f"\n  {C.GREEN}✅ ESCALATION PASSED — Admin was notified!{C.END}")
        print(f"  {C.CYAN}Admin alert message ({len(admin_messages[-1][1])} chars):{C.END}")
        print(f"    {admin_messages[-1][1][:300]}…")
    else:
        print(f"\n  {C.YELLOW}⚠️  No admin alert sent — Gemini may not have flagged escalation.{C.END}")

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 7: Bot silence while escalated
    # ══════════════════════════════════════════════════════════════════════════
    banner("TEST 7: Bot Silence — user messages while escalated (should be ignored)", C.YELLOW)
    # Ensure escalation flag is set (in case Gemini didn't flag it)
    await mem.set_escalation_status(USER_PHONE, True)
    sent_count_before = len(wa.sent_messages)
    payload = make_payload(USER_PHONE, USER_NAME, "Hello? Are you there?", "msg_006")
    print(f"  {C.BOLD}📨 USER:{C.END} Hello? Are you there?")
    await pipeline._process(payload)
    sent_count_after = len(wa.sent_messages)
    if sent_count_after == sent_count_before:
        print(f"  {C.GREEN}✅ BOT SILENCE PASSED — Bot correctly ignored the escalated user.{C.END}")
    else:
        print(f"  {C.RED}❌ BOT SILENCE FAILED — Bot replied despite escalation flag!{C.END}")
    await print_firestore_state(mem, USER_PHONE, "Test 7 — Bot Silence")

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 8: Admin /resolve command
    # ══════════════════════════════════════════════════════════════════════════
    banner("TEST 8: Admin /resolve — resuming bot for user", C.GREEN)
    resolve_payload = make_payload(ADMIN_PHONE, "Admin", f"/resolve {USER_PHONE}", "msg_admin_001")
    print(f"  {C.BOLD}📨 ADMIN:{C.END} /resolve {USER_PHONE}")
    await pipeline._process(resolve_payload)
    await asyncio.sleep(1)
    state = await print_firestore_state(mem, USER_PHONE, "Test 8 — Admin Resolve")
    if not state.get("escalated_to_human", True):
        print(f"\n  {C.GREEN}✅ ADMIN RESOLVE PASSED — Bot is back in control!{C.END}")
    else:
        print(f"\n  {C.RED}❌ ADMIN RESOLVE FAILED — escalation flag still True!{C.END}")

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 9: User messages again after resolve — bot should respond
    # ══════════════════════════════════════════════════════════════════════════
    banner("TEST 9: Post-Resolve — \"Do you have any upcoming trips?\"", C.BLUE)
    sent_count_before = len(wa.sent_messages)
    payload = make_payload(USER_PHONE, USER_NAME, "Do you have any upcoming trips planned for seniors?", "msg_007")
    print(f"  {C.BOLD}📨 USER:{C.END} Do you have any upcoming trips planned for seniors?")
    await pipeline._process(payload)
    await asyncio.sleep(3)
    sent_count_after = len(wa.sent_messages)
    if sent_count_after > sent_count_before:
        print(f"  {C.GREEN}✅ POST-RESOLVE PASSED — Bot is responding again!{C.END}")
    else:
        print(f"  {C.RED}❌ POST-RESOLVE FAILED — Bot is still silent!{C.END}")
    await print_firestore_state(mem, USER_PHONE, "Test 9 — Post-Resolve")

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 10: Admin sends a non-command message (should get help text)
    # ══════════════════════════════════════════════════════════════════════════
    banner("TEST 10: Admin Non-Command — admin sends random message", C.YELLOW)
    admin_payload = make_payload(ADMIN_PHONE, "Admin", "Hey, what's up?", "msg_admin_002")
    print(f"  {C.BOLD}📨 ADMIN:{C.END} Hey, what's up?")
    await pipeline._process(admin_payload)

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    banner("FINAL STATE — Complete Firestore Document Dump", C.BOLD)
    final_state = await mem.get_state(USER_PHONE)

    print(f"{C.BOLD}User:{C.END}            {final_state.get('display_name')}")
    print(f"{C.BOLD}Phone:{C.END}           {USER_PHONE}")
    print(f"{C.BOLD}Total Turns:{C.END}     {final_state.get('turn_count')}")
    print(f"{C.BOLD}Escalated:{C.END}       {final_state.get('escalated_to_human')}")
    print(f"{C.BOLD}Last Seen:{C.END}       {final_state.get('last_seen')}")
    print()

    summary = final_state.get("summary", "")
    if summary:
        print(f"{C.YELLOW}{C.BOLD}Rolling Summary:{C.END}")
        print(f"  {summary}")
    else:
        print(f"{C.BOLD}Rolling Summary:{C.END} (none)")
    print()

    recent = final_state.get("recent_turns", [])
    print(f"{C.BOLD}Full Conversation Log ({len(recent)} entries):{C.END}")
    print(f"{C.CYAN}{'─'*60}{C.END}")
    for i, turn in enumerate(recent):
        role = turn.get("role", "?")
        content = turn.get("content", "")
        ts = turn.get("ts", "")
        if role == "user":
            print(f"  {C.BLUE}👤 CUSTOMER [{ts}]:{C.END}")
            print(f"     {content}")
        else:
            print(f"  {C.GREEN}🤖 BOT [{ts}]:{C.END}")
            print(f"     {content}")
        print()
    print(f"{C.CYAN}{'─'*60}{C.END}")

    # ── All outgoing messages summary ─────────────────────────────────────────
    banner("ALL OUTGOING MESSAGES SENT BY BOT", C.GREEN)
    for i, (to, body) in enumerate(wa.sent_messages, 1):
        recipient = "USER" if to == USER_PHONE else "ADMIN" if to == ADMIN_PHONE else to
        print(f"  {C.BOLD}[{i}] → {recipient} ({to}):{C.END}")
        print(f"     {body[:300]}{'…' if len(body)>300 else ''}")
        print()

    banner("SIMULATION COMPLETE ✅", C.GREEN)


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(run_simulation())
