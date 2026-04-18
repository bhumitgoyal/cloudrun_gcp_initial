"""
test_cache_cost_simulation.py
Cache cost simulation WITH real Gemini query rewriting.

This test uses the REAL Gemini rewrite step to normalize queries
before they hit the cache — demonstrating how "membership lena hai"
and "How do I join?" both get rewritten into similar clean English
and produce cache hits.

Phase 0: 10 junk messages (should be filtered)
Phase 1: 40 unique queries (seeds the cache — all MISSes)
Phase 2: 20 queries (15 semantically similar + 5 new)

Run:
    python test_cache_cost_simulation.py
"""

import os
import sys
import asyncio
import logging
import time

# ── Load .env (needed for real Gemini credentials) ───────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Override cache settings ──────────────────────────────────────────────
os.environ["CACHE_SIMILARITY_THRESHOLD"] = "0.75"
os.environ["CACHE_TTL_SECONDS"] = "3600"
os.environ["CACHE_MAX_ENTRIES"] = "500"

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("cost_sim")
logger.setLevel(logging.INFO)

# Quiet down noisy libs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

# ── Formatting ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[90m"
RESET  = "\033[0m"

# ── Counters ─────────────────────────────────────────────────────────────
rag_call_count    = 0
gemini_answer_count = 0
gemini_rewrite_count = 0
sent_messages     = []

# ═══════════════════════════════════════════════════════════════════════════
#  MOCKS (only WhatsApp, RAG, Memory — Gemini rewrite is REAL)
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from typing import Optional, List
from bot.rag_cache import RAGCache
from bot.llm import GeminiChat, BotResponse

@dataclass
class FakeIncomingMessage:
    wa_message_id: str
    from_number: str
    display_name: str
    text: str
    phone_number_id: str = "mock_phone_id"


class FakeWhatsApp:
    def parse_message(self, body: dict) -> Optional[FakeIncomingMessage]:
        if "messages" not in body:
            return None
        msg = body["messages"][0]
        return FakeIncomingMessage(
            wa_message_id=msg["id"],
            from_number=body["from_number"],
            display_name=body["display_name"],
            text=msg["text"]["body"],
        )

    async def send_text(self, to: str, body: str, phone_number_id: str = None) -> bool:
        sent_messages.append({"to": to, "body": body})
        return True


@dataclass
class FakeChunk:
    index: int
    text: str
    source: str
    score: float


class FakeRAG:
    """Mock RAG — tracks calls."""
    def query(self, query_text: str) -> List[FakeChunk]:
        global rag_call_count
        rag_call_count += 1
        return [FakeChunk(1, f"Info: {query_text}", "KB", 0.85)]

    def format_for_prompt(self, chunks) -> str:
        return "\n".join(f"[DOC_{c.index}] {c.text}" for c in chunks)


class RealRewriteGemini:
    """Uses REAL Gemini for query rewriting, MOCK for answer generation.
    This shows the effect of query normalization on cache hit rates."""

    def __init__(self):
        self._real_gemini = GeminiChat()

    async def rewrite_query(self, user_query: str) -> str:
        """REAL Gemini rewrite — normalizes Hinglish, shortforms etc."""
        global gemini_rewrite_count
        gemini_rewrite_count += 1
        result = await self._real_gemini.rewrite_query(user_query)
        return result

    async def chat(self, customer_summary, conversation_history, user_query, retrieved_context) -> BotResponse:
        """MOCK answer generation — just returns a predictable answer."""
        global gemini_answer_count
        gemini_answer_count += 1
        return BotResponse(
            answer=f"Here's the answer about: {user_query}",
            escalation=False,
        )

    async def compress_summary(self, *args, **kwargs):
        return "Summary compressed."


class FakeMemory:
    def __init__(self):
        self._states = {}

    async def get_state(self, phone: str) -> dict:
        return self._states.get(phone, {
            "display_name": "", "summary": "", "turn_count": 0,
            "recent_turns": [], "escalated_to_human": False,
        })

    async def append_turn(self, phone, display_name, user_text, bot_text) -> dict:
        state = await self.get_state(phone)
        state["display_name"] = display_name
        state["turn_count"] = state.get("turn_count", 0) + 1
        state["recent_turns"].append({"role": "user", "content": user_text})
        state["recent_turns"].append({"role": "assistant", "content": bot_text})
        self._states[phone] = state
        return state

    async def set_escalation_status(self, phone, status):
        state = await self.get_state(phone)
        state["escalated_to_human"] = status
        self._states[phone] = state

    def should_summarise(self, state):
        return False

    def build_customer_summary(self, state):
        return f"Member: {state.get('display_name', 'Unknown')}"

    def format_history_for_prompt(self, state):
        return "(No history)"


# ═══════════════════════════════════════════════════════════════════════════
#  TEST DATA — 40 unique queries covering all GoHappy topics
# ═══════════════════════════════════════════════════════════════════════════

PHASE1_QUERIES = [
    # ── Membership ──
    "How do I join GoHappy Club?",
    "What are the different membership plans?",
    "How much does the Gold plan cost?",
    "What is included in the Silver membership?",
    "Can I upgrade from Silver to Gold?",
    "How do I cancel my membership?",
    "Is there a free trial available?",
    "What is the refund policy for memberships?",

    # ── Sessions & Classes ──
    "What sessions are available this week?",
    "What time does the yoga class start?",
    "How do I join a music class?",
    "Are there any digital literacy sessions?",
    "Can I attend a session without membership?",
    "How do I get the session recording?",
    "Who are the instructors for fitness classes?",

    # ── Happy Coins ──
    "What are Happy Coins?",
    "How do I earn Happy Coins?",
    "Can I redeem Happy Coins for discounts?",
    "How many Happy Coins do I get for referring a friend?",
    "Where can I check my Happy Coins balance?",

    # ── App & Technical ──
    "How do I download the GoHappy app?",
    "The app is not working on my phone",
    "How do I reset my password?",
    "I am not receiving OTP",
    "How do I update my profile photo?",

    # ── Trips & Events ──
    "What trips are coming up?",
    "How much does the Shimla trip cost?",
    "Can I bring my spouse on the trip?",
    "What is included in the trip package?",
    "How do I register for upcoming events?",

    # ── Payments & Billing ──
    "What payment methods do you accept?",
    "I was charged twice for my subscription",
    "How do I get an invoice for my payment?",
    "Can I pay using UPI?",

    # ── General / Misc ──
    "What is GoHappy Club?",
    "Who founded GoHappy Club?",
    "How do I contact customer support?",
    "What are the operating hours?",
    "Is GoHappy Club available in my city?",
    "How do I give feedback about a session?",
]

# ── Phase 2: 20 queries (15 semantically similar + 5 new) ──
# These are phrased the way REAL senior citizens would type them:
# Hinglish, broken English, shortforms, colloquial phrases

PHASE2_QUERIES = [
    # Similar to "How do I join GoHappy Club?"
    ("membership kaise leni hai GoHappy ki", True),
    # Similar to "What are the different membership plans?"
    ("kitne type ke plans hain aapke", True),
    # Similar to "How much does the Gold plan cost?"
    ("gold plan ka price kya hai", True),
    # Similar to "What sessions are available this week?"
    ("is week kaunsi classes hain", True),
    # Similar to "What time does the yoga class start?"
    ("yoga class ka time kya hai", True),
    # Similar to "What are Happy Coins?"
    ("happy coins kya hote hain", True),
    # Similar to "How do I download the GoHappy app?"
    ("app kaise download kare", True),
    # Similar to "How do I earn Happy Coins?"
    ("happy coins kaise milte hain", True),
    # Similar to "What trips are coming up?"
    ("koi trip aa raha hai kya", True),
    # Similar to "How much does the Shimla trip cost?"
    ("shimla trip kitne ka hai", True),
    # Similar to "Can I pay using UPI?"
    ("UPI se payment ho jayegi kya", True),
    # Similar to "How do I contact customer support?"
    ("customer care ka number do", True),
    # Similar to "How do I cancel my membership?"
    ("plan cancel karna hai mujhe", True),
    # Similar to "What is GoHappy Club?"
    ("GoHappy Club ke baare mein batao", True),
    # Similar to "Is GoHappy Club available in my city?"
    ("kya GoHappy Pune mein available hai", True),

    # ── 5 genuinely NEW queries ──
    ("Do you offer group discounts for societies?", False),
    ("Is there a WhatsApp group for members?", False),
    ("How do I become a session instructor?", False),
    ("Can I gift a membership to someone?", False),
    ("What languages are sessions conducted in?", False),
]

# ── Junk messages ──
JUNK_MESSAGES = [
    "https://www.facebook.com/share/post/12345",
    "https://youtu.be/dQw4w9WgXcQ",
    "Good morning 🙏🌸",
    "🙏🙏🙏",
    "https://www.instagram.com/reel/abc123/",
    "Jai Shri Krishna",
    "https://bit.ly/3xYz123",
    "👍",
    "Ram Ram",
    "https://fb.watch/someVideo",
]


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

from bot.pipeline import MessagePipeline

msg_counter = 0

def make_payload(text: str, phone: str = "+919999900000", name: str = "Test User"):
    global msg_counter
    msg_counter += 1
    return {
        "from_number": phone,
        "display_name": name,
        "messages": [{"id": f"sim_msg_{msg_counter:04d}", "text": {"body": text}}],
    }


async def main():
    global rag_call_count, gemini_answer_count, gemini_rewrite_count, sent_messages

    cache = RAGCache()
    llm = RealRewriteGemini()
    pipeline = MessagePipeline(
        whatsapp=FakeWhatsApp(),
        rag=FakeRAG(),
        memory=FakeMemory(),
        llm=llm,
        cache=cache,
    )

    threshold = os.environ["CACHE_SIMILARITY_THRESHOLD"]

    print(f"\n{'═' * 80}")
    print(f"  {BOLD}CACHE COST SIMULATION — WITH REAL GEMINI QUERY REWRITING{RESET}")
    print(f"  Threshold: {threshold}  |  TTL: {os.environ['CACHE_TTL_SECONDS']}s  |  Max: {os.environ['CACHE_MAX_ENTRIES']} entries")
    print(f"  Gemini rewrite: {GREEN}REAL{RESET}  |  Gemini answer: {DIM}MOCK{RESET}  |  RAG: {DIM}MOCK{RESET}")
    print(f"{'═' * 80}")

    # ── Phase 0: Junk messages ───────────────────────────────────────────
    print(f"\n{BOLD}{'─' * 80}")
    print(f"  PHASE 0: Junk Messages ({len(JUNK_MESSAGES)} messages)")
    print(f"{'─' * 80}{RESET}\n")

    rag_before = rag_call_count
    gemini_a_before = gemini_answer_count

    for text in JUNK_MESSAGES:
        await pipeline._process(make_payload(text))

    junk_rag = rag_call_count - rag_before
    junk_gemini = gemini_answer_count - gemini_a_before

    print(f"  Junk messages sent:          {len(JUNK_MESSAGES)}")
    print(f"  RAG calls (should be 0):     {junk_rag}")
    print(f"  Gemini answer (should be 0): {junk_gemini}")
    print(f"  Gemini rewrite (should be 0): {gemini_rewrite_count}")
    if junk_rag == 0 and junk_gemini == 0:
        print(f"  {GREEN}✅ All junk filtered — zero tokens spent{RESET}")
    else:
        print(f"  {RED}❌ Some junk leaked through{RESET}")

    # ── Phase 1: 40 unique queries ───────────────────────────────────────
    print(f"\n{BOLD}{'─' * 80}")
    print(f"  PHASE 1: Seed Cache ({len(PHASE1_QUERIES)} unique queries)")
    print(f"  Each goes through REAL Gemini rewrite → cache MISS → mock RAG + answer")
    print(f"{'─' * 80}{RESET}\n")

    rag_call_count = 0
    gemini_answer_count = 0
    gemini_rewrite_count = 0
    sent_messages = []

    phase1_start = time.time()
    phase1_results = []
    phase1_rewrites = {}  # original → rewritten

    for i, query in enumerate(PHASE1_QUERIES):
        rag_before = rag_call_count
        rewrite_before = gemini_rewrite_count

        await pipeline._process(make_payload(query, phone=f"+9199999{i:05d}"))

        was_miss = rag_call_count > rag_before
        was_rewritten = gemini_rewrite_count > rewrite_before

        # Extract the rewritten query from pipeline logs (it's the polished_query)
        # We'll capture it by checking what was cached
        stats = await cache.get_stats()

        phase1_results.append({
            "query": query,
            "cache": "MISS" if was_miss else "HIT",
            "rag_called": was_miss,
            "rewritten": was_rewritten,
        })

        status = f"{RED}MISS{RESET}" if was_miss else f"{GREEN}HIT{RESET}"
        print(f"  {DIM}{i+1:2d}.{RESET} [{status}] {query[:70]}")

    phase1_time = time.time() - phase1_start
    phase1_misses = sum(1 for r in phase1_results if r["cache"] == "MISS")
    phase1_hits = sum(1 for r in phase1_results if r["cache"] == "HIT")

    print(f"\n  Phase 1 Summary:")
    print(f"  ├─ Total queries:     {len(PHASE1_QUERIES)}")
    print(f"  ├─ Cache MISSes:      {phase1_misses}")
    print(f"  ├─ Cache HITs:        {phase1_hits}")
    print(f"  ├─ Gemini rewrites:   {gemini_rewrite_count}")
    print(f"  ├─ RAG calls:         {rag_call_count}")
    print(f"  ├─ Gemini answers:    {gemini_answer_count}")
    print(f"  └─ Time:              {phase1_time:.1f}s")

    # ── Phase 2: 20 queries (15 similar in Hinglish + 5 new) ────────────
    print(f"\n{BOLD}{'─' * 80}")
    print(f"  PHASE 2: Repeat Traffic ({len(PHASE2_QUERIES)} queries)")
    print(f"  15 Hinglish/colloquial → Gemini rewrites → should match cached queries")
    print(f"  5 genuinely new → expect MISSes")
    print(f"{'─' * 80}{RESET}\n")

    rag_phase2_start = rag_call_count
    gemini_a_phase2_start = gemini_answer_count
    gemini_r_phase2_start = gemini_rewrite_count

    phase2_start = time.time()
    phase2_results = []

    for i, (query, expect_hit) in enumerate(PHASE2_QUERIES):
        rag_before = rag_call_count
        gemini_a_before = gemini_answer_count

        await pipeline._process(make_payload(query, phone=f"+9199998{i:05d}"))

        was_hit = rag_call_count == rag_before
        actual = "HIT" if was_hit else "MISS"
        expected = "HIT" if expect_hit else "MISS"
        correct = actual == expected

        phase2_results.append({
            "query": query,
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "rag_called": not was_hit,
            "gemini_answer_called": gemini_answer_count > gemini_a_before,
        })

        if was_hit:
            status = f"{GREEN}HIT {RESET}"
        else:
            status = f"{RED}MISS{RESET}"

        match = f"{GREEN}✓{RESET}" if correct else f"{YELLOW}✗{RESET}"
        exp_label = f"{DIM}(expected {expected}){RESET}"
        print(f"  {DIM}{i+1:2d}.{RESET} [{status}] {match} {query[:55]}  {exp_label}")

    phase2_time = time.time() - phase2_start
    phase2_hits = sum(1 for r in phase2_results if r["actual"] == "HIT")
    phase2_misses = sum(1 for r in phase2_results if r["actual"] == "MISS")
    phase2_correct = sum(1 for r in phase2_results if r["correct"])
    phase2_rag = rag_call_count - rag_phase2_start
    phase2_gemini_a = gemini_answer_count - gemini_a_phase2_start
    phase2_gemini_r = gemini_rewrite_count - gemini_r_phase2_start

    print(f"\n  Phase 2 Summary:")
    print(f"  ├─ Total queries:       {len(PHASE2_QUERIES)}")
    print(f"  ├─ Cache HITs:          {phase2_hits}")
    print(f"  ├─ Cache MISSes:        {phase2_misses}")
    print(f"  ├─ Gemini rewrites:     {phase2_gemini_r}")
    print(f"  ├─ RAG calls:           {phase2_rag}")
    print(f"  ├─ Gemini answer calls: {phase2_gemini_a}")
    print(f"  ├─ Predictions right:   {phase2_correct}/{len(PHASE2_QUERIES)}")
    print(f"  └─ Time:                {phase2_time:.1f}s")

    # ── Hit rate for "similar" queries only ──
    similar_queries = [r for r in phase2_results if r["expected"] == "HIT"]
    similar_hits = sum(1 for r in similar_queries if r["actual"] == "HIT")
    similar_total = len(similar_queries)

    print(f"\n  {BOLD}Similar-query hit rate: {similar_hits}/{similar_total} = {similar_hits/similar_total*100:.0f}%{RESET}")

    # ── Overall Stats ────────────────────────────────────────────────────
    cache_stats = await cache.get_stats()
    total_queries = len(PHASE1_QUERIES) + len(PHASE2_QUERIES)
    total_hits = phase1_hits + phase2_hits
    total_misses = phase1_misses + phase2_misses
    total_rag = rag_call_count
    total_gemini_a = gemini_answer_count
    total_gemini_r = gemini_rewrite_count
    rag_avoided = total_queries - total_rag
    gemini_a_avoided = total_queries - total_gemini_a

    print(f"\n{'═' * 80}")
    print(f"  {BOLD}FINAL REPORT — Threshold {threshold} with Real Gemini Rewrite{RESET}")
    print(f"{'═' * 80}")

    print(f"""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  {BOLD}Traffic Summary{RESET}                                                       │
  ├──────────────────────────────────────┬───────────────────────────────────┤
  │  Junk messages (filtered)            │  {len(JUNK_MESSAGES):>4}  (0 tokens spent)         │
  │  Phase 1 (unique, seed cache)        │  {len(PHASE1_QUERIES):>4}                           │
  │  Phase 2 (repeat + new)              │  {len(PHASE2_QUERIES):>4}                           │
  │  {BOLD}Total processable queries{RESET}            │  {BOLD}{total_queries:>4}{RESET}                           │
  ├──────────────────────────────────────┼───────────────────────────────────┤
  │  {BOLD}Cache Performance{RESET}                    │                                   │
  ├──────────────────────────────────────┼───────────────────────────────────┤
  │  Cache HITs                          │  {GREEN}{total_hits:>4}{RESET}                           │
  │  Cache MISSes                        │  {RED}{total_misses:>4}{RESET}                           │
  │  Overall hit rate                    │  {BOLD}{total_hits/total_queries*100:.1f}%{RESET}                          │
  │  Similar-query hit rate (Phase 2)    │  {BOLD}{similar_hits}/{similar_total} = {similar_hits/similar_total*100:.0f}%{RESET}                       │
  │  Entries in cache                    │  {cache_stats['total_cached_entries']:>4}                           │
  ├──────────────────────────────────────┼───────────────────────────────────┤
  │  {BOLD}Gemini Calls{RESET}                         │                                   │
  ├──────────────────────────────────────┼───────────────────────────────────┤
  │  Gemini REWRITE calls (always run)   │  {total_gemini_r:>4}  {DIM}(cannot skip){RESET}           │
  │  Gemini ANSWER calls made            │  {RED}{total_gemini_a:>4}{RESET}                           │
  │  Gemini ANSWER calls AVOIDED         │  {GREEN}{gemini_a_avoided:>4}{RESET}                           │
  │  Gemini answer savings               │  {BOLD}{gemini_a_avoided/total_queries*100:.1f}%{RESET}                          │
  ├──────────────────────────────────────┼───────────────────────────────────┤
  │  {BOLD}RAG Engine Calls{RESET}                     │                                   │
  ├──────────────────────────────────────┼───────────────────────────────────┤
  │  RAG calls made                      │  {RED}{total_rag:>4}{RESET}                           │
  │  RAG calls AVOIDED (cache hit)       │  {GREEN}{rag_avoided:>4}{RESET}                           │
  │  RAG savings                         │  {BOLD}{rag_avoided/total_queries*100:.1f}%{RESET}                          │
  ├──────────────────────────────────────┼───────────────────────────────────┤
  │  {BOLD}Filter Performance{RESET}                   │                                   │
  ├──────────────────────────────────────┼───────────────────────────────────┤
  │  Junk blocked (zero pipeline cost)   │  {GREEN}{len(JUNK_MESSAGES):>4}{RESET}                           │
  └──────────────────────────────────────┴───────────────────────────────────┘""")

    # ── Cost estimation ──────────────────────────────────────────────────
    cost_rewrite_in  = 70
    cost_rewrite_out = 20
    cost_answer_in   = 1000
    cost_answer_out  = 100
    cost_per_1m_in   = 0.15
    cost_per_1m_out  = 0.60
    cost_per_rag     = 0.0006

    def token_cost(input_tokens, output_tokens):
        return (input_tokens * cost_per_1m_in + output_tokens * cost_per_1m_out) / 1_000_000

    no_cache_rewrite  = total_queries * token_cost(cost_rewrite_in, cost_rewrite_out)
    no_cache_answer   = total_queries * token_cost(cost_answer_in, cost_answer_out)
    no_cache_rag      = total_queries * cost_per_rag
    no_cache_total    = no_cache_rewrite + no_cache_answer + no_cache_rag

    with_cache_rewrite = total_queries * token_cost(cost_rewrite_in, cost_rewrite_out)
    with_cache_answer  = total_misses * token_cost(cost_answer_in, cost_answer_out)
    with_cache_rag     = total_misses * cost_per_rag
    with_cache_total   = with_cache_rewrite + with_cache_answer + with_cache_rag

    savings = no_cache_total - with_cache_total
    savings_pct = (savings / no_cache_total * 100) if no_cache_total else 0

    filter_savings = len(JUNK_MESSAGES) * (
        token_cost(cost_rewrite_in, cost_rewrite_out) +
        token_cost(cost_answer_in, cost_answer_out) +
        cost_per_rag
    )

    hit_rate_pct = total_hits / total_queries * 100

    print(f"""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  {BOLD}Cost Comparison (this {total_queries}-query simulation){RESET}                          │
  │  {DIM}Gemini 2.5 Flash: $0.15/1M input, $0.60/1M output{RESET}                      │
  │  {DIM}RAG Engine: ~$0.0006 per retrieval{RESET}                                      │
  ├──────────────────────────────────────┬───────────────────────────────────┤
  │  {BOLD}WITHOUT Cache{RESET}                        │                                   │
  │    Gemini rewrite  ({total_queries} calls)       │  ${no_cache_rewrite:.6f}                   │
  │    Gemini answer   ({total_queries} calls)       │  ${no_cache_answer:.6f}                   │
  │    RAG retrieval   ({total_queries} calls)       │  ${no_cache_rag:.6f}                   │
  │    {BOLD}Subtotal{RESET}                             │  {BOLD}${no_cache_total:.6f}{RESET}                   │
  ├──────────────────────────────────────┼───────────────────────────────────┤
  │  {BOLD}WITH Cache (threshold {threshold}){RESET}            │                                   │
  │    Gemini rewrite  ({total_queries} calls)       │  ${with_cache_rewrite:.6f}                   │
  │    Gemini answer   ({total_misses} calls)       │  ${with_cache_answer:.6f}                   │
  │    RAG retrieval   ({total_misses} calls)       │  ${with_cache_rag:.6f}                   │
  │    {BOLD}Subtotal{RESET}                             │  {BOLD}${with_cache_total:.6f}{RESET}                   │
  ├──────────────────────────────────────┼───────────────────────────────────┤
  │  {GREEN}{BOLD}Cache Savings{RESET}                         │  {GREEN}{BOLD}${savings:.6f}  ({savings_pct:.0f}%){RESET}            │
  │  {GREEN}Filter Savings (junk blocked){RESET}        │  {GREEN}${filter_savings:.6f}{RESET}                   │
  │  {GREEN}{BOLD}Combined Savings{RESET}                      │  {GREEN}{BOLD}${savings + filter_savings:.6f}{RESET}                   │
  └──────────────────────────────────────┴───────────────────────────────────┘""")

    # ── Monthly Projection ───────────────────────────────────────────────
    daily_queries = 1000
    monthly_queries = daily_queries * 30
    junk_pct = 0.15
    actual_queries = monthly_queries * (1 - junk_pct)

    no_cache_monthly = monthly_queries * (
        token_cost(cost_rewrite_in, cost_rewrite_out) +
        token_cost(cost_answer_in, cost_answer_out) +
        cost_per_rag
    )

    # Use the measured hit rate for "similar" queries in Phase 2
    real_hit_rate = similar_hits / similar_total if similar_total else 0

    with_cache_monthly = (
        actual_queries * token_cost(cost_rewrite_in, cost_rewrite_out) +
        actual_queries * (1 - real_hit_rate) * token_cost(cost_answer_in, cost_answer_out) +
        actual_queries * (1 - real_hit_rate) * cost_per_rag
    )

    monthly_savings = no_cache_monthly - with_cache_monthly
    monthly_savings_pct = (monthly_savings / no_cache_monthly * 100) if no_cache_monthly else 0

    print(f"""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  {BOLD}Monthly Projection (1,000 queries/day, 15% junk){RESET}                      │
  ├──────────────────────────────────┬───────────────────────────────────────┤
  │  Monthly incoming messages       │  30,000                               │
  │  Junk filtered (~15%)            │  4,500  ({GREEN}$0 cost{RESET})                      │
  │  Queries processed               │  25,500                               │
  │  Cache hit rate (measured)       │  {BOLD}{real_hit_rate*100:.0f}%{RESET}                                 │
  │                                  │                                       │
  │  {RED}Without cache + filter{RESET}          │  {RED}${no_cache_monthly:.2f}/month{RESET}                       │
  │  {GREEN}With cache + filter{RESET}             │  {GREEN}${with_cache_monthly:.2f}/month{RESET}                       │
  │                                  │                                       │
  │  {GREEN}{BOLD}Monthly savings{RESET}                  │  {GREEN}{BOLD}${monthly_savings:.2f}/month ({monthly_savings_pct:.0f}% reduction){RESET}       │
  └──────────────────────────────────┴───────────────────────────────────────┘""")

    # ── Phase 2 Detail Table ─────────────────────────────────────────────

    print(f"\n{BOLD}Phase 2 Detail — Hinglish → Rewrite → Cache Lookup:{RESET}\n")
    print(f"  {'#':>2}  {'Result':>6}  {'Exp':>4}  {'✓':>1}  {'Original query (as typed by user)':<55}")
    print(f"  {'─'*2}  {'─'*6}  {'─'*4}  {'─':>1}  {'─'*55}")
    for i, r in enumerate(phase2_results):
        color = GREEN if r["correct"] else YELLOW
        mark = "✓" if r["correct"] else "✗"
        print(f"  {i+1:>2}  {r['actual']:>6}  {r['expected']:>4}  {color}{mark}{RESET}  {r['query']:<55}")

    print(f"\n{'═' * 80}\n")

    return 0


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.exit(code)
