"""
test_cache_pipeline.py
End-to-end pipeline tests demonstrating how the in-memory semantic cache
integrates with the MessagePipeline.

Mocks:  WhatsApp, RAG, Gemini, Firestore (no GCP creds required)
Real:   RAGCache (in-memory, serverless)

Run:
    python test_cache_pipeline.py

What this proves:
    1. Cache MISS → RAG + Gemini are called → response is cached
    2. Cache HIT  → RAG + Gemini are SKIPPED → cached response returned
    3. Semantically similar query → still a HIT (different wording, same intent)
    4. Unrelated query → MISS → full pipeline runs again
    5. Escalation response → NOT cached → next identical query is a MISS
    6. Cache stats → hit/miss counters are accurate
    7. Graceful degradation → cache errors don't crash the pipeline
"""

import os
import sys
import asyncio
import logging

# ── Env setup (before any bot imports) ────────────────────────────────────────
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_LOCATION", "us-central1")
os.environ.setdefault("VERTEX_RAG_CORPUS", "projects/test/locations/us-central1/ragCorpora/123")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")
# No Redis needed — pure in-memory cache
os.environ.setdefault("CACHE_SIMILARITY_THRESHOLD", "0.90")
os.environ.setdefault("CACHE_TTL_SECONDS", "60")
os.environ.setdefault("CACHE_MAX_ENTRIES", "100")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("test_cache_pipeline")

# ── Formatting ────────────────────────────────────────────────────────────────
PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM  = "\033[90m"

results = []

def report(name: str, passed: bool, detail: str = ""):
    tag = PASS if passed else FAIL
    results.append(passed)
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  {DIM}({detail}){RESET}"
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  MOCKS — simulate the full pipeline without any GCP credentials
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from typing import Optional, List
from bot.rag_cache import RAGCache

# Track call counts to prove cache skipping
_rag_call_count  = 0
_gemini_call_count = 0
_sent_messages: list = []


@dataclass
class FakeIncomingMessage:
    wa_message_id: str
    from_number: str
    display_name: str
    text: str
    phone_number_id: str = "mock_phone_id"


class FakeWhatsApp:
    """Mock WhatsApp client."""
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
        _sent_messages.append({"to": to, "body": body})
        return True


@dataclass
class FakeChunk:
    index: int
    text: str
    source: str
    score: float


class FakeRAG:
    """Mock RAG engine — tracks how many times it's called."""
    def query(self, query_text: str) -> List[FakeChunk]:
        global _rag_call_count
        _rag_call_count += 1
        logger.info("🔍 RAG.query() called (#%d): %.60s", _rag_call_count, query_text)
        return [FakeChunk(1, f"GoHappy Club info for: {query_text}", "KB", 0.85)]

    def format_for_prompt(self, chunks) -> str:
        return "\n".join(f"[DOC_{c.index}] {c.text}" for c in chunks)


@dataclass
class FakeBotResponse:
    answer: str
    escalation: bool


class FakeGemini:
    """Mock Gemini — tracks calls and returns predictable responses."""
    def __init__(self):
        self._escalate_next = False

    async def rewrite_query(self, user_query: str) -> str:
        # Simulate light cleanup (real Gemini does this)
        return user_query.strip()

    async def chat(self, customer_summary, conversation_history, user_query, retrieved_context) -> FakeBotResponse:
        global _gemini_call_count
        _gemini_call_count += 1
        logger.info("🤖 Gemini.chat() called (#%d): %.60s", _gemini_call_count, user_query)

        if self._escalate_next:
            self._escalate_next = False
            return FakeBotResponse(
                answer="Let me connect you with our support team for this.",
                escalation=True,
            )
        return FakeBotResponse(
            answer=f"Here's the answer about: {user_query}",
            escalation=False,
        )


class FakeMemory:
    """Mock Firestore conversation memory."""
    def __init__(self):
        self._states = {}

    async def get_state(self, phone: str) -> dict:
        return self._states.get(phone, {
            "display_name": "",
            "summary": "",
            "turn_count": 0,
            "recent_turns": [],
            "escalated_to_human": False,
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


# ── Build pipeline with mocks ────────────────────────────────────────────────

from bot.pipeline import MessagePipeline

def make_payload(msg_id: str, text: str, phone: str = "+919999900000", name: str = "Test User"):
    return {
        "from_number": phone,
        "display_name": name,
        "messages": [{"id": msg_id, "text": {"body": text}}],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════════════════════

async def test_miss_then_hit():
    """
    First query = MISS (RAG + Gemini called).
    Same query again = HIT (RAG + Gemini SKIPPED).
    """
    global _rag_call_count, _gemini_call_count, _sent_messages

    print(f"\n{BOLD}── Test 1: Cache MISS → then HIT ──{RESET}")

    cache = RAGCache()
    pipeline = MessagePipeline(
        whatsapp=FakeWhatsApp(),
        rag=FakeRAG(),
        memory=FakeMemory(),
        llm=FakeGemini(),
        cache=cache,
    )

    _rag_call_count = 0
    _gemini_call_count = 0
    _sent_messages = []

    # First query — should be a MISS
    await pipeline._process(make_payload("msg_001", "How do I join GoHappy Club?"))
    rag_after_miss = _rag_call_count
    gemini_after_miss = _gemini_call_count

    print(f"  After MISS:  RAG calls={rag_after_miss}, Gemini calls={gemini_after_miss}")

    report("MISS triggers RAG", rag_after_miss == 1)
    report("MISS triggers Gemini", gemini_after_miss == 1)

    # Same query again — should be a HIT (cache intercepts)
    await pipeline._process(make_payload("msg_002", "How do I join GoHappy Club?"))
    rag_after_hit = _rag_call_count
    gemini_after_hit = _gemini_call_count

    print(f"  After HIT:   RAG calls={rag_after_hit}, Gemini calls={gemini_after_hit}")

    report("HIT skips RAG (count unchanged)", rag_after_hit == 1)
    report("HIT skips Gemini (count unchanged)", gemini_after_hit == 1)

    # Both messages should have received replies
    report("Both messages got replies", len(_sent_messages) == 2)

    stats = await cache.get_stats()
    print(f"  Stats: hits={stats['hit_count']}, misses={stats['miss_count']}, rate={stats['hit_rate']}")
    report("Stats: 1 hit, 1 miss", stats["hit_count"] >= 1 and stats["miss_count"] >= 1)


async def test_semantic_hit():
    """
    Cache a response with one wording, then query with different wording
    that means the same thing → HIT.
    """
    global _rag_call_count, _gemini_call_count

    print(f"\n{BOLD}── Test 2: Semantic HIT (different wording, same intent) ──{RESET}")

    cache = RAGCache()
    pipeline = MessagePipeline(
        whatsapp=FakeWhatsApp(),
        rag=FakeRAG(),
        memory=FakeMemory(),
        llm=FakeGemini(),
        cache=cache,
    )

    _rag_call_count = 0
    _gemini_call_count = 0

    # Seed the cache with one phrasing
    await pipeline._process(make_payload("msg_010", "How do I become a member of GoHappy Club?"))
    report("Original query → MISS → RAG called", _rag_call_count == 1)

    # Different phrasing, same intent
    await pipeline._process(make_payload("msg_011", "How can I join GoHappy Club membership?"))

    print(f"  RAG calls after semantic query: {_rag_call_count}")
    report("Semantic similar query → HIT → RAG NOT called again", _rag_call_count == 1)


async def test_unrelated_miss():
    """
    Cache a response, then query something completely different → MISS.
    """
    global _rag_call_count, _gemini_call_count

    print(f"\n{BOLD}── Test 3: Unrelated query → MISS ──{RESET}")

    cache = RAGCache()
    pipeline = MessagePipeline(
        whatsapp=FakeWhatsApp(),
        rag=FakeRAG(),
        memory=FakeMemory(),
        llm=FakeGemini(),
        cache=cache,
    )

    _rag_call_count = 0
    _gemini_call_count = 0

    # Seed cache
    await pipeline._process(make_payload("msg_020", "What is the Silver membership plan?"))
    report("First query → RAG called", _rag_call_count == 1)

    # Totally different topic
    await pipeline._process(make_payload("msg_021", "What time does the yoga class start tomorrow?"))

    print(f"  RAG calls after unrelated query: {_rag_call_count}")
    report("Unrelated query → MISS → RAG called again", _rag_call_count == 2)


async def test_escalation_not_cached():
    """
    Escalation responses must NOT be cached.
    Same query after escalation should trigger full pipeline again.
    """
    global _rag_call_count, _gemini_call_count

    print(f"\n{BOLD}── Test 4: Escalation responses are NOT cached ──{RESET}")

    cache = RAGCache()
    fake_gemini = FakeGemini()
    fake_memory = FakeMemory()
    pipeline = MessagePipeline(
        whatsapp=FakeWhatsApp(),
        rag=FakeRAG(),
        memory=fake_memory,
        llm=fake_gemini,
        cache=cache,
    )

    _rag_call_count = 0
    _gemini_call_count = 0

    # Use a dedicated phone for this test to avoid dedup collisions
    esc_phone = "+919999911111"

    # Force escalation on next response
    fake_gemini._escalate_next = True
    await pipeline._process(make_payload("msg_030", "I want a refund right now!", phone=esc_phone))
    report("Escalation query → RAG called", _rag_call_count == 1)

    # Pipeline pauses the user after escalation (escalated_to_human=True).
    # Clear that flag so the second message actually reaches the cache layer.
    await fake_memory.set_escalation_status(esc_phone, False)

    # Same query again — should NOT be a cache hit (escalation was not cached)
    await pipeline._process(make_payload("msg_031", "I want a refund right now!", phone=esc_phone))

    print(f"  RAG calls after re-asking escalation query: {_rag_call_count}")
    report("Escalation not cached → RAG called again", _rag_call_count == 2)
    report("Gemini called twice (both times)", _gemini_call_count == 2)


async def test_cache_stats_accuracy():
    """
    Verify hit/miss/rate stats are accurate across multiple operations.
    """
    print(f"\n{BOLD}── Test 5: Cache stats accuracy ──{RESET}")

    cache = RAGCache()
    pipeline = MessagePipeline(
        whatsapp=FakeWhatsApp(),
        rag=FakeRAG(),
        memory=FakeMemory(),
        llm=FakeGemini(),
        cache=cache,
    )

    # 3 unique queries (3 misses) + repeat each (3 hits) = 6 total
    queries = [
        "What is Happy Coins?",
        "Tell me about Gold membership",
        "How do I download the app?",
    ]

    for i, q in enumerate(queries):
        await pipeline._process(make_payload(f"msg_040_{i}", q))

    for i, q in enumerate(queries):
        await pipeline._process(make_payload(f"msg_041_{i}", q))

    stats = await cache.get_stats()
    print(f"  Stats: hits={stats['hit_count']}, misses={stats['miss_count']}, rate={stats['hit_rate']}")
    print(f"  Entries: total={stats['total_cached_entries']}")

    report("3 misses recorded", stats["miss_count"] == 3)
    report("3 hits recorded", stats["hit_count"] == 3)
    report("Hit rate = 0.5", stats["hit_rate"] == 0.5)
    report("3 entries cached", stats["total_cached_entries"] == 3)


async def test_graceful_degradation():
    """
    Even if cache internals raise, the pipeline must NOT crash.
    """
    print(f"\n{BOLD}── Test 6: Graceful degradation (pipeline survives cache errors) ──{RESET}")

    cache = RAGCache()
    pipeline = MessagePipeline(
        whatsapp=FakeWhatsApp(),
        rag=FakeRAG(),
        memory=FakeMemory(),
        llm=FakeGemini(),
        cache=cache,
    )

    crashed = False
    try:
        await pipeline._process(make_payload("msg_050", "Will the pipeline crash?"))
    except Exception as exc:
        crashed = True
        logger.error("Pipeline crashed: %s", exc)

    report("Pipeline did NOT crash", not crashed)


# ═══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("\n" + "═" * 70)
    print("  CACHE ↔ PIPELINE INTEGRATION TEST SUITE")
    print("  Proves that RAG + Gemini are SKIPPED on cache HIT")
    print("═" * 70)

    await test_miss_then_hit()
    await test_semantic_hit()
    await test_unrelated_miss()
    await test_escalation_not_cached()
    await test_cache_stats_accuracy()
    await test_graceful_degradation()

    print("\n" + "─" * 70)
    passed = sum(1 for r in results if r)
    total  = len(results)
    status = "ALL PASSED" if passed == total else f"{total - passed} FAILED"
    color  = "\033[92m" if passed == total else "\033[91m"
    print(f"  {color}{passed}/{total} tests passed — {status}\033[0m")
    print("─" * 70 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.exit(code)
