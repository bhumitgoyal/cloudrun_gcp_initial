"""
test_rag_cache.py
Tests for in-memory semantic cache AND message filtering.

Covers:
  CACHE:
  - In-memory cache hit / miss / threshold behaviour
  - Semantically similar queries → HIT
  - Escalation responses are never cached
  - Cache stats tracking
  - invalidate_all() clears all entries
  - TTL expiry

  FILTER:
  - Facebook/Instagram/YouTube links → BLOCKED
  - Pure URLs → BLOCKED
  - Emoji-only → BLOCKED
  - Greeting-only → BLOCKED
  - Normal questions → PASS
  - Question with link → PASS (link with real question text)
  - Short messages → BLOCKED

Run:
  python test_rag_cache.py
"""

import os
import sys
import asyncio
import time
import logging

# ── Env setup ────────────────────────────────────────────────────────────────
os.environ.setdefault("CACHE_SIMILARITY_THRESHOLD", "0.90")
os.environ.setdefault("CACHE_TTL_SECONDS", "60")
os.environ.setdefault("CACHE_MAX_ENTRIES", "100")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("test_rag_cache")

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
BOLD = "\033[1m"
DIM  = "\033[90m"
RESET = "\033[0m"

results = []

def report(name: str, passed: bool, detail: str = ""):
    tag = PASS if passed else FAIL
    results.append(passed)
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  {DIM}({detail}){RESET}"
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  CACHE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

async def test_cache_hit():
    """Store a response, then retrieve with the same query → HIT."""
    from bot.rag_cache import RAGCache

    cache = RAGCache()
    query    = "How do I join GoHappy Club?"
    response = {"answer": "You can join via the app!", "escalation": False}

    await cache.set(query, response)
    result = await cache.get(query)

    report(
        "Cache HIT (identical query)",
        result is not None and result["answer"] == response["answer"],
        f"got: {result}",
    )


async def test_cache_semantic_hit():
    """Store a response, then retrieve with semantically similar query → HIT."""
    from bot.rag_cache import RAGCache

    cache = RAGCache()
    query1   = "How do I become a member of GoHappy Club?"
    query2   = "How can I join GoHappy Club membership?"
    response = {"answer": "Download the app and sign up!", "escalation": False}

    await cache.set(query1, response)
    result = await cache.get(query2)

    report(
        "Cache HIT (semantically similar query)",
        result is not None and result["answer"] == response["answer"],
        f"sim query hit: {result is not None}",
    )


async def test_cache_miss():
    """Query something completely different from cached entries → MISS."""
    from bot.rag_cache import RAGCache

    cache = RAGCache()
    query1   = "What is the Silver membership plan?"
    response = {"answer": "Silver plan costs 999 per year.", "escalation": False}
    query2   = "What time does the yoga class start tomorrow?"

    await cache.set(query1, response)
    result = await cache.get(query2)

    report(
        "Cache MISS (unrelated query)",
        result is None,
        f"got: {result}",
    )


async def test_escalation_not_cached():
    """Escalation responses must never be stored in cache."""
    from bot.rag_cache import RAGCache

    cache = RAGCache()
    query    = "I want a refund right now!"
    response = {"answer": "Let me connect you with our team.", "escalation": True}

    await cache.set(query, response)
    result = await cache.get(query)

    report(
        "Escalation response NOT cached",
        result is None,
        "escalation=True correctly skipped",
    )


async def test_cache_stats():
    """Hit and miss counters should track correctly."""
    from bot.rag_cache import RAGCache

    cache = RAGCache()
    query    = "Tell me about Happy Coins"
    response = {"answer": "Happy Coins are rewards you earn.", "escalation": False}

    await cache.get("random unrelated question about weather")
    await cache.set(query, response)
    await cache.get(query)   # should be HIT

    stats = await cache.get_stats()

    report(
        "Cache stats tracking",
        stats["miss_count"] >= 1 and stats["hit_count"] >= 1,
        f"hits={stats['hit_count']} misses={stats['miss_count']} rate={stats['hit_rate']}",
    )


async def test_invalidate_all():
    """After invalidation, previously cached entries should miss."""
    from bot.rag_cache import RAGCache

    cache = RAGCache()
    query    = "What sessions are available?"
    response = {"answer": "We have yoga, music, and digital skills sessions.", "escalation": False}

    await cache.set(query, response)
    result_before = await cache.get(query)
    await cache.invalidate_all()
    result_after = await cache.get(query)

    report(
        "invalidate_all() clears cache",
        result_before is not None and result_after is None,
        f"before={result_before is not None}, after={result_after is None}",
    )


async def test_ttl_expiry():
    """Entries should expire after TTL."""
    from bot.rag_cache import RAGCache

    old_ttl = os.environ.get("CACHE_TTL_SECONDS", "")
    os.environ["CACHE_TTL_SECONDS"] = "1"

    cache = RAGCache()
    query    = "How do I cancel my membership?"
    response = {"answer": "Go to Settings > Cancel.", "escalation": False}

    await cache.set(query, response)
    result_before = await cache.get(query)

    await asyncio.sleep(1.5)
    result_after = await cache.get(query)

    os.environ["CACHE_TTL_SECONDS"] = old_ttl or "60"

    report(
        "TTL expiry (entries expire after CACHE_TTL_SECONDS)",
        result_before is not None and result_after is None,
        f"before_ttl={result_before is not None}, after_ttl={result_after is None}",
    )


async def test_no_redis_references():
    """Verify the cache module has no Redis imports or references."""
    import inspect
    from bot import rag_cache

    source = inspect.getsource(rag_cache)
    has_redis = "redis" in source.lower() and "no redis" not in source.lower().split("redis")[0][-20:]
    # More accurate: check for actual import statements
    has_redis_import = "import redis" in source

    report(
        "No Redis imports in rag_cache.py",
        not has_redis_import,
        "clean in-memory only",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MESSAGE FILTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_filter_facebook_link():
    """Facebook links should be blocked."""
    from bot.message_filter import filter_message

    cases = [
        "https://www.facebook.com/share/some-post-id",
        "https://fb.watch/abc123/",
        "Check this out https://www.facebook.com/gohappyclub/posts/12345",
    ]
    all_blocked = all(not filter_message(c).is_actionable for c in cases)
    report("Facebook links → BLOCKED", all_blocked)


def test_filter_instagram_link():
    """Instagram links should be blocked."""
    from bot.message_filter import filter_message

    cases = [
        "https://www.instagram.com/reel/abc123/",
        "https://instagr.am/p/xyz789/",
    ]
    all_blocked = all(not filter_message(c).is_actionable for c in cases)
    report("Instagram links → BLOCKED", all_blocked)


def test_filter_youtube_link():
    """YouTube links should be blocked."""
    from bot.message_filter import filter_message

    cases = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
    ]
    all_blocked = all(not filter_message(c).is_actionable for c in cases)
    report("YouTube links → BLOCKED", all_blocked)


def test_filter_random_url():
    """Random URLs (without question text) should be blocked."""
    from bot.message_filter import filter_message

    cases = [
        "https://some-random-website.com/article/12345",
        "http://news.example.com/breaking-story",
        "https://bit.ly/3xYz123",
    ]
    all_blocked = all(not filter_message(c).is_actionable for c in cases)
    report("Random URLs (link-only) → BLOCKED", all_blocked)


def test_filter_emoji_only():
    """Emoji-only messages should be blocked."""
    from bot.message_filter import filter_message

    cases = [
        "🙏🙏🙏",
        "😀😀",
        "🌺🌸💐",
        "👍",
    ]
    all_blocked = all(not filter_message(c).is_actionable for c in cases)
    report("Emoji-only messages → BLOCKED", all_blocked)


def test_filter_greetings():
    """Greeting-only messages (common senior forwards) should be blocked."""
    from bot.message_filter import filter_message

    cases = [
        "Good morning",
        "Good Morning 🙏",
        "good morning!!",
        "Jai Shri Krishna",
        "Namaste 🙏🌸",
        "Ram Ram",
        "Shubh Prabhat",
        "Good night 🌙",
    ]
    all_blocked = all(not filter_message(c).is_actionable for c in cases)
    details = []
    for c in cases:
        r = filter_message(c)
        if r.is_actionable:
            details.append(f"MISSED: '{c}'")
    report(
        "Greeting-only messages → BLOCKED",
        all_blocked,
        ", ".join(details) if details else "all blocked",
    )


def test_filter_too_short():
    """Single character or empty messages should be blocked."""
    from bot.message_filter import filter_message

    cases = ["", " ", "k", ".", "!"]
    all_blocked = all(not filter_message(c).is_actionable for c in cases)
    report("Too short / empty → BLOCKED", all_blocked)


def test_filter_normal_questions_pass():
    """Normal questions should PASS through the filter."""
    from bot.message_filter import filter_message

    cases = [
        "How do I join GoHappy Club?",
        "What is the Silver membership plan?",
        "membership kaise leni hai",
        "Tell me about Happy Coins",
        "mujhe yoga class ka time chahiye",
        "How to download the app?",
        "What are the trip options for Shimla?",
        "can I get a refund",
        "I need help with my account",
        "kya gold plan mein recording milti hai",
    ]
    all_passed = all(filter_message(c).is_actionable for c in cases)
    details = []
    for c in cases:
        r = filter_message(c)
        if not r.is_actionable:
            details.append(f"WRONGLY BLOCKED: '{c}' ({r.reason})")
    report(
        "Normal questions → PASS",
        all_passed,
        ", ".join(details) if details else "all passed",
    )


def test_filter_question_with_link_passes():
    """A question that includes a link should still PASS (the question is real)."""
    from bot.message_filter import filter_message

    cases = [
        "Is this the right link to download the app? https://play.google.com/store/apps/details?id=com.gohappy",
        "I saw this on the website https://gohappyclub.in/membership but what is the Gold plan price?",
    ]
    all_passed = all(filter_message(c).is_actionable for c in cases)
    report(
        "Question + link → PASS (real question with URL)",
        all_passed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("\n" + "═" * 60)
    print("  CACHE + FILTER TEST SUITE")
    print("═" * 60)

    print(f"\n{BOLD}── In-Memory Cache Tests ──{RESET}\n")
    await test_cache_hit()
    await test_cache_semantic_hit()
    await test_cache_miss()
    await test_escalation_not_cached()
    await test_cache_stats()
    await test_invalidate_all()
    await test_ttl_expiry()
    await test_no_redis_references()

    print(f"\n{BOLD}── Message Filter Tests ──{RESET}\n")
    test_filter_facebook_link()
    test_filter_instagram_link()
    test_filter_youtube_link()
    test_filter_random_url()
    test_filter_emoji_only()
    test_filter_greetings()
    test_filter_too_short()
    test_filter_normal_questions_pass()
    test_filter_question_with_link_passes()

    print("\n" + "─" * 60)
    passed = sum(1 for r in results if r)
    total  = len(results)
    status = "ALL PASSED" if passed == total else f"{total - passed} FAILED"
    color  = "\033[92m" if passed == total else "\033[91m"
    print(f"  {color}{passed}/{total} tests passed — {status}\033[0m")
    print("─" * 60 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.exit(code)
