"""
bot/rag_cache.py
Serverless in-memory exact-match cache — no external dependencies (no Redis).

Intercepts after query rewriting, before Vertex AI RAG retrieval.
Since the LLM rewriting step standardizes the query into a canonical format,
we simply use an exact string match for caching.

Pipeline position:
  parse → filter → rewrite query → >>>CACHE CHECK<<< → RAG → Gemini → >>>CACHE STORE<<<

Memory budget (App Engine optimised):
  - Cache entries: ~1 KB each
  - 1000 entries ≈ 1 MB — negligible for App Engine.

Env vars:
  CACHE_TTL_SECONDS            — entry expiry in seconds   (default 86400 = 24 h)
  CACHE_MAX_ENTRIES            — max stored entries         (default 1000)
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger("gohappy.rag_cache")


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    """Atomic cache hit/miss counters (safe under asyncio single-thread loop)."""
    hit_count:  int = 0
    miss_count: int = 0

    @property
    def total(self) -> int:
        return self.hit_count + self.miss_count

    @property
    def hit_rate(self) -> float:
        return self.hit_count / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "hit_count":  self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate":   round(self.hit_rate, 4),
        }


# ── In-Memory Cache ──────────────────────────────────────────────────────────

class _InMemoryCache:
    """
    Bounded in-memory cache with exact string match.
    Optimised for App Engine serverless deployment.
    """

    def __init__(self, max_entries: int, ttl: int):
        self._store: Dict[str, dict] = {}   # query → {response, created_at}
        self._max       = max_entries
        self._ttl       = ttl

    def get(self, query: str) -> Optional[dict]:
        """Return the cached response if an exact match exists."""
        self._evict_expired()
        entry = self._store.get(query)
        if entry is not None:
            return entry["response"]
        return None

    def set(self, query: str, response: dict):
        """Store an entry, evicting the oldest if at capacity."""
        self._evict_expired()
        if len(self._store) >= self._max and query not in self._store:
            oldest_key = min(self._store, key=lambda k: self._store[k]["created_at"])
            del self._store[oldest_key]

        self._store[query] = {
            "response":   response,
            "created_at": time.time(),
        }

    def clear(self):
        self._store.clear()

    def count(self) -> int:
        self._evict_expired()
        return len(self._store)

    def _evict_expired(self):
        now     = time.time()
        expired = [k for k, v in self._store.items() if now - v["created_at"] > self._ttl]
        for k in expired:
            del self._store[k]


# ── Main Cache ────────────────────────────────────────────────────────────────

class RAGCache:
    """
    Serverless in-memory cache for RAG responses.

    Flow:
        1. Exact string match search over in-memory store using the rewritten query.
        2. Return cached {answer, escalation} on HIT, or None on MISS.

    On MISS the caller runs the normal RAG→Gemini pipeline, then calls
    ``set()`` to cache the response.  Escalation responses are never cached.
    """

    def __init__(self):
        self._ttl         = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))
        self._max_entries = int(os.environ.get("CACHE_MAX_ENTRIES", "1000"))

        # Stats
        self.stats = CacheStats()

        # In-memory cache (the only store — no Redis)
        self._memory = _InMemoryCache(
            max_entries=self._max_entries,
            ttl=self._ttl,
        )

        logger.info(
            "RAGCache configured | mode=in-memory  ttl=%ds  max=%d",
            self._ttl, self._max_entries,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    async def get(self, rewritten_query: str) -> Optional[dict]:
        """
        Look up an exact matched cached response.

        Returns:
            dict  ``{answer: str, escalation: bool}``  on cache HIT.
            None  on MISS or error (pipeline proceeds normally).
        """
        result = self._memory.get(rewritten_query)
        if result is not None:
            self.stats.hit_count += 1
            logger.info("Cache HIT for: %.80s", rewritten_query)
            return result

        self.stats.miss_count += 1
        logger.debug("Cache MISS for: %.80s", rewritten_query)
        return None

    async def set(self, rewritten_query: str, response: dict) -> None:
        """
        Store a query → response pair in the cache.
        Silently skips if the response has ``escalation == True``.
        """
        if response.get("escalation", False):
            logger.debug("Skipping cache store — escalation response")
            return

        self._memory.set(rewritten_query, response)

    async def invalidate_all(self) -> None:
        """Flush every cached entry."""
        self._memory.clear()
        logger.info("Cache invalidated")

    async def get_stats(self) -> dict:
        """Return hit/miss counters and entry count."""
        stats = self.stats.to_dict()
        stats["total_cached_entries"] = self._memory.count()
        return stats
