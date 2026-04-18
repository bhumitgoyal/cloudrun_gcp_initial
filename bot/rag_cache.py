"""
bot/rag_cache.py
Serverless in-memory semantic cache — no external dependencies (no Redis).

Intercepts after query rewriting, before Vertex AI RAG retrieval.
Uses sentence-transformers (all-MiniLM-L6-v2) for embedding and brute-force
cosine similarity search over a bounded in-memory store.

Pipeline position:
  parse → filter → rewrite query → >>>CACHE CHECK<<< → RAG → Gemini → >>>CACHE STORE<<<

Memory budget (Cloud Run optimised):
  - Embedding model: ~80 MB (loaded lazily on first cache use)
  - Cache entries: ~2.5 KB each (384×4 bytes vector + query + response)
  - 500 entries ≈ 1.25 MB  |  1000 entries ≈ 2.5 MB  |  2000 entries ≈ 5 MB
  - Default 1000 entries is safe for Cloud Run with 1–2 GiB memory

Env vars:
  CACHE_SIMILARITY_THRESHOLD   — cosine similarity threshold (default 0.92)
  CACHE_TTL_SECONDS            — entry expiry in seconds   (default 86400 = 24 h)
  CACHE_MAX_ENTRIES            — max stored entries         (default 1000)
"""

import os
import time
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict
from uuid import uuid4

import numpy as np

logger = logging.getLogger("gohappy.rag_cache")

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384


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
    Bounded in-memory cache with brute-force cosine similarity search.
    Optimised for Cloud Run serverless deployment.

    Memory footprint per entry:
      - embedding: 384 × 4 bytes = 1,536 bytes
      - query + response JSON:    ~500–1,000 bytes
      - Python dict overhead:     ~200 bytes
      Total: ~2,000–2,700 bytes per entry

    1000 entries ≈ 2.5 MB — negligible for a 1–2 GiB Cloud Run instance.
    """

    def __init__(self, max_entries: int, ttl: int, threshold: float):
        self._store: Dict[str, dict] = {}   # id → {query, response, embedding, created_at}
        self._max       = max_entries
        self._ttl       = ttl
        self._threshold = threshold

    def get(self, embedding: np.ndarray) -> Optional[dict]:
        """Return the cached response if a close enough match exists."""
        self._evict_expired()
        best_sim      = -1.0
        best_response = None

        for entry in self._store.values():
            sim = self._cosine_sim(embedding, entry["embedding"])
            if sim > best_sim:
                best_sim      = sim
                best_response = entry["response"]

        if best_sim >= self._threshold and best_response is not None:
            return best_response
        return None

    def set(self, query: str, response: dict, embedding: np.ndarray):
        """Store an entry, evicting the oldest if at capacity."""
        self._evict_expired()
        if len(self._store) >= self._max:
            oldest_key = min(self._store, key=lambda k: self._store[k]["created_at"])
            del self._store[oldest_key]

        self._store[uuid4().hex] = {
            "query":      query,
            "response":   response,
            "embedding":  embedding,
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

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── Main Cache ────────────────────────────────────────────────────────────────

class RAGCache:
    """
    Serverless in-memory semantic cache for RAG responses.

    Flow:
        1. Embed the rewritten query via sentence-transformers.
        2. Cosine similarity search over in-memory store.
        3. Return cached {answer, escalation} on HIT, or None on MISS.

    On MISS the caller runs the normal RAG→Gemini pipeline, then calls
    ``set()`` to cache the response.  Escalation responses are never cached.
    """

    def __init__(self):
        self._threshold   = float(os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.92"))
        self._ttl         = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))
        self._max_entries = int(os.environ.get("CACHE_MAX_ENTRIES", "1000"))

        # Stats
        self.stats = CacheStats()

        # Embedding model — loaded lazily on first call
        self._model      = None
        self._model_lock = asyncio.Lock()

        # In-memory cache (the only store — no Redis)
        self._memory = _InMemoryCache(
            max_entries=self._max_entries,
            ttl=self._ttl,
            threshold=self._threshold,
        )

        logger.info(
            "RAGCache configured | mode=in-memory  threshold=%.2f  ttl=%ds  max=%d",
            self._threshold, self._ttl, self._max_entries,
        )

    # ── Lazy initialisation ──────────────────────────────────────────────────

    async def _ensure_model(self):
        """Load sentence-transformers model on first use."""
        if self._model is not None:
            return
        async with self._model_lock:
            if self._model is not None:
                return                                       # double-check
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(None, self._load_model)
            logger.info("Embedding model loaded (all-MiniLM-L6-v2, dim=%d)", EMBEDDING_DIM)

    @staticmethod
    def _load_model():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")

    # ── Embedding ────────────────────────────────────────────────────────────

    def _embed_sync(self, text: str) -> np.ndarray:
        """CPU-bound — always called via ``run_in_executor``."""
        return self._model.encode(text, normalize_embeddings=True)

    async def _embed(self, text: str) -> np.ndarray:
        """Async wrapper — compute embedding in the default thread pool."""
        await self._ensure_model()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, text)

    # ── Public API ───────────────────────────────────────────────────────────

    async def get(self, rewritten_query: str) -> Optional[dict]:
        """
        Look up a semantically similar cached response.

        Returns:
            dict  ``{answer: str, escalation: bool}``  on cache HIT.
            None  on MISS or error (pipeline proceeds normally).
        """
        try:
            embedding = await self._embed(rewritten_query)
        except Exception as exc:
            logger.error("Cache get — embedding failed: %s", exc)
            self.stats.miss_count += 1
            return None

        result = self._memory.get(embedding)
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

        try:
            embedding = await self._embed(rewritten_query)
        except Exception as exc:
            logger.error("Cache set — embedding failed: %s", exc)
            return

        self._memory.set(rewritten_query, response, embedding)

    async def invalidate_all(self) -> None:
        """Flush every cached entry."""
        self._memory.clear()
        logger.info("Cache invalidated")

    async def get_stats(self) -> dict:
        """Return hit/miss counters and entry count."""
        stats = self.stats.to_dict()
        stats["total_cached_entries"] = self._memory.count()
        return stats
