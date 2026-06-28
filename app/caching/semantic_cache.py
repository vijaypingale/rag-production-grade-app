"""
Semantic Cache — Section 13

The idea (and why it matters in production):
--------------------------------------------
A normal cache is EXACT-match: the same key returns the same value. That fails
for LLM apps because users phrase the same intent many ways:
    "What is the WISeR model?"  ==  "Explain the WISeR model"  (different strings!)

A SEMANTIC cache keys on the query's MEANING — its embedding vector. If a new
query is close enough (cosine similarity >= threshold) to one we've answered,
we return the cached answer in milliseconds and skip retrieval + rerank + the
LLM call. In production this typically cuts LLM cost ~40-70% and tail latency
dramatically on repeat-heavy workloads (FAQ bots, agents).

How it works (same 3 steps in every implementation):
    1. Embed the incoming query into a vector.
    2. Find the nearest cached query vector (cosine similarity).
    3. If similarity >= threshold -> HIT (return cached answer); else MISS
       (run the pipeline, then store the new {query -> answer}).
Entries also carry a TTL so answers go stale-safe when source docs change.

Two backends behind one interface:
----------------------------------
- InMemorySemanticCache  : a readable reference implementation that performs the
                           3 steps explicitly. Per-process (lost on restart, not
                           shared across instances) — ideal for dev/demo + tests.
- RedisVLSemanticCache   : PRODUCTION backend. RedisVL does the exact same 3
                           steps but backed by Redis — an HNSW vector index for
                           fast nearest-neighbour, cosine distance, and TTL,
                           shared across all app instances and persistent.

Pick the backend via settings.CACHE_BACKEND. ask_service only depends on the
BaseSemanticCache interface, so swapping memory <-> redis needs no app changes.
"""

import time
import json
from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np

from app.config.settings import (
    CACHE_ENABLED,
    CACHE_BACKEND,
    SEMANTIC_CACHE_THRESHOLD,
    CACHE_TTL_SECONDS,
    REDIS_URL,
)
from app.utils.logger import logger


# =========================================================
# Similarity helper
# =========================================================

def cosine_similarity(a, b) -> float:
    """
    Cosine similarity in [-1, 1] (1 = identical direction). This is the core
    "how close in meaning" measure. RedisVL computes the equivalent server-side;
    we compute it here for the in-memory backend.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


# =========================================================
# Interface
# =========================================================

class BaseSemanticCache(ABC):
    """Common contract so the rest of the app is backend-agnostic."""

    @abstractmethod
    def lookup(self, query: str) -> Optional[dict]:
        """Return the cached payload dict for a semantically-close query, or None (MISS)."""

    @abstractmethod
    def store(self, query: str, payload: dict) -> None:
        """Cache the answer payload keyed by the query's meaning."""


# =========================================================
# In-memory backend (reference implementation + dev/tests)
# =========================================================

class InMemorySemanticCache(BaseSemanticCache):
    """
    Reference implementation that shows EXACTLY how semantic caching works.
    State lives in this process only (not shared, lost on restart), so it is
    NOT for multi-instance production — use RedisVLSemanticCache there. It is
    perfect for local demos and for deterministic unit tests (inject a fake
    embedder + clock).

    Dependencies are injected (embed_fn, now_fn) so the logic is testable
    without calling OpenAI or relying on the wall clock.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list],
        threshold: float,
        ttl_seconds: int,
        now_fn: Callable[[], float] = time.time,
    ):
        self._embed = embed_fn          # query -> embedding vector
        self._threshold = threshold     # cosine similarity needed for a HIT
        self._ttl = ttl_seconds
        self._now = now_fn
        # Each entry: {"embedding": [...], "payload": {...}, "expires_at": float}
        self._entries: list[dict] = []

    def lookup(self, query: str) -> Optional[dict]:
        query_vec = self._embed(query)          # step 1: embed the query
        now = self._now()
        best_payload, best_sim = None, -1.0

        # step 2: scan cached entries for the closest non-expired match.
        # (Linear scan here for clarity; Redis uses an HNSW index so this is
        #  sub-millisecond even with millions of entries.)
        for entry in self._entries:
            if entry["expires_at"] <= now:
                continue                        # skip expired (TTL)
            sim = cosine_similarity(query_vec, entry["embedding"])
            if sim > best_sim:
                best_sim, best_payload = sim, entry["payload"]

        # step 3: HIT only if the closest match clears the threshold.
        if best_payload is not None and best_sim >= self._threshold:
            logger.info("semantic_cache_hit", backend="memory", similarity=round(best_sim, 4))
            return best_payload

        return None  # MISS

    def store(self, query: str, payload: dict) -> None:
        self._entries.append({
            "embedding": self._embed(query),
            "payload": payload,
            "expires_at": self._now() + self._ttl,
        })


# =========================================================
# RedisVL backend (production)
# =========================================================

class RedisVLSemanticCache(BaseSemanticCache):
    """
    PRODUCTION backend using RedisVL's SemanticCache. RedisVL performs the same
    embed -> nearest-neighbour -> threshold -> TTL flow, but backed by Redis:
    an HNSW vector index for fast search, cosine distance, and server-side TTL.
    Because the data lives in Redis (not the process), the cache is SHARED
    across every app instance and survives restarts — what you need at scale.

    We store our answer payload as JSON in the cache entry, and let RedisVL
    embed the prompt and handle the vector index for us.
    """

    def __init__(self, redis_url: str, threshold: float, ttl_seconds: int,
                 name: str = "rag_semantic_cache"):
        # Lazy import so the app runs without redisvl installed when the
        # in-memory backend is selected.
        from redisvl.extensions.cache.llm import SemanticCache

        # IMPORTANT GOTCHA: RedisVL uses DISTANCE (= 1 - cosine_similarity),
        # while our config is expressed as a similarity threshold. Convert it:
        #   similarity 0.92  ->  distance 0.08
        distance_threshold = 1.0 - threshold

        self._cache = SemanticCache(
            name=name,
            redis_url=redis_url,
            distance_threshold=distance_threshold,
            ttl=ttl_seconds,
        )

    def lookup(self, query: str) -> Optional[dict]:
        hits = self._cache.check(prompt=query, num_results=1)
        if hits:
            logger.info("semantic_cache_hit", backend="redis")
            # We stored the payload as JSON in the entry's "response" field.
            return json.loads(hits[0]["response"])
        return None

    def store(self, query: str, payload: dict) -> None:
        # RedisVL embeds `prompt` and indexes it; `response` is the value we get
        # back on a hit. We put our full payload there as JSON.
        self._cache.store(prompt=query, response=json.dumps(payload))


# =========================================================
# Default embedder (used by the in-memory backend in the running app)
# =========================================================

def _default_embed(text: str) -> list:
    """
    Embed a query with the SAME model the app uses (text-embedding-3-small),
    so cache similarity is consistent with retrieval. Imported lazily to keep
    module import cheap and test-friendly.
    """
    from app.utils.openai_client import get_embedding_model
    return get_embedding_model().embed_query(text)


# =========================================================
# Factory (singleton)
# =========================================================

_cache_instance: Optional[BaseSemanticCache] = None


def get_semantic_cache() -> Optional[BaseSemanticCache]:
    """
    Return the configured cache, or None if caching is disabled / unavailable.

    - CACHE_ENABLED=false            -> None (pipeline always runs)
    - CACHE_BACKEND="redis"          -> RedisVLSemanticCache (falls back to None
                                        with a warning if Redis can't be reached,
                                        so a cache outage never breaks requests)
    - CACHE_BACKEND="memory" (default) -> InMemorySemanticCache

    Cached as a module-level singleton so we don't rebuild it per request.
    """
    global _cache_instance

    if not CACHE_ENABLED:
        return None
    if _cache_instance is not None:
        return _cache_instance

    if CACHE_BACKEND == "redis":
        try:
            _cache_instance = RedisVLSemanticCache(
                redis_url=REDIS_URL,
                threshold=SEMANTIC_CACHE_THRESHOLD,
                ttl_seconds=CACHE_TTL_SECONDS,
            )
            logger.info("semantic_cache_initialized", backend="redis")
        except Exception as exc:
            # Graceful degradation: never fail a request because the cache is
            # down. Just log and run without caching.
            logger.warning("semantic_cache_redis_init_failed", error=str(exc))
            return None
    else:
        _cache_instance = InMemorySemanticCache(
            embed_fn=_default_embed,
            threshold=SEMANTIC_CACHE_THRESHOLD,
            ttl_seconds=CACHE_TTL_SECONDS,
        )
        logger.info("semantic_cache_initialized", backend="memory")

    return _cache_instance
