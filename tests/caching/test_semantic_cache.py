"""
Tests for app/caching/semantic_cache.py

These test the DETERMINISTIC caching logic via the in-memory backend with an
injected fake embedder and fake clock — no Redis, no OpenAI, no wall clock.
This is exactly how you unit-test cache behaviour: control the inputs so HIT/
MISS/TTL outcomes are predictable.

The RedisVL backend is NOT unit-tested here (it needs a live Redis); it shares
the same interface and is exercised in the live demo.

Run with:
    pytest tests/caching/test_semantic_cache.py -v
"""

import pytest

from app.caching.semantic_cache import (
    InMemorySemanticCache,
    cosine_similarity,
)


# ---------------------------------------------------------
# Fake embedder: maps known phrases to fixed vectors so we
# control similarity precisely.
#   "what is wiser"            -> [1, 0, 0]
#   "explain the wiser model"  -> [0.98, 0.2, 0]  (very close -> ~0.98 sim)
#   "cookie recipe"            -> [0, 0, 1]        (orthogonal -> 0.0 sim)
# ---------------------------------------------------------
_VECTORS = {
    "what is wiser":           [1.0, 0.0, 0.0],
    "explain the wiser model": [0.98, 0.2, 0.0],
    "cookie recipe":           [0.0, 0.0, 1.0],
}


def fake_embed(text: str) -> list:
    return _VECTORS[text.lower()]


class FakeClock:
    """Controllable clock so TTL is deterministic (no sleep)."""
    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float):
        self.t += seconds


def _make_cache(threshold=0.92, ttl=86400, clock=None):
    return InMemorySemanticCache(
        embed_fn=fake_embed,
        threshold=threshold,
        ttl_seconds=ttl,
        now_fn=clock or FakeClock(),
    )


# =========================================================
# cosine_similarity
# =========================================================

class TestCosineSimilarity:

    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 0, 1]) == pytest.approx(0.0)

    def test_close_vectors_high_similarity(self):
        assert cosine_similarity([1, 0, 0], [0.98, 0.2, 0]) > 0.95


# =========================================================
# HIT / MISS behaviour
# =========================================================

class TestHitMiss:

    def test_exact_same_query_hits(self):
        cache = _make_cache()
        cache.store("What is WISeR", {"answer": "cached"})
        assert cache.lookup("What is WISeR") == {"answer": "cached"}

    def test_semantically_similar_query_hits(self):
        # different words, same meaning -> should HIT above the 0.92 threshold
        cache = _make_cache(threshold=0.92)
        cache.store("What is WISeR", {"answer": "cached"})
        hit = cache.lookup("Explain the WISeR model")
        assert hit == {"answer": "cached"}

    def test_unrelated_query_misses(self):
        cache = _make_cache(threshold=0.92)
        cache.store("What is WISeR", {"answer": "cached"})
        assert cache.lookup("cookie recipe") is None

    def test_empty_cache_misses(self):
        cache = _make_cache()
        assert cache.lookup("What is WISeR") is None

    def test_threshold_is_respected(self):
        # With a stricter threshold (0.99), the 0.98-similar query should MISS
        cache = _make_cache(threshold=0.99)
        cache.store("What is WISeR", {"answer": "cached"})
        assert cache.lookup("Explain the WISeR model") is None


# =========================================================
# TTL behaviour
# =========================================================

class TestTTL:

    def test_entry_valid_before_expiry(self):
        clock = FakeClock()
        cache = _make_cache(ttl=100, clock=clock)
        cache.store("What is WISeR", {"answer": "cached"})
        clock.advance(50)  # still within TTL
        assert cache.lookup("What is WISeR") == {"answer": "cached"}

    def test_entry_expires_after_ttl(self):
        clock = FakeClock()
        cache = _make_cache(ttl=100, clock=clock)
        cache.store("What is WISeR", {"answer": "cached"})
        clock.advance(101)  # past TTL
        assert cache.lookup("What is WISeR") is None


# =========================================================
# Payload integrity
# =========================================================

class TestPayload:

    def test_payload_returned_unchanged(self):
        cache = _make_cache()
        payload = {"answer": "A", "citations": [{"n": 1}], "trustworthy": True}
        cache.store("What is WISeR", payload)
        assert cache.lookup("What is WISeR") == payload
