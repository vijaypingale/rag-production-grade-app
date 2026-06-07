"""
Tests for app/generation/context_assembler.py

What is tested here:
---------------------
1. Empty input          — returns safe empty result, no crash
2. Normal assembly      — citation tags [1][2]... appear correctly,
                          citation_map keys match tag numbers,
                          returned stats are accurate
3. Token budget         — when chunks exceed max_tokens, assembler
                          stops early and does not exceed the limit
4. Single chunk         — edge case: only one chunk in the list
5. Missing content      — chunks with empty content are skipped
6. Metadata in map      — citation_map carries source/page/chunk_id
                          and rerank_score correctly

These tests use NO external services (no OpenAI, no FAISS, no BM25).
All inputs are plain Python dicts — fast and runnable offline.

Run with:
    pytest tests/generation/test_context_assembler.py -v
"""

import pytest
import tiktoken

from app.generation.context_assembler import assemble_context, DEFAULT_MAX_TOKENS


# =========================================================
# Helpers
# =========================================================

def _make_chunk(content: str, source: str = "doc.pdf", page: int = 1,
                chunk_id: str = "abc123", rerank_score: float = 0.9) -> dict:
    """
    Convenience factory to build a chunk dict in the same shape
    the reranker produces. Keeps test cases concise.
    """
    return {
        "content": content,
        "metadata": {
            "source":   source,
            "page":     page,
            "chunk_id": chunk_id,
        },
        "rerank_score": rerank_score,
    }


def _token_count(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens the same way the assembler does — used to verify stats."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# =========================================================
# Test Cases
# =========================================================

class TestEmptyInput:

    def test_empty_list_returns_empty_context(self):
        result = assemble_context([])
        assert result["context"] == ""

    def test_empty_list_returns_empty_citation_map(self):
        result = assemble_context([])
        assert result["citation_map"] == {}

    def test_empty_list_returns_zero_stats(self):
        result = assemble_context([])
        assert result["chunks_used"]  == 0
        assert result["chunks_total"] == 0
        assert result["tokens_used"]  == 0


class TestNormalAssembly:

    def setup_method(self):
        """Three normal chunks used across multiple tests."""
        self.chunks = [
            _make_chunk("The sky is blue.",   source="a.pdf", page=1, chunk_id="c1", rerank_score=0.95),
            _make_chunk("Water boils at 100C.", source="b.pdf", page=2, chunk_id="c2", rerank_score=0.80),
            _make_chunk("Python is a language.", source="c.pdf", page=3, chunk_id="c3", rerank_score=0.60),
        ]
        self.result = assemble_context(self.chunks)

    def test_all_chunks_included(self):
        assert self.result["chunks_used"]  == 3
        assert self.result["chunks_total"] == 3

    def test_citation_tags_present_in_context(self):
        ctx = self.result["context"]
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "[3]" in ctx

    def test_citation_map_has_correct_keys(self):
        assert set(self.result["citation_map"].keys()) == {1, 2, 3}

    def test_citation_map_source_is_correct(self):
        cm = self.result["citation_map"]
        assert cm[1]["source"] == "a.pdf"
        assert cm[2]["source"] == "b.pdf"
        assert cm[3]["source"] == "c.pdf"

    def test_citation_map_page_is_correct(self):
        cm = self.result["citation_map"]
        assert cm[1]["page"] == 1
        assert cm[2]["page"] == 2
        assert cm[3]["page"] == 3

    def test_citation_map_chunk_id_is_correct(self):
        cm = self.result["citation_map"]
        assert cm[1]["chunk_id"] == "c1"
        assert cm[2]["chunk_id"] == "c2"
        assert cm[3]["chunk_id"] == "c3"

    def test_citation_map_rerank_score_is_correct(self):
        cm = self.result["citation_map"]
        assert cm[1]["rerank_score"] == pytest.approx(0.95)
        assert cm[2]["rerank_score"] == pytest.approx(0.80)
        assert cm[3]["rerank_score"] == pytest.approx(0.60)

    def test_chunk_content_appears_in_context(self):
        ctx = self.result["context"]
        assert "The sky is blue."      in ctx
        assert "Water boils at 100C."  in ctx
        assert "Python is a language." in ctx

    def test_tokens_used_is_positive(self):
        assert self.result["tokens_used"] > 0


class TestTokenBudget:

    def test_assembler_respects_max_tokens(self):
        """
        Force a very small token budget so only the first chunk fits.
        Verify tokens_used does not exceed the budget.
        """
        chunks = [
            _make_chunk("First chunk — should fit.",   chunk_id="c1"),
            _make_chunk("Second chunk — should not fit.", chunk_id="c2"),
            _make_chunk("Third chunk — definitely not.", chunk_id="c3"),
        ]
        # Set a tiny budget: enough for ~1 short chunk only
        tiny_budget = 15
        result = assemble_context(chunks, max_tokens=tiny_budget)

        assert result["tokens_used"] <= tiny_budget

    def test_assembler_stops_early_when_budget_exceeded(self):
        """
        With a tiny budget only 1 chunk should be included even though
        3 were offered.
        """
        chunks = [
            _make_chunk("First chunk — should fit.",   chunk_id="c1"),
            _make_chunk("Second chunk — should not fit.", chunk_id="c2"),
            _make_chunk("Third chunk — definitely not.", chunk_id="c3"),
        ]
        tiny_budget = 15
        result = assemble_context(chunks, max_tokens=tiny_budget)

        assert result["chunks_used"]  < result["chunks_total"]

    def test_chunks_total_always_reflects_input_size(self):
        """chunks_total should always equal len(input), regardless of budget."""
        chunks = [_make_chunk(f"Chunk {i}") for i in range(10)]
        result = assemble_context(chunks, max_tokens=50)
        assert result["chunks_total"] == 10


class TestEdgeCases:

    def test_single_chunk(self):
        chunks = [_make_chunk("Only one chunk here.", chunk_id="solo")]
        result = assemble_context(chunks)

        assert result["chunks_used"] == 1
        assert "[1]" in result["context"]
        assert 1 in result["citation_map"]

    def test_empty_content_chunks_are_skipped(self):
        """Chunks with empty/whitespace content should be silently skipped."""
        chunks = [
            _make_chunk("",        chunk_id="empty"),
            _make_chunk("   ",     chunk_id="whitespace"),
            _make_chunk("Real content.", chunk_id="real"),
        ]
        result = assemble_context(chunks)

        # Only the real chunk should appear
        assert result["chunks_used"] == 1
        assert "Real content." in result["context"]

    def test_context_starts_with_citation_one(self):
        """The assembled context must always start with [1]."""
        chunks = [_make_chunk("Alpha."), _make_chunk("Beta.")]
        result = assemble_context(chunks)
        assert result["context"].startswith("[1]")

    def test_default_max_tokens_is_respected(self):
        """
        Generate enough chunks to exceed DEFAULT_MAX_TOKENS and confirm
        the assembler never goes over.
        """
        # Each chunk is ~200 chars; enough chunks to overflow 6000 tokens
        long_text = "A" * 200
        chunks = [_make_chunk(long_text, chunk_id=str(i)) for i in range(100)]
        result = assemble_context(chunks)

        assert result["tokens_used"] <= DEFAULT_MAX_TOKENS
