"""
Tests for app/generation/grounding.py

Scope:
------
These tests cover the DETERMINISTIC parts of Section 9:
  - enforce_citations()  — pure string/regex logic, no LLM
  - _parse_judge_json()  — defensive JSON parsing of judge output

The faithfulness check itself (check_faithfulness) makes a live LLM call,
so it is NOT unit-tested here — that belongs to the evaluation layer
(Section 10) where LLM behaviour is measured with scores, not asserted.
This split mirrors the core principle: deterministic code → unit tests,
LLM-dependent quality → evals.

Run with:
    pytest tests/generation/test_grounding.py -v
"""

import pytest

from app.generation.grounding import (
    enforce_citations,
    CitationAudit,
    _parse_judge_json,
)


# =========================================================
# Citation Enforcement
# =========================================================

class TestEnforceCitations:

    def _map(self, *nums):
        """Build a citation_map with the given citation numbers as keys."""
        return {n: {"source": f"doc{n}.pdf"} for n in nums}

    def test_all_citations_valid(self):
        answer = "The sky is blue [1] and water boils at 100C [2]."
        audit = enforce_citations(answer, self._map(1, 2, 3))
        assert audit.all_valid is True
        assert audit.cited_numbers == [1, 2]
        assert audit.valid_numbers == [1, 2]
        assert audit.orphan_numbers == []

    def test_orphan_citation_detected(self):
        # [7] is not in the citation_map → orphan (a provenance hallucination)
        answer = "This claim is supported [1] but this one is invented [7]."
        audit = enforce_citations(answer, self._map(1, 2, 3))
        assert audit.all_valid is False
        assert audit.orphan_numbers == [7]
        assert audit.valid_numbers == [1]

    def test_no_citations_in_answer(self):
        answer = "This answer has no citation markers at all."
        audit = enforce_citations(answer, self._map(1, 2))
        assert audit.cited_numbers == []
        assert audit.orphan_numbers == []
        assert audit.all_valid is True

    def test_duplicate_citations_deduped(self):
        answer = "Point one [1]. Repeated [1]. Also point two [2]."
        audit = enforce_citations(answer, self._map(1, 2))
        assert audit.cited_numbers == [1, 2]   # deduped + sorted

    def test_citations_sorted(self):
        answer = "Out of order [3] then [1] then [2]."
        audit = enforce_citations(answer, self._map(1, 2, 3))
        assert audit.cited_numbers == [1, 2, 3]

    def test_multiple_orphans(self):
        answer = "Valid [1], invented [8], also invented [9]."
        audit = enforce_citations(answer, self._map(1, 2))
        assert audit.orphan_numbers == [8, 9]
        assert audit.all_valid is False

    def test_multi_digit_citations(self):
        answer = "A claim with a two-digit marker [12]."
        audit = enforce_citations(answer, self._map(12))
        assert audit.cited_numbers == [12]
        assert audit.all_valid is True

    def test_returns_citation_audit_type(self):
        audit = enforce_citations("text [1]", self._map(1))
        assert isinstance(audit, CitationAudit)


# =========================================================
# Judge JSON Parsing (defensive)
# =========================================================

class TestParseJudgeJson:

    def test_clean_json(self):
        raw = '{"claims": [{"claim": "x", "verdict": "supported"}]}'
        claims = _parse_judge_json(raw)
        assert claims == [{"claim": "x", "verdict": "supported"}]

    def test_json_wrapped_in_markdown_fence(self):
        raw = '```json\n{"claims": [{"claim": "y", "verdict": "unsupported"}]}\n```'
        claims = _parse_judge_json(raw)
        assert claims == [{"claim": "y", "verdict": "unsupported"}]

    def test_json_with_surrounding_prose(self):
        raw = 'Here is my analysis:\n{"claims": []}\nHope that helps!'
        claims = _parse_judge_json(raw)
        assert claims == []

    def test_invalid_json_returns_none(self):
        raw = "this is not json at all"
        assert _parse_judge_json(raw) is None

    def test_missing_claims_key_returns_none(self):
        raw = '{"something_else": 123}'
        assert _parse_judge_json(raw) is None

    def test_malformed_json_returns_none(self):
        raw = '{"claims": [{"claim": "x", "verdict": '   # truncated
        assert _parse_judge_json(raw) is None
