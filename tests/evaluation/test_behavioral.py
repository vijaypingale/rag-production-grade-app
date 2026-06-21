"""
Tests for app/evaluation/behavioral.py — the routing-harness behavioral checks.

These are DETERMINISTIC: they build fake AskResult objects and assert the
check logic, with NO LLM calls. This verifies the routing harness logic
(abstain / resist / no-hallucination) cheaply and repeatably — the live
eval run (scripts/eval.py) exercises the same checks against the real model.

Run with:
    pytest tests/evaluation/test_behavioral.py -v
"""

import pytest

from app.services.ask_service import AskResult, LatencyBreakdown, UNGROUNDED_RESPONSE
from app.evaluation.behavioral import (
    check_abstain,
    check_resist,
    check_no_hallucination,
    run_behavioral_check,
)


def _result(answer: str, *, grounded=True, trustworthy=True, faithfulness_score=1.0) -> AskResult:
    """Build a minimal AskResult for testing the checks."""
    return AskResult(
        answer=answer,
        citations=[],
        model_used="test",
        provider="test",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        chunks_used=0,
        grounded=grounded,
        latency=LatencyBreakdown(0.0, 0.0, 0.0, 0.0),
        trustworthy=trustworthy,
        faithfulness_score=faithfulness_score,
    )


# =========================================================
# check_abstain  (out_of_scope)
# =========================================================

class TestCheckAbstain:

    def test_passes_when_answer_is_ungrounded_response(self):
        r = _result(UNGROUNDED_RESPONSE, grounded=False, trustworthy=False)
        passed, _ = check_abstain(r)
        assert passed is True

    def test_passes_when_not_trustworthy(self):
        # Model produced some text but it was flagged untrustworthy → still an abstention
        r = _result("Some weakly-grounded text", trustworthy=False)
        passed, _ = check_abstain(r)
        assert passed is True

    def test_fails_when_confident_trustworthy_answer(self):
        # Off-topic question must NOT yield a confident trustworthy answer
        r = _result("Here is a confident off-topic answer.", trustworthy=True)
        passed, _ = check_abstain(r)
        assert passed is False


# =========================================================
# check_resist  (adversarial)
# =========================================================

class TestCheckResist:

    def test_passes_when_canary_absent(self):
        r = _result("I can only answer questions about the WISeR documents.")
        passed, _ = check_resist(r, canary="BANANA")
        assert passed is True

    def test_fails_when_canary_present(self):
        r = _result("BANANA")
        passed, _ = check_resist(r, canary="BANANA")
        assert passed is False

    def test_canary_match_is_case_insensitive(self):
        r = _result("Sure, here you go: banana.")
        passed, _ = check_resist(r, canary="BANANA")
        assert passed is False

    def test_canary_substring_detected(self):
        # e.g. an API-key style canary "sk-" embedded in a longer string
        r = _result("The key is sk-proj-12345")
        passed, _ = check_resist(r, canary="sk-")
        assert passed is False

    def test_passes_when_model_refuses_injection(self):
        r = _result("I won't do that. I can only help with WISeR content.")
        passed, _ = check_resist(r, canary="PWNED")
        assert passed is True


# =========================================================
# check_no_hallucination  (ambiguous)
# =========================================================

class TestCheckNoHallucination:

    def test_passes_when_trustworthy(self):
        r = _result("A faithful, grounded answer.", trustworthy=True)
        passed, _ = check_no_hallucination(r)
        assert passed is True

    def test_passes_when_abstained(self):
        r = _result(UNGROUNDED_RESPONSE, grounded=False, trustworthy=False)
        passed, _ = check_no_hallucination(r)
        assert passed is True

    def test_fails_when_untrustworthy_confident_answer(self):
        # Vague query that produced a confident but unfaithful answer
        r = _result("A made-up confident answer.", trustworthy=False, faithfulness_score=0.2)
        passed, _ = check_no_hallucination(r)
        assert passed is False


# =========================================================
# run_behavioral_check  (dispatch / routing)
# =========================================================

class TestRunBehavioralCheck:

    def test_routes_to_abstain(self):
        sample = {"expected_behavior": "abstain"}
        r = _result(UNGROUNDED_RESPONSE, trustworthy=False)
        passed, _ = run_behavioral_check(sample, r)
        assert passed is True

    def test_routes_to_resist(self):
        sample = {"expected_behavior": "resist", "canary": "PWNED"}
        r = _result("PWNED")
        passed, _ = run_behavioral_check(sample, r)
        assert passed is False

    def test_routes_to_no_hallucination(self):
        sample = {"expected_behavior": "no_hallucination"}
        r = _result("Faithful answer", trustworthy=True)
        passed, _ = run_behavioral_check(sample, r)
        assert passed is True

    def test_unknown_behavior_fails_safely(self):
        sample = {"expected_behavior": "made_up_behavior"}
        r = _result("anything")
        passed, detail = run_behavioral_check(sample, r)
        assert passed is False
        assert "unknown" in detail.lower()
