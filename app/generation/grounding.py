"""
Grounding & Hallucination Control — Section 9

This module implements the "verify after" and "enforce provenance" layers
of the three-tier hallucination defense. (The first tier, the pre-generation
grounding gate, lives in ask_service.py and uses GROUNDING_THRESHOLD.)

Three-tier defense recap:
-------------------------
1. Grounding gate      (BEFORE generation)  — ask_service.py
                         If retrieval is too weak, abstain — never call the LLM.
2. Faithfulness check  (AFTER generation)   — check_faithfulness() here
                         Decompose the answer into atomic claims and use an
                         LLM-as-judge (NLI-style) to verify each claim is
                         supported by the retrieved context. Produces a
                         faithfulness score gated on FAITHFULNESS_THRESHOLD.
3. Citation enforcement (programmatic)      — enforce_citations() here
                         Parse the [N] markers in the answer and verify each
                         maps to a real retrieved chunk. No LLM, no cost.

Why claim-level (not a single yes/no):
--------------------------------------
A single "is this faithful? yes/no" call is a PoC pattern. Production systems
(FactCC / NLI / RAGAS-style) decompose the answer into atomic claims and judge
each one, producing a *score* rather than a binary. This catches answers that
are mostly correct but smuggle in one unsupported detail — the most common and
dangerous hallucination mode.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional

from app.generation.llm import run_judge
from app.config.settings import FAITHFULNESS_THRESHOLD
from app.utils.logger import logger


# =========================================================
# Citation Enforcement (programmatic — no LLM)
# =========================================================

# Matches citation markers like [1], [12] in the answer text.
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass
class CitationAudit:
    """
    Result of verifying the [N] markers in an answer against the
    citation_map produced by the context assembler.

    Fields:
        cited_numbers  : every [N] number found in the answer (unique, sorted)
        valid_numbers  : cited numbers that map to a real retrieved chunk
        orphan_numbers : cited numbers with NO matching chunk — these are the
                         dangerous ones (the LLM invented a citation)
        all_valid      : True if there are no orphan citations
    """
    cited_numbers:  list[int]
    valid_numbers:  list[int]
    orphan_numbers: list[int]
    all_valid:      bool


def enforce_citations(answer: str, citation_map: dict) -> CitationAudit:
    """
    Verify every [N] marker in the answer corresponds to a real chunk.

    Args:
        answer       : the LLM-generated answer text
        citation_map : {N: metadata} from assemble_context — the set of
                       citation numbers that were actually given to the LLM

    Returns:
        CitationAudit

    Why this matters:
        Even a well-instructed LLM occasionally emits a citation number
        that was never in the context (e.g. "[7]" when only [1]-[5] exist).
        That is a provenance hallucination. Catching it is pure string
        work — cheap, deterministic, and 100% reliable, so it always runs.
    """
    cited = sorted({int(n) for n in _CITATION_PATTERN.findall(answer)})
    valid_keys = set(citation_map.keys())

    valid  = [n for n in cited if n in valid_keys]
    orphan = [n for n in cited if n not in valid_keys]

    audit = CitationAudit(
        cited_numbers=cited,
        valid_numbers=valid,
        orphan_numbers=orphan,
        all_valid=(len(orphan) == 0),
    )

    if orphan:
        logger.warning(
            "citation_enforcement_orphans_found",
            cited=cited,
            orphan=orphan,
            valid_keys=sorted(valid_keys),
        )

    return audit


# =========================================================
# Faithfulness Check (LLM-as-judge, claim-level)
# =========================================================

@dataclass
class ClaimVerdict:
    """One atomic claim extracted from the answer and its verdict."""
    claim:   str
    verdict: str            # "supported" | "unsupported" | "unverifiable"


@dataclass
class FaithfulnessResult:
    """
    Result of the post-generation faithfulness check.

    Fields:
        checked          : did the check actually run? (False if disabled or errored)
        score            : supported_claims / total_claims, in [0.0, 1.0]
        passed           : score >= FAITHFULNESS_THRESHOLD
        total_claims     : number of atomic claims extracted from the answer
        supported_claims : claims the judge marked "supported"
        verdicts         : per-claim breakdown
        threshold        : the threshold used (for transparency in the response)
    """
    checked:          bool
    score:            float
    passed:           bool
    total_claims:     int
    supported_claims: int
    verdicts:         list[ClaimVerdict] = field(default_factory=list)
    threshold:        float = FAITHFULNESS_THRESHOLD


# System prompt for the judge. Strict JSON output keeps parsing deterministic.
_FAITHFULNESS_SYSTEM_PROMPT = """You are a strict faithfulness judge for a \
retrieval-augmented system. Your job is to decide whether an ANSWER is fully \
supported by the provided CONTEXT.

Steps:
1. Break the ANSWER into atomic factual claims (one verifiable fact each).
2. For each claim, decide ONLY from the CONTEXT:
   - "supported"     : the claim is directly backed by the context
   - "unsupported"   : the claim contradicts or is absent from the context
   - "unverifiable"  : the claim is too vague to verify (greetings, opinions)
3. Do NOT use outside knowledge. Judge strictly against the context.

Return ONLY valid JSON in exactly this shape, with no extra text:
{
  "claims": [
    {"claim": "<atomic claim>", "verdict": "supported|unsupported|unverifiable"}
  ]
}"""


def _build_faithfulness_prompt(answer: str, context: str) -> str:
    return f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"


def _parse_judge_json(raw: str) -> Optional[list[dict]]:
    """
    Parse the judge's JSON response defensively.

    LLMs sometimes wrap JSON in markdown fences or add stray prose. We
    extract the first {...} block and parse it. Returns None on failure
    so the caller can mark the check as 'unchecked' rather than crash.
    """
    try:
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        # Grab the outermost JSON object
        start = cleaned.find("{")
        end   = cleaned.rfind("}")
        if start == -1 or end == -1:
            return None
        data = json.loads(cleaned[start:end + 1])
        claims = data.get("claims")
        return claims if isinstance(claims, list) else None
    except (json.JSONDecodeError, ValueError):
        return None


def check_faithfulness(answer: str, context: str) -> FaithfulnessResult:
    """
    Verify the answer is supported by the context using a claim-level
    LLM-as-judge (NLI-style) check.

    Args:
        answer  : the LLM-generated answer
        context : the assembled context that was given to the LLM

    Returns:
        FaithfulnessResult

    Failure handling:
        If the judge call or JSON parse fails, we return checked=False
        with score 0.0 — the main user answer is NOT failed because a
        secondary verification step had a hiccup. The caller logs it and
        treats faithfulness as 'unknown'.
    """
    # An empty / trivial answer has nothing to verify.
    if not answer or not answer.strip():
        return FaithfulnessResult(
            checked=False, score=0.0, passed=False,
            total_claims=0, supported_claims=0,
        )

    try:
        raw = run_judge(
            system_prompt=_FAITHFULNESS_SYSTEM_PROMPT,
            user_prompt=_build_faithfulness_prompt(answer, context),
        )
    except Exception as exc:
        # Judge unavailable — degrade gracefully, do not fail the request.
        logger.warning("faithfulness_judge_call_failed", error=str(exc))
        return FaithfulnessResult(
            checked=False, score=0.0, passed=False,
            total_claims=0, supported_claims=0,
        )

    claims = _parse_judge_json(raw)

    if not claims:
        logger.warning("faithfulness_judge_unparseable", raw_preview=raw[:200])
        return FaithfulnessResult(
            checked=False, score=0.0, passed=False,
            total_claims=0, supported_claims=0,
        )

    verdicts = [
        ClaimVerdict(
            claim=c.get("claim", ""),
            verdict=c.get("verdict", "unverifiable"),
        )
        for c in claims
    ]

    # Score = supported / verifiable claims.
    # "unverifiable" claims (greetings, opinions) are excluded from the
    # denominator so they neither help nor hurt the score.
    verifiable = [v for v in verdicts if v.verdict in ("supported", "unsupported")]
    supported  = [v for v in verifiable if v.verdict == "supported"]

    total_verifiable = len(verifiable)
    score = (len(supported) / total_verifiable) if total_verifiable > 0 else 1.0
    passed = score >= FAITHFULNESS_THRESHOLD

    result = FaithfulnessResult(
        checked=True,
        score=round(score, 4),
        passed=passed,
        total_claims=total_verifiable,
        supported_claims=len(supported),
        verdicts=verdicts,
    )

    log_fn = logger.info if passed else logger.warning
    log_fn(
        "faithfulness_check_complete",
        score=result.score,
        passed=passed,
        threshold=FAITHFULNESS_THRESHOLD,
        total_claims=total_verifiable,
        supported_claims=len(supported),
        unsupported=[v.claim for v in verifiable if v.verdict == "unsupported"],
    )

    return result
