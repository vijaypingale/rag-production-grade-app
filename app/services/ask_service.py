"""
Ask Service — Section 8

Responsibilities:
-----------------
- Orchestrate the full RAG pipeline for a single user question:
    1. Retrieve + rerank candidates (hybrid_rerank mode)
    2. Grounding gate  — abort early if top rerank score is too low
                         (Section 9 placeholder: gate is live, full
                          faithfulness check added in Section 9)
    3. Assemble context — token-budgeted, citation-tagged string
    4. Generate answer  — LLM call via llm.generate_answer()
    5. Return structured AskResult

Architecture principle — thin API, fat service:
    ask_api.py only validates HTTP input/output and maps exceptions
    to HTTP status codes.  ALL pipeline logic lives here, making it
    fully testable without starting FastAPI.

Latency breakdown:
    Every stage is timed independently. The returned AskResult
    includes latency_breakdown so dashboards can pinpoint which
    stage is slow on a per-query basis.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from app.retrieval.reranker           import rerank
from app.retrieval.hybrid_retriever   import hybrid_search
from app.generation.context_assembler import assemble_context
from app.generation.llm               import (
    generate_answer,
    LLMError,
    LLMTimeoutError,
    LLMQuotaError,
    LLMProviderError,
)
from app.generation.grounding         import (
    enforce_citations,
    check_faithfulness,
    CitationAudit,
    FaithfulnessResult,
)
from app.config.settings import (
    RERANK_FETCH_K,
    RERANK_TOP_K,
    HYBRID_FETCH_K,
    MAX_CONTEXT_TOKENS,
    LLM_MODEL,
    GROUNDING_THRESHOLD,
    FAITHFULNESS_CHECK_ENABLED,
    FAITHFULNESS_THRESHOLD,
)
from app.observability.tracing import get_tracer
from app.observability.cost import compute_cost
from opentelemetry.trace import Status, StatusCode
from app.utils.logger import logger


# =========================================================
# Response Schema
# =========================================================

@dataclass
class CitationEntry:
    """
    Maps a single [N] citation tag to its source chunk.

    Fields mirror the citation_map produced by assemble_context,
    exposed as a typed dataclass so the API layer can serialise
    them cleanly via Pydantic.
    """
    citation_number: int
    source:          str
    page:            Optional[int]
    chunk_id:        Optional[str]
    rerank_score:    Optional[float]


@dataclass
class LatencyBreakdown:
    """Per-stage wall-clock timings in milliseconds."""
    retrieval_ms:  float
    assembly_ms:   float
    generation_ms: float
    total_ms:      float


@dataclass
class AskResult:
    """
    Full structured response from a single ask() call.

    Fields:
        answer           : LLM-generated answer with [N] citation tags
        citations        : list of CitationEntry objects (one per [N] tag used)
        model_used       : exact model that generated the answer (e.g. gpt-4o-mini)
        provider         : LLM provider ("openai" | "bedrock")
        prompt_tokens    : tokens consumed by the prompt
        completion_tokens: tokens in the generated answer
        total_tokens     : prompt + completion
        chunks_used      : number of context chunks passed to the LLM
        grounded         : True if the grounding gate passed (pre-generation)
        latency          : LatencyBreakdown for observability

        --- Section 9: post-generation verification ---
        faithfulness_checked   : did the faithfulness check run?
        faithfulness_score     : supported_claims / verifiable_claims (0.0–1.0)
        faithfulness_passed    : score >= FAITHFULNESS_THRESHOLD
        citations_valid        : True if every [N] in the answer maps to a chunk
        orphan_citations       : [N] numbers in the answer with no source chunk
        trustworthy            : overall verdict — grounded AND citations_valid
                                 AND (faithfulness passed OR not checked)
    """
    answer:            str
    citations:         list[CitationEntry]
    model_used:        str
    provider:          str
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    chunks_used:       int
    grounded:          bool
    latency:           LatencyBreakdown

    # Section 9 — verification (defaults keep the ungrounded-path return simple)
    faithfulness_checked: bool        = False
    faithfulness_score:   float       = 0.0
    faithfulness_passed:  bool        = False
    citations_valid:      bool        = True
    orphan_citations:     list[int]   = field(default_factory=list)
    trustworthy:          bool        = False

    # Raw retrieved chunk texts (used by Section 10 eval & observability).
    # Not exposed over the HTTP API — internal/eval use only.
    retrieved_contexts:   list[str]   = field(default_factory=list)


# =========================================================
# Ungrounded response constant
# =========================================================
# Returned verbatim when the grounding gate blocks generation.
# Consistent wording matters — downstream quality dashboards
# can detect "ungrounded" responses by matching this string.

UNGROUNDED_RESPONSE = (
    "I don't have enough information in the provided documents "
    "to answer this question."
)


# =========================================================
# Internal helpers
# =========================================================

def _run_retrieval(
    query: str,
    top_k: int,
    metadata_filter: Optional[dict],
) -> tuple[list[dict], float]:
    """
    Run hybrid retrieval followed by Cohere reranking.

    Returns:
        (reranked_chunks, latency_ms)

    Strategy:
        hybrid_search pulls RERANK_FETCH_K candidates from FAISS + BM25,
        then rerank() re-scores them and returns the top top_k.
        This two-stage approach maximises recall before the expensive
        cross-encoder reranking step.
    """
    start = time.perf_counter()

    candidates = hybrid_search(
        query=query,
        top_k=RERANK_FETCH_K,
        metadata_filter=metadata_filter,
        fetch_k=HYBRID_FETCH_K,
    )

    reranked = rerank(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return reranked, latency_ms


def _check_grounding(reranked_chunks: list[dict]) -> tuple[bool, float]:
    """
    Grounding gate: inspect the highest rerank score in the result set.

    Returns:
        (is_grounded, max_score)

    Decision rule:
        If max rerank score < GROUNDING_THRESHOLD the retrieved content
        is too weakly related to the query to generate a reliable answer.
        We return False to block the LLM call entirely — cheaper and
        safer than generating and then checking faithfulness post-hoc.

    Section 9 extension point:
        A faithfulness check (LLM-as-judge comparing the generated answer
        to the cited chunks) will be added here in Section 9. That check
        runs AFTER generation and complements this pre-generation gate.
    """
    if not reranked_chunks:
        return False, 0.0

    max_score = max(
        chunk.get("rerank_score", 0.0) for chunk in reranked_chunks
    )
    return max_score >= GROUNDING_THRESHOLD, max_score


# =========================================================
# Public API
# =========================================================

def ask(
    query: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[dict] = None,
    max_context_tokens: Optional[int] = None,
) -> AskResult:
    """
    Observability wrapper (Section 12) around the RAG pipeline.

    Creates one OpenTelemetry span per request ("rag.ask") and records the
    key metrics as span attributes — cost, tokens, faithfulness, latency,
    trustworthiness. This is vendor-neutral: the same span exports to console
    (local) or Datadog/Grafana (prod) depending only on OTEL_EXPORTER.

    All pipeline logic lives in _ask_impl(); this wrapper just instruments it.
    """
    tracer = get_tracer()

    with tracer.start_as_current_span("rag.ask") as span:
        span.set_attribute("rag.query", query[:500])
        span.set_attribute("rag.top_k", top_k or RERANK_TOP_K)

        try:
            result = _ask_impl(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
                max_context_tokens=max_context_tokens,
            )
        except Exception as exc:
            # Mark the span as errored so it shows red in the backend UI.
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        # Record outcome metrics as span attributes (what dashboards chart).
        cost = compute_cost(
            result.model_used, result.prompt_tokens, result.completion_tokens
        )
        span.set_attribute("rag.grounded", result.grounded)
        span.set_attribute("rag.trustworthy", result.trustworthy)
        span.set_attribute("rag.chunks_used", result.chunks_used)
        span.set_attribute("rag.prompt_tokens", result.prompt_tokens)
        span.set_attribute("rag.completion_tokens", result.completion_tokens)
        span.set_attribute("rag.total_tokens", result.total_tokens)
        span.set_attribute("rag.cost_usd", cost)
        span.set_attribute("rag.faithfulness_score", result.faithfulness_score)
        span.set_attribute("rag.citations_valid", result.citations_valid)
        span.set_attribute("rag.latency.retrieval_ms", result.latency.retrieval_ms)
        span.set_attribute("rag.latency.generation_ms", result.latency.generation_ms)
        span.set_attribute("rag.latency.total_ms", result.latency.total_ms)

        return result


def _ask_impl(
    query: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[dict] = None,
    max_context_tokens: Optional[int] = None,
) -> AskResult:
    """
    Execute the full RAG pipeline for one user question.

    Args:
        query              : the user's natural-language question
        top_k              : number of chunks to pass to the LLM after
                             reranking. Defaults to settings.RERANK_TOP_K.
        metadata_filter    : optional metadata constraints passed through
                             to retrieval (e.g. {"source": "policy.pdf"})
        max_context_tokens : override the token budget for this request.
                             Defaults to settings.MAX_CONTEXT_TOKENS.

    Returns:
        AskResult dataclass

    Raises:
        LLMTimeoutError    — LLM timed out after all retries
        LLMQuotaError      — rate limit exhausted
        LLMProviderError   — permanent provider error (bad key, etc.)
        LLMError           — unexpected LLM failure
        FileNotFoundError  — FAISS index does not exist yet
        ValueError         — invalid input
    """

    effective_top_k     = top_k or RERANK_TOP_K
    effective_max_tokens = max_context_tokens or MAX_CONTEXT_TOKENS
    pipeline_start      = time.perf_counter()

    logger.info(
        "ask_pipeline_started",
        query_preview=query[:200],
        top_k=effective_top_k,
        metadata_filter=metadata_filter,
        max_context_tokens=effective_max_tokens,
    )

    # =====================================================
    # Stage 1: Retrieval + Reranking
    # =====================================================

    reranked_chunks, retrieval_ms = _run_retrieval(
        query=query,
        top_k=effective_top_k,
        metadata_filter=metadata_filter,
    )

    logger.info(
        "ask_retrieval_complete",
        chunks_retrieved=len(reranked_chunks),
        retrieval_ms=retrieval_ms,
    )

    # =====================================================
    # Stage 2: Grounding Gate
    # =====================================================

    is_grounded, max_rerank_score = _check_grounding(reranked_chunks)

    logger.info(
        "ask_grounding_check",
        is_grounded=is_grounded,
        max_rerank_score=round(max_rerank_score, 4),
        threshold=GROUNDING_THRESHOLD,
    )

    if not is_grounded:
        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

        logger.warning(
            "ask_grounding_gate_blocked",
            query_preview=query[:200],
            max_rerank_score=round(max_rerank_score, 4),
            threshold=GROUNDING_THRESHOLD,
            total_ms=total_ms,
        )

        return AskResult(
            answer=UNGROUNDED_RESPONSE,
            citations=[],
            model_used=LLM_MODEL,
            provider="n/a",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            chunks_used=0,
            grounded=False,
            latency=LatencyBreakdown(
                retrieval_ms=retrieval_ms,
                assembly_ms=0.0,
                generation_ms=0.0,
                total_ms=total_ms,
            ),
        )

    # =====================================================
    # Stage 3: Context Assembly
    # =====================================================

    assembly_start  = time.perf_counter()

    assembled = assemble_context(
        chunks=reranked_chunks,
        max_tokens=effective_max_tokens,
    )

    assembly_ms = round((time.perf_counter() - assembly_start) * 1000, 2)

    logger.info(
        "ask_assembly_complete",
        chunks_used=assembled["chunks_used"],
        tokens_used=assembled["tokens_used"],
        assembly_ms=assembly_ms,
    )

    # =====================================================
    # Stage 4: LLM Generation
    # =====================================================

    llm_response = generate_answer(
        context=assembled["context"],
        query=query,
    )

    # =====================================================
    # Stage 5: Post-generation verification (Section 9)
    # =====================================================
    # Two checks run AFTER the answer is produced:
    #   (a) Citation enforcement — programmatic, always runs, ~free
    #   (b) Faithfulness check   — LLM-as-judge, optional (costs a call)
    # Neither one fails the request; they ANNOTATE the answer with a
    # trustworthiness verdict the caller/UI can act on.
    # =====================================================

    citation_audit: CitationAudit = enforce_citations(
        answer=llm_response.answer,
        citation_map=assembled["citation_map"],
    )

    if FAITHFULNESS_CHECK_ENABLED:
        faithfulness: FaithfulnessResult = check_faithfulness(
            answer=llm_response.answer,
            context=assembled["context"],
        )
    else:
        # Check disabled — leave an unchecked result.
        faithfulness = FaithfulnessResult(
            checked=False, score=0.0, passed=False,
            total_claims=0, supported_claims=0,
        )

    # Overall trustworthiness verdict.
    # If the faithfulness check did not run, we don't penalise the answer
    # for it (passed-or-unchecked), but citation validity always counts.
    faithfulness_ok = faithfulness.passed or not faithfulness.checked
    trustworthy = citation_audit.all_valid and faithfulness_ok

    # =====================================================
    # Stage 6: Build structured response
    # =====================================================

    total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

    citations = [
        CitationEntry(
            citation_number=num,
            source=entry.get("source", "unknown"),
            page=entry.get("page"),
            chunk_id=entry.get("chunk_id"),
            rerank_score=entry.get("rerank_score"),
        )
        for num, entry in assembled["citation_map"].items()
    ]

    logger.info(
        "ask_pipeline_complete",
        query_preview=query[:200],
        chunks_used=assembled["chunks_used"],
        citations_count=len(citations),
        model_used=llm_response.model_used,
        total_tokens=llm_response.total_tokens,
        faithfulness_checked=faithfulness.checked,
        faithfulness_score=faithfulness.score,
        faithfulness_passed=faithfulness.passed,
        citations_valid=citation_audit.all_valid,
        orphan_citations=citation_audit.orphan_numbers,
        trustworthy=trustworthy,
        retrieval_ms=retrieval_ms,
        assembly_ms=assembly_ms,
        generation_ms=llm_response.latency_ms,
        total_ms=total_ms,
    )

    return AskResult(
        answer=llm_response.answer,
        citations=citations,
        model_used=llm_response.model_used,
        provider=llm_response.provider,
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        total_tokens=llm_response.total_tokens,
        chunks_used=assembled["chunks_used"],
        grounded=True,
        latency=LatencyBreakdown(
            retrieval_ms=retrieval_ms,
            assembly_ms=assembly_ms,
            generation_ms=llm_response.latency_ms,
            total_ms=total_ms,
        ),
        faithfulness_checked=faithfulness.checked,
        faithfulness_score=faithfulness.score,
        faithfulness_passed=faithfulness.passed,
        citations_valid=citation_audit.all_valid,
        orphan_citations=citation_audit.orphan_numbers,
        trustworthy=trustworthy,
        retrieved_contexts=[c.get("content", "") for c in reranked_chunks],
    )
