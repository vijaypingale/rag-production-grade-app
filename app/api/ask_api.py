"""
Ask API — POST /api/v1/ask

Responsibilities:
-----------------
- Validate the incoming HTTP request (Pydantic)
- Delegate entirely to ask_service.ask()
- Map service-layer exceptions to correct HTTP status codes
- Return a structured JSON response

Design principle — thin API layer:
    No pipeline logic lives here. This file only speaks HTTP:
    it validates input, calls the service, and translates
    exceptions into appropriate status codes. All RAG logic
    is in app/services/ask_service.py and is fully testable
    without starting the HTTP server.

HTTP status code mapping:
    200  — answer returned (grounded or ungrounded)
    400  — empty or invalid query
    409  — FAISS index does not exist (ingest first)
    503  — LLM provider unavailable (timeout / quota)
    500  — unexpected internal error
"""

from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.ask_service import (
    ask,
    AskResult,
    CitationEntry,
    LatencyBreakdown,
    UNGROUNDED_RESPONSE,
)
from app.generation.llm import (
    LLMTimeoutError,
    LLMQuotaError,
    LLMProviderError,
    LLMError,
)
from app.utils.logger import logger


router = APIRouter(tags=["Ask"])


# ============================================================================
# Request Schema
# ============================================================================

class AskRequest(BaseModel):
    """
    Body for POST /api/v1/ask.

    Fields:
        query              : the user's natural-language question (required)
        top_k              : how many reranked chunks to use as context.
                             Defaults to settings.RERANK_TOP_K (5).
        metadata_filter    : optional dict to restrict retrieval to a subset
                             of documents (e.g. {"source": "policy.pdf"}).
        max_context_tokens : override the default 6000-token context budget
                             for this request only.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question to answer from ingested documents.",
        examples=["What is the supplier onboarding process?"],
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of reranked chunks to include in the LLM context.",
    )

    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional metadata filter applied at retrieval time. "
            "Keys are AND-ed. Example: {\"source\": \"policy.pdf\"}."
        ),
        examples=[{"source": "supplier-guide.pdf"}],
    )

    max_context_tokens: Optional[int] = Field(
        default=None,
        ge=500,
        le=20000,
        description=(
            "Override the default token budget for assembled context. "
            "Useful when routing to a model with a smaller context window."
        ),
    )


# ============================================================================
# Response Schemas
# ============================================================================
# Pydantic response models give us:
#   - automatic OpenAPI documentation at /docs
#   - type-safe serialisation (no accidental field leaks)
#   - consistent field names across all API versions

class CitationResponse(BaseModel):
    citation_number: int
    source:          str
    page:            Optional[int]           = None
    chunk_id:        Optional[str]           = None
    rerank_score:    Optional[float]         = None


class LatencyResponse(BaseModel):
    retrieval_ms:  float
    assembly_ms:   float
    generation_ms: float
    total_ms:      float


class AskResponse(BaseModel):
    """
    Full response body for POST /api/v1/ask.

    Fields:
        answer            : LLM-generated answer with [N] citation markers
        citations         : source metadata for each [N] citation in the answer
        grounded          : False means the grounding gate blocked generation;
                            the answer will be the standard "I don't have
                            enough information" message
        model_used        : exact model identifier used for generation
        provider          : LLM provider ("openai" | "bedrock" | "n/a")
        prompt_tokens     : tokens consumed by the prompt
        completion_tokens : tokens in the generated answer
        total_tokens      : prompt + completion
        chunks_used       : number of context chunks sent to the LLM
        latency           : per-stage latency breakdown in milliseconds
    """

    answer:            str
    citations:         List[CitationResponse]
    grounded:          bool
    model_used:        str
    provider:          str
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    chunks_used:       int
    latency:           LatencyResponse

    # Section 9 — post-generation verification
    trustworthy:          bool       = False
    citations_valid:      bool       = True
    orphan_citations:     List[int]  = []
    faithfulness_checked: bool       = False
    faithfulness_score:   float      = 0.0
    faithfulness_passed:  bool       = False

    # Section 13 — True if served from the semantic cache (no retrieval/LLM)
    cache_hit:            bool       = False


# ============================================================================
# Conversion helper
# ============================================================================

def _build_response(result: AskResult) -> AskResponse:
    """Convert the service-layer AskResult dataclass to the API response model."""
    return AskResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                citation_number=c.citation_number,
                source=c.source,
                page=c.page,
                chunk_id=c.chunk_id,
                rerank_score=c.rerank_score,
            )
            for c in result.citations
        ],
        grounded=result.grounded,
        model_used=result.model_used,
        provider=result.provider,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        chunks_used=result.chunks_used,
        latency=LatencyResponse(
            retrieval_ms=result.latency.retrieval_ms,
            assembly_ms=result.latency.assembly_ms,
            generation_ms=result.latency.generation_ms,
            total_ms=result.latency.total_ms,
        ),
        trustworthy=result.trustworthy,
        citations_valid=result.citations_valid,
        orphan_citations=result.orphan_citations,
        faithfulness_checked=result.faithfulness_checked,
        faithfulness_score=result.faithfulness_score,
        faithfulness_passed=result.faithfulness_passed,
        cache_hit=result.cache_hit,
    )


# ============================================================================
# Endpoint
# ============================================================================

@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question answered from ingested documents",
    responses={
        200: {"description": "Answer generated (check `grounded` field)."},
        400: {"description": "Empty or invalid query."},
        409: {"description": "No documents ingested yet — run /ingest/upload first."},
        503: {"description": "LLM provider temporarily unavailable."},
        500: {"description": "Unexpected internal error."},
    },
)
async def ask_endpoint(request: AskRequest):
    """
    Answer a natural-language question using the ingested documents.

    The pipeline runs:
      1. Hybrid retrieval (FAISS + BM25) with Cohere reranking
      2. Grounding gate  — returns "I don't know" if context is too weak
      3. Token-budgeted context assembly with [N] citation tags
      4. LLM generation  — answer cites sources as [1], [2], ...

    Check the `grounded` field in the response:
      - `true`  → answer is backed by retrieved content
      - `false` → insufficient information found; answer is the standard
                  "I don't have enough information" message
    """

    logger.info(
        "ask_api_invoked",
        query_preview=request.query[:200],
        top_k=request.top_k,
        metadata_filter=request.metadata_filter,
    )

    try:
        result = ask(
            query=request.query,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
            max_context_tokens=request.max_context_tokens,
        )
        return _build_response(result)

    except FileNotFoundError as exc:
        # FAISS index missing — caller needs to ingest documents first
        logger.warning("ask_api_index_missing", error=str(exc))
        raise HTTPException(
            status_code=409,
            detail=(
                "Vector index does not exist yet. "
                "Upload at least one document via /api/v1/ingest/upload first."
            ),
        )

    except (LLMTimeoutError, LLMQuotaError) as exc:
        # Transient provider issue — caller can retry
        logger.error("ask_api_llm_unavailable", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=(
                "The language model is temporarily unavailable. "
                "Please try again in a few seconds."
            ),
        )

    except LLMProviderError as exc:
        # Permanent config error — needs operator attention
        logger.error("ask_api_llm_provider_error", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="LLM provider configuration error. Check server logs.",
        )

    except LLMError as exc:
        logger.error("ask_api_llm_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    except ValueError as exc:
        logger.warning("ask_api_invalid_input", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        logger.error("ask_api_unexpected_error", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Check server logs.",
        )
