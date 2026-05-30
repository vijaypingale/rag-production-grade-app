"""
Retrieval API Routes

Responsibilities:
-----------------
- Accept a search query (JSON body)
- Validate input
- Delegate to the retrieval service
- Return results + index stats

This route is thin. All search logic lives in
app/services/retrieval_service.py.
"""

from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.retrieval_service import run_query, stats

from app.utils.logger import logger


router = APIRouter(
    tags=["Retrieval"]
)


# ============================================================================
# Request schema
# ============================================================================
# Using Pydantic for the request body gives us:
#   - automatic validation
#   - self-documenting OpenAPI schema (visible at /docs)
#   - type safety for the metadata_filter dict
# ============================================================================

class SearchRequest(BaseModel):

    query: str = Field(
        ...,
        description="Natural-language search query.",
        examples=["What is the supplier onboarding process?"],
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of chunks to return (default from settings).",
    )

    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional metadata filter. Keys are AND-ed. "
            "Example: {\"doc_type\": \"pdf\"} or "
            "{\"source\": \"policy.pdf\"} or "
            "{\"doc_id\": \"abc-123\"}."
        ),
        examples=[{"doc_type": "pdf"}],
    )

    # ---- ADD THIS NEW FIELD ------------------------------
    mode: Optional[str] = Field(
        default=None,
        description=(
            "Retrieval mode: 'dense' (vector only), 'hybrid' "
            "(BM25 + dense via RRF), 'mmr' (diverse vector), "
            "or 'hybrid_rerank' (hybrid + Cohere Rerank v3). "
            "Defaults to settings.RETRIEVAL_MODE."
        ),
        examples=["hybrid_rerank"],
    )


# ============================================================================
# Search endpoint
# ============================================================================

@router.post("/search")
async def search(request: SearchRequest):
    """
    Run a semantic search against the persistent vector index.

    Returns:
        - status
        - echoed query / filter / top_k
        - latency_ms (server-side)
        - results: list of {content, metadata, similarity}
    """

    logger.info(
        "search_api_invoked",
        query_preview=request.query[:200],
        top_k=request.top_k,
        metadata_filter=request.metadata_filter,
    )

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query must be a non-empty string."
        )

    try:

        return run_query(
            query=request.query,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
            mode=request.mode, 
        )

    except FileNotFoundError as ex:

        # No index on disk -- caller hasn't ingested anything yet.
        logger.warning("search_api_index_missing", error=str(ex))

        raise HTTPException(
            status_code=409,
            detail=(
                "Vector index does not exist yet. "
                "Upload at least one document via /api/v1/ingest/upload first."
            ),
        )

    except Exception as ex:

        logger.error("search_api_failed", error=str(ex))

        raise HTTPException(
            status_code=500,
            detail=str(ex),
        )


# ============================================================================
# Index stats endpoint
# ============================================================================

@router.get("/index/stats")
async def index_stats():
    """
    Lightweight diagnostics about the persisted FAISS index.

    Useful for:
    - confirming ingestion actually wrote vectors
    - dashboards / readiness checks
    """

    logger.info("index_stats_api_invoked")

    return stats()
