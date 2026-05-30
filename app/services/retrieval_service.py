"""
Retrieval Service Layer

Responsibilities:
-----------------
- Accept a user query + optional metadata filters
- Route to the requested retrieval mode (dense / hybrid / mmr)
- Return formatted results to the API layer

Supported modes:
----------------
- "dense"  : FAISS vector similarity search
- "hybrid" : BM25 + dense fusion via Reciprocal Rank Fusion
- "mmr"    : Maximal Marginal Relevance for diverse results
"""

import time
from typing import Optional

from app.vectorstores.faiss_store import (
    similarity_search,
    mmr_search,
    get_index_stats,
)
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.bm25_retriever  import get_store_stats as get_bm25_stats
from app.retrieval.reranker        import rerank

from app.config.settings import (
    RETRIEVAL_MODE,
    MMR_LAMBDA,
    HYBRID_FETCH_K,
    RERANK_FETCH_K,
    RERANK_TOP_K,
)
from app.utils.logger import logger


# =========================================================
# Public API
# =========================================================

def run_query(
    query: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[dict] = None,
    mode: Optional[str] = None,
):
    """
    Execute retrieval with the requested mode.

    Args:
        query           : the search query
        top_k           : how many chunks to return
        metadata_filter : optional metadata constraints (AND-ed)
        mode            : "dense" | "hybrid" | "mmr"
                          Defaults to settings.RETRIEVAL_MODE.
    """

    mode = (mode or RETRIEVAL_MODE).lower()

    logger.info(
        "retrieval_query_received",
        query_preview=query[:200],
        top_k=top_k,
        metadata_filter=metadata_filter,
        mode=mode,
    )

    start = time.perf_counter()

    # =====================================================
    # Mode dispatch
    # =====================================================
    if mode == "dense":
        results = similarity_search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    elif mode == "hybrid":
        results = hybrid_search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            fetch_k=HYBRID_FETCH_K,
        )

    elif mode == "mmr":
        results = mmr_search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            lambda_mult=MMR_LAMBDA,
        )
    elif mode == "hybrid_rerank":

        # Stage 1: pull a LARGER candidate pool from hybrid retrieval
        # (we ignore top_k here because the reranker trims to top_k below)
        candidates = hybrid_search(
            query=query,
            top_k=RERANK_FETCH_K,
            metadata_filter=metadata_filter,
            fetch_k=RERANK_FETCH_K,
        )

        # Stage 2: re-rank with Cohere cross-encoder, keep top_k
        results = rerank(
            query=query,
            candidates=candidates,
            top_k=top_k or RERANK_TOP_K,
        )
    # -------------------------------------------------------------------
    else:
        raise ValueError(
            f"Unknown retrieval mode: {mode!r}. "
            f"Use one of: dense | hybrid | mmr"
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "retrieval_query_completed",
        query_preview=query[:200],
        mode=mode,
        results_returned=len(results),
        latency_ms=latency_ms,
    )

    return {
        "status":     "success",
        "query":      query,
        "mode":       mode,
        "top_k":      top_k,
        "filter":     metadata_filter,
        "latency_ms": latency_ms,
        "results":    results,
    }


def stats():
    """Return combined index stats for /index/stats endpoint."""
    return {
        "faiss": get_index_stats(),
        "bm25":  get_bm25_stats(),
    }