"""
Retrieval Service Layer

Responsibilities:
-----------------
- Accept a user query + optional metadata filters
- Delegate similarity search to the active vector store
- Return formatted results to the API layer

Why a service layer exists for retrieval too:
---------------------------------------------
Keeping retrieval logic OUT of the API route means we can:
- swap FAISS for OpenSearch without touching the API
- add reranking / query rewriting / caching here later
- unit-test retrieval without spinning up FastAPI
"""

import time

from app.vectorstores.faiss_store import (
    similarity_search,
    get_index_stats,
)

from app.utils.logger import logger


def run_query(
    query: str,
    top_k: int = None,
    metadata_filter: dict = None,
):
    """
    Execute a similarity search and return formatted results.

    Args:
        query           : user question / search string
        top_k           : number of chunks to return (None -> settings default)
        metadata_filter : optional dict of metadata constraints

    Returns:
        dict containing:
            - status      : "success"
            - query       : echoed query
            - top_k       : number requested
            - filter      : echoed metadata filter
            - latency_ms  : end-to-end latency
            - results     : list of {content, metadata, similarity}
    """

    logger.info(
        "retrieval_query_received",
        query_preview=query[:200],
        top_k=top_k,
        metadata_filter=metadata_filter,
    )

    start = time.perf_counter()

    results = similarity_search(
        query=query,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "retrieval_query_completed",
        query_preview=query[:200],
        results_returned=len(results),
        latency_ms=latency_ms,
    )

    return {
        "status": "success",
        "query": query,
        "top_k": top_k,
        "filter": metadata_filter,
        "latency_ms": latency_ms,
        "results": results,
    }


def stats():
    """Thin pass-through for /index/stats endpoint."""
    return get_index_stats()
