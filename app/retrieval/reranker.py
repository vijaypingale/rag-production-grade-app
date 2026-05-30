"""
Cohere Rerank Module (Cross-Encoder Reranking)

Responsibilities:
-----------------
- Re-score retrieval candidates using a cross-encoder model
- Reorder candidates by true relevance (not just embedding similarity)
- Trim down to the final top-N for the LLM / caller

Why a separate reranking stage:
-------------------------------
Retrieval (FAISS / BM25 / hybrid) uses BI-ENCODERS:
  - query and document are vectorized INDEPENDENTLY
  - fast, but the model never sees both together
  - moderate precision

Reranking uses a CROSS-ENCODER:
  - query and document are fed JOINTLY through a transformer
  - the model can directly reason about relevance
  - ~100x slower per pair, but much more accurate

Production pattern:
  retrieval (fast, recall-oriented)  -> fetch_k=50 candidates
  reranking (slow, precision-oriented) -> top_k=5 final results
  generation (LLM)                     -> reads the top 5 to answer

Why Cohere Rerank specifically:
-------------------------------
- Most widely deployed hosted reranker in production RAG today
- Strong quality (top of public reranker benchmarks)
- Multilingual (100+ languages in v3.5)
- One simple HTTP call -- no GPU to manage
- Available natively on AWS Bedrock and Azure AI
- Free tier (100 calls/min) is plenty for prototypes

Alternative implementations (for the interview):
- BGE-reranker-v2-m3   (open source, self-host on GPU)
- Jina Reranker v2     (open weights, hosted API or self-host)
- LLM-as-reranker      (GPT-4 / Claude, highest quality, highest cost)
"""

import os
from typing import List, Dict, Optional

import cohere
from dotenv import load_dotenv

from app.config.settings import COHERE_RERANK_MODEL, RERANK_TOP_K
from app.utils.logger import logger


# Load .env so local dev picks up the Cohere API key.
load_dotenv()


# =========================================================
# Client factory
# =========================================================
# We construct the client lazily so import time stays fast
# and tests can mock it cleanly.

def _get_client():
    """
    Return a configured Cohere client.

    Raises:
        ValueError if COHERE_API_KEY is missing from the env.
    """

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise ValueError(
            "COHERE_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://dashboard.cohere.com/api-keys"
        )

    return cohere.ClientV2(api_key=api_key)


# =========================================================
# Public API
# =========================================================

def rerank(
    query: str,
    candidates: List[Dict],
    top_k: Optional[int] = None,
):
    """
    Re-rank a list of candidate chunks by their true relevance
    to the query using Cohere's cross-encoder.

    Args:
        query      : the user's query
        candidates : list of dicts from any retriever
                     (dense_search, bm25 search, or hybrid_search).
                     Each dict MUST have a "content" key with the
                     chunk text. All other fields are passed through.
        top_k      : how many final results to keep after rerank.
                     Defaults to settings.RERANK_TOP_K.

    Returns:
        List[dict]: candidates re-sorted by Cohere relevance score.
                    Each returned dict now also contains:
                      - rerank_score   : float in [0, 1]
                                         (1.0 = perfect match)
                      - original_rank  : the candidate's position
                                         BEFORE reranking
    """

    if top_k is None:
        top_k = RERANK_TOP_K

    if not candidates:
        logger.warning("rerank_called_with_no_candidates")
        return []

    client = _get_client()

    # =====================================================
    # Step 1: Extract the text Cohere will rank
    # =====================================================
    # The Cohere API expects a flat list of document strings.
    # Our retrievers return dicts with a "content" key.

    documents = [c["content"] for c in candidates]

    logger.info(
        "rerank_request_started",
        model=COHERE_RERANK_MODEL,
        query_preview=query[:200],
        candidates_in=len(candidates),
        top_k=top_k,
    )

    # =====================================================
    # Step 2: Call Cohere Rerank
    # =====================================================
    # response.results is a list of objects with:
    #   - index           : position in the input documents list
    #   - relevance_score : float in [0, 1]
    # They come back ALREADY SORTED by relevance_score descending.

    try:
        response = client.rerank(
            model=COHERE_RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=top_k,
        )

    except cohere.core.api_error.ApiError as ex:
        logger.error(
            "cohere_rerank_api_failed",
            error=str(ex),
            status_code=getattr(ex, "status_code", None),
        )
        raise

    # =====================================================
    # Step 3: Rebuild the full result dicts in the new order
    # =====================================================
    # We attach rerank_score and the candidate's original_rank
    # so callers (and the interview demo) can see how the
    # reranker reordered the list.

    reranked = []

    for r in response.results:

        original = candidates[r.index]

        reranked.append({
            **original,                              # keep content + metadata + any retriever scores
            "rerank_score":  float(r.relevance_score),
            "original_rank": r.index + 1,            # 1-based rank in the input
        })

    logger.info(
        "rerank_request_completed",
        model=COHERE_RERANK_MODEL,
        query_preview=query[:200],
        candidates_in=len(candidates),
        results_returned=len(reranked),
        top_rerank_score=reranked[0]["rerank_score"] if reranked else None,
    )

    return reranked