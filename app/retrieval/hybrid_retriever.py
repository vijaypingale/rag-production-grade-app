"""
Hybrid Retriever Module -- Dense + Sparse Fusion via RRF

Responsibilities:
-----------------
- Run BOTH the dense (FAISS) and sparse (BM25) retrievers
- Fuse their rankings using Reciprocal Rank Fusion (RRF)
- Return a single unified ranked list

Why hybrid retrieval is the production default:
-----------------------------------------------
- Dense vectors capture SEMANTIC similarity but miss
  exact-keyword queries (codes, names, jargon).
- BM25 captures LEXICAL overlap but misses paraphrases
  and concept-level matches.
- Used together, they cover each other's weaknesses.

Why Reciprocal Rank Fusion (RRF):
---------------------------------
RRF combines rankings without needing to normalize raw
scores (FAISS distance is on a different scale than BM25
score; normalizing them is error-prone).

The formula is simply:

    RRF_score(d) = sum over retrievers r of  1 / (k + rank_r(d))

where k is a smoothing constant (60 is the canonical value
from the original RRF paper by Cormack et al., 2009).

This dampens the contribution of low-ranked results and
naturally rewards documents that BOTH retrievers consider
relevant.
"""

from typing import List, Optional

from app.vectorstores.faiss_store import similarity_search as dense_search
from app.retrieval.bm25_retriever  import search           as sparse_search

from app.config.settings import RETRIEVAL_TOP_K
from app.utils.logger    import logger


# =========================================================
# RRF smoothing constant
# =========================================================
# 60 is the value from the canonical RRF paper.
# Higher values flatten the score curve (more equal weights
# across ranks); lower values emphasize top-ranked items.

RRF_K = 60  # edited by Claude Code on 2026-06-07


# =========================================================
# Public API
# =========================================================

def hybrid_search(
    query: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[dict] = None,
    fetch_k: int = 20,
):
    """
    Run dense + sparse retrieval in parallel and fuse with RRF.

    Args:
        query           : user query
        top_k           : final number of fused results to return
        metadata_filter : dict, AND-ed across both retrievers
        fetch_k         : how many candidates to pull from EACH
                          retriever before fusion. Higher = more
                          recall, slightly slower. fetch_k > top_k
                          is required for RRF to be meaningful.

    Returns:
        List[dict] with keys:
            - content
            - metadata
            - rrf_score        : fused score
            - dense_rank       : rank in dense results (or None)
            - sparse_rank      : rank in BM25 results (or None)
    """

    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    logger.info(
        "hybrid_search_started",
        query_preview=query[:200],
        top_k=top_k,
        fetch_k=fetch_k,
        metadata_filter=metadata_filter,
    )

    # =====================================================
    # Step 1: pull top fetch_k from each retriever
    # =====================================================
    # Both calls are independent and could be parallelized
    # (e.g. asyncio.gather) but for prototype simplicity
    # we keep them sequential.
    # =====================================================

    dense_results  = dense_search(
        query=query,
        top_k=fetch_k,
        metadata_filter=metadata_filter,
    )

    sparse_results = sparse_search(
        query=query,
        top_k=fetch_k,
        metadata_filter=metadata_filter,
    )

    logger.info(
        "hybrid_search_candidates_fetched",
        dense_candidates=len(dense_results),
        sparse_candidates=len(sparse_results),
    )

    # =====================================================
    # Step 2: build a unified pool keyed by chunk_id
    # =====================================================
    # We use the chunk_id (UUID stamped during chunking) as
    # the canonical key. Each entry tracks the ranks from
    # BOTH retrievers so we can compute RRF below.
    # =====================================================

    pool = {}

    for rank, result in enumerate(dense_results, start=1):

        chunk_id = result["metadata"].get("chunk_id")

        if chunk_id is None:
            # Defensive: skip results without a chunk_id (shouldn't happen
            # if ingestion went through our text_splitter).
            continue

        pool[chunk_id] = {
            "content":     result["content"],
            "metadata":    result["metadata"],
            "dense_rank":  rank,
            "sparse_rank": None,
        }

    for rank, result in enumerate(sparse_results, start=1):

        chunk_id = result["metadata"].get("chunk_id")

        if chunk_id is None:
            continue

        if chunk_id in pool:
            pool[chunk_id]["sparse_rank"] = rank
        else:
            pool[chunk_id] = {
                "content":     result["content"],
                "metadata":    result["metadata"],
                "dense_rank":  None,
                "sparse_rank": rank,
            }

    # =====================================================
    # Step 3: compute RRF score for every pooled doc
    # =====================================================

    for chunk_id, entry in pool.items():

        score = 0.0

        if entry["dense_rank"] is not None:
            score += 1.0 / (RRF_K + entry["dense_rank"])

        if entry["sparse_rank"] is not None:
            score += 1.0 / (RRF_K + entry["sparse_rank"])

        entry["rrf_score"] = score

    # =====================================================
    # Step 4: sort by RRF score, return top_k
    # =====================================================

    fused = sorted(
        pool.values(),
        key=lambda e: e["rrf_score"],
        reverse=True,
    )[:top_k]

    logger.info(
        "hybrid_search_completed",
        query_preview=query[:200],
        pool_size=len(pool),
        results_returned=len(fused),
        top_rrf_score=fused[0]["rrf_score"] if fused else None,
    )

    return fused