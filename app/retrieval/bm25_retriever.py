"""
BM25 Sparse Retriever Module

Responsibilities:
-----------------
- Maintain a persistent BM25 index alongside FAISS
- Provide keyword-based retrieval as a complement to dense vector search
- Designed to be combined with dense retrieval via RRF (see hybrid_retriever.py)

Why BM25 in a RAG system:
-------------------------
Dense embeddings excel at SEMANTIC similarity ("how does X work")
but often miss queries that need EXACT keyword matching:
  - product codes, SKUs, error codes        (e.g. "ERR-4042", "WISeR-2025")
  - person / organization names              (e.g. "Acme Corp")
  - rare technical jargon                    (e.g. "k-NN HNSW efSearch")
  - acronyms not seen during model training
BM25 catches all of these because it scores by term frequency
relative to document length and inverse document frequency.

In production RAG systems, BM25 + dense + RRF fusion is the
near-universal default ("hybrid retrieval"). This module
implements the sparse half; hybrid_retriever.py fuses the two.

Persistence strategy:
---------------------
We pickle ONLY the tokenized corpus + Documents to disk,
NOT the BM25Okapi object itself. BM25Okapi rebuilds in
milliseconds for prototype scale and avoids version-skew
issues with the rank_bm25 library.

For corpora beyond ~1M documents, replace this with
OpenSearch (which has BM25 + k-NN both built in).
"""

import os
import re
import pickle
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.config.settings import FAISS_INDEX_DIR, RETRIEVAL_TOP_K
from app.utils.logger import logger


# =========================================================
# BM25 index file location
# =========================================================
# We co-locate the BM25 store under the same vector_db/
# directory so the dense + sparse indexes travel together.

BM25_STORE_PATH = Path(FAISS_INDEX_DIR).parent / "bm25_store.pkl"


# =========================================================
# Tokenizer
# =========================================================
# BM25 works on TOKENS, not raw strings. We use a simple
# lowercase + punctuation-strip + whitespace-split tokenizer.
# Good enough for prototype; for production swap in a real
# tokenizer (spaCy, NLTK, or HuggingFace tokenizers).

_TOKEN_PATTERN = re.compile(r"[^\w\s]")


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = _TOKEN_PATTERN.sub(" ", text)
    return text.split()


# =========================================================
# Persistence helpers
# =========================================================

def _store_exists() -> bool:
    return BM25_STORE_PATH.exists()


def _load_store():
    """
    Load the persisted (tokenized_corpus, documents) tuple.

    Returns:
        tuple[list[list[str]], list[Document]]
        or (None, None) if no store exists yet.
    """
    if not _store_exists():
        return None, None

    with open(BM25_STORE_PATH, "rb") as f:
        data = pickle.load(f)

    return data["tokenized_corpus"], data["documents"]


def _save_store(tokenized_corpus, documents):
    """Persist the corpus so we can rebuild the BM25 index on next process start."""
    BM25_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BM25_STORE_PATH, "wb") as f:
        pickle.dump(
            {
                "tokenized_corpus": tokenized_corpus,
                "documents":        documents,
            },
            f,
        )


# =========================================================
# Public API -- Indexing
# =========================================================

def upsert_chunks(chunks: List[Document]):
    """
    Add new chunks to the BM25 index.

    Behavior:
    ---------
    - Load existing corpus (if any)
    - Tokenize new chunks
    - Append to the corpus
    - Persist back to disk

    Note: BM25 IDF stats depend on the WHOLE corpus, so we
    rebuild the BM25Okapi object lazily at search time
    rather than caching it across ingests.
    """

    if not chunks:
        logger.warning("bm25_upsert_called_with_empty_list")
        return {"status": "no-op", "chunks_added": 0}

    # ---- Load existing corpus -----------------------------
    existing_corpus, existing_docs = _load_store()

    if existing_corpus is None:
        tokenized_corpus = []
        documents = []
        operation = "created"
    else:
        tokenized_corpus = existing_corpus
        documents = existing_docs
        operation = "appended"

    # ---- Tokenize and append new chunks -------------------
    for chunk in chunks:
        tokenized_corpus.append(_tokenize(chunk.page_content))
        documents.append(chunk)

    # ---- Persist ------------------------------------------
    _save_store(tokenized_corpus, documents)

    logger.info(
        "bm25_index_upsert_complete",
        operation=operation,
        chunks_added=len(chunks),
        store_total_documents=len(documents),
        store_path=str(BM25_STORE_PATH),
    )

    return {
        "status": "success",
        "operation": operation,
        "chunks_added": len(chunks),
        "store_total_documents": len(documents),
    }


# =========================================================
# Public API -- Retrieval
# =========================================================

def search(
    query: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[dict] = None,
):
    """
    Run BM25 sparse retrieval against the persisted corpus.

    Args:
        query           : natural-language or keyword query
        top_k           : how many results to return
        metadata_filter : optional dict; results whose metadata
                          doesn't match ALL keys are dropped.

    Returns:
        List[dict]: each result has
            - content       : chunk text
            - metadata      : full chunk metadata
            - bm25_score    : raw BM25 score (higher = better)
    """

    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    if not _store_exists():
        logger.error(
            "bm25_store_missing",
            store_path=str(BM25_STORE_PATH),
            hint="ingest at least one document before searching",
        )
        raise FileNotFoundError(
            f"No BM25 store found at {BM25_STORE_PATH}. Ingest first."
        )

    tokenized_corpus, documents = _load_store()

    # Rebuild BM25Okapi -- fast for prototype scale.
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = _tokenize(query)

    logger.info(
        "bm25_search_started",
        query_preview=query[:200],
        top_k=top_k,
        metadata_filter=metadata_filter,
        corpus_size=len(documents),
    )

    # get_scores returns one score per document, in corpus order
    scores = bm25.get_scores(tokenized_query)

    # Pair (doc, score) and sort descending by score
    paired = list(zip(documents, scores))
    paired.sort(key=lambda x: x[1], reverse=True)

    # ----------------------------------------------------
    # Metadata post-filtering
    # ----------------------------------------------------
    # BM25Okapi has no native pre-filter -- we filter the
    # ranked list afterwards. Acceptable at prototype scale.
    # ----------------------------------------------------
    if metadata_filter:
        filtered = []
        for doc, score in paired:
            if all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                filtered.append((doc, score))
        paired = filtered

    # Take top_k
    top = paired[:top_k]

    results = [
        {
            "content":    doc.page_content,
            "metadata":   doc.metadata,
            "bm25_score": float(score),
        }
        for doc, score in top
    ]

    logger.info(
        "bm25_search_completed",
        query_preview=query[:200],
        results_returned=len(results),
        top_score=results[0]["bm25_score"] if results else None,
    )

    return results


# =========================================================
# Public API -- Stats
# =========================================================

def get_store_stats():
    """Diagnostics about the persisted BM25 store."""
    if not _store_exists():
        return {"exists": False, "store_path": str(BM25_STORE_PATH)}

    _, documents = _load_store()

    return {
        "exists": True,
        "store_path": str(BM25_STORE_PATH),
        "total_documents": len(documents),
    }