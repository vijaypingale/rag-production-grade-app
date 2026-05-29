"""
FAISS Vector Store Module

Responsibilities:
-----------------
- Build / update a persistent FAISS index from chunks
- Persist index + metadata to disk
- Load existing index on demand
- Run similarity search with optional metadata filtering

Why FAISS for the prototype:
----------------------------
- in-process, no external service needed
- great single-node performance for up to a few million vectors
- LangChain wrapper supports metadata filtering and persistence
- easy to swap for a managed vector DB (OpenSearch, Pinecone,
  Qdrant) later -- see app/vectorstores/opensearch_store.py for
  the OpenSearch migration sketch.

Index lifecycle:
----------------
1. First ingest -> FAISS.from_documents()      (creates new index)
2. Later ingests -> existing.add_documents()    (appends to index)
3. Both paths -> vector_store.save_local()       (persist to disk)
4. Query path  -> load_local() + similarity_search_with_score()

Files written under FAISS_INDEX_DIR:
    rag_index.faiss   binary FAISS index
    rag_index.pkl     pickled docstore + index_to_docstore_id map
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.config.settings import (
    FAISS_INDEX_DIR,
    FAISS_INDEX_NAME,
    RETRIEVAL_TOP_K,
)

from app.embeddings.embedding_generator import get_embedding_model
from app.utils.logger import logger


# =========================================================
# Internal helpers
# =========================================================

def _index_exists() -> bool:
    """
    Check whether a persisted FAISS index already exists.

    LangChain saves two files per index:
      <name>.faiss  -- raw FAISS binary
      <name>.pkl    -- docstore + id mapping
    Both must be present for a successful load.
    """

    index_dir = Path(FAISS_INDEX_DIR)
    faiss_file = index_dir / f"{FAISS_INDEX_NAME}.faiss"
    pkl_file   = index_dir / f"{FAISS_INDEX_NAME}.pkl"

    return faiss_file.exists() and pkl_file.exists()


def _ensure_index_dir():
    """Create the index directory if it doesn't exist yet."""
    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)


# =========================================================
# Public API -- Indexing
# =========================================================

def upsert_chunks(chunks):
    """
    Add chunks to the FAISS index (create-or-append).

    Behavior:
    ---------
    - If no index exists on disk, build a new one.
    - If an index exists, load it and append the new chunks.
    - Always persist the resulting index back to disk.

    Args:
        chunks: List[LangChain Document] from text_splitter.
                Each chunk MUST already carry enterprise metadata
                (doc_id, chunk_id, source, etc.) -- see
                app/ingestion/text_splitter._enrich_chunk_metadata().

    Returns:
        dict: {
            "status": "success",
            "chunks_added": int,
            "index_total_vectors": int,   # AFTER this upsert
            "index_path": str,
            "operation": "created" | "appended",
        }
    """

    if not chunks:
        logger.warning("upsert_chunks_called_with_empty_list")
        return {
            "status": "no-op",
            "chunks_added": 0,
        }

    _ensure_index_dir()

    embedding_model = get_embedding_model()

    if _index_exists():

        # ---- Append path -----------------------------------
        # Loading the index requires allow_dangerous_deserialization
        # because LangChain unpickles the docstore. This is safe
        # when WE created the file ourselves; treat any externally
        # supplied index as untrusted.
        logger.info(
            "faiss_index_loading_existing",
            index_dir=FAISS_INDEX_DIR,
            index_name=FAISS_INDEX_NAME,
        )

        vector_store = FAISS.load_local(
            folder_path=FAISS_INDEX_DIR,
            embeddings=embedding_model,
            index_name=FAISS_INDEX_NAME,
            allow_dangerous_deserialization=True,
        )

        vector_store.add_documents(chunks)
        operation = "appended"

    else:

        # ---- Create path -----------------------------------
        logger.info(
            "faiss_index_creating_new",
            index_dir=FAISS_INDEX_DIR,
            index_name=FAISS_INDEX_NAME,
            initial_chunks=len(chunks),
        )

        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )

        operation = "created"

    # ---- Persist back to disk ------------------------------
    vector_store.save_local(
        folder_path=FAISS_INDEX_DIR,
        index_name=FAISS_INDEX_NAME,
    )

    # FAISS exposes its raw index as .index; ntotal = vector count
    total_vectors = vector_store.index.ntotal

    logger.info(
        "faiss_index_upsert_complete",
        operation=operation,
        chunks_added=len(chunks),
        index_total_vectors=total_vectors,
        index_path=FAISS_INDEX_DIR,
    )

    return {
        "status": "success",
        "operation": operation,
        "chunks_added": len(chunks),
        "index_total_vectors": total_vectors,
        "index_path": FAISS_INDEX_DIR,
    }


# =========================================================
# Public API -- Retrieval
# =========================================================

def similarity_search(
    query: str,
    top_k: int = None,
    metadata_filter: dict = None,
):
    """
    Run similarity search against the persisted FAISS index.

    Args:
        query           : natural language query string
        top_k           : how many chunks to return (default: settings)
        metadata_filter : optional dict for metadata pre-filtering.
                          Example: {"doc_type": "pdf"}
                          Example: {"source": "policy.pdf"}
                          Example: {"doc_id": "abc-123"}
                          Multiple keys are AND-ed.

    Returns:
        List[dict]: each result has
            - content        : chunk text
            - metadata       : full chunk metadata
            - similarity     : score (lower = closer for L2;
                               LangChain returns L2 distance)

    Why metadata filtering matters:
    -------------------------------
    Pre-filtering is THE mechanism for enforcing:
      - tenant isolation       (filter by tenant_id)
      - access control         (filter by access_level)
      - freshness              (filter by ingested_at)
      - scope                  (filter by department / source)
    without it, vector search is unsafe in multi-tenant systems.
    """

    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    if not _index_exists():
        logger.error(
            "faiss_index_missing",
            index_dir=FAISS_INDEX_DIR,
            hint="ingest at least one document before searching"
        )
        raise FileNotFoundError(
            f"No FAISS index found at {FAISS_INDEX_DIR}. "
            "Ingest documents first."
        )

    embedding_model = get_embedding_model()

    vector_store = FAISS.load_local(
        folder_path=FAISS_INDEX_DIR,
        embeddings=embedding_model,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

    logger.info(
        "faiss_similarity_search_started",
        query_preview=query[:200],
        top_k=top_k,
        metadata_filter=metadata_filter,
        index_total_vectors=vector_store.index.ntotal,
    )

    # ----------------------------------------------------
    # LangChain's FAISS wrapper supports a `filter` kwarg
    # which applies AFTER the ANN search. For prototype-
    # scale corpora this is fine; at 100K+ vectors prefer
    # vector DBs with native PRE-filter support (OpenSearch,
    # Qdrant, Pinecone) -- see opensearch_store.py.
    # ----------------------------------------------------
    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter=metadata_filter,
    )

    formatted = []

    for document, score in results:

        formatted.append({
            "content":    document.page_content,
            "metadata":   document.metadata,
            "similarity": float(score),
        })

    logger.info(
        "faiss_similarity_search_completed",
        query_preview=query[:200],
        results_returned=len(formatted),
        top_score=formatted[0]["similarity"] if formatted else None,
    )

    return formatted


# =========================================================
# Public API -- Stats / Maintenance
# =========================================================

def get_index_stats():
    """
    Lightweight introspection endpoint.

    Useful for /healthz-style diagnostics and for unit tests
    that need to verify ingestion actually persisted vectors.
    """

    if not _index_exists():
        return {
            "exists": False,
            "index_path": FAISS_INDEX_DIR,
        }

    embedding_model = get_embedding_model()

    vector_store = FAISS.load_local(
        folder_path=FAISS_INDEX_DIR,
        embeddings=embedding_model,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

    return {
        "exists": True,
        "index_path": FAISS_INDEX_DIR,
        "index_name": FAISS_INDEX_NAME,
        "total_vectors": vector_store.index.ntotal,
        "dimension": vector_store.index.d,
    }
