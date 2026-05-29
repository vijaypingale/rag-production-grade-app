"""
Document Chunking Module (Hybrid Strategy + Rich Metadata)

Responsibilities:
-----------------
- Split documents into retrieval-ready chunks
- Support multiple strategies: recursive | semantic | hybrid
- Inject enterprise metadata onto every chunk
- Preserve contextual overlap

This module directly impacts:
- retrieval quality
- semantic relevance
- hallucination reduction

---------------------------------------------------------------
Why three strategies?
---------------------------------------------------------------
1) RecursiveCharacterTextSplitter (RCTS)
   - Splits text recursively at natural separators
     (paragraph -> sentence -> word -> char)
   - Deterministic, fast, no model calls
   - Best when documents have clean structure
     (Markdown, well-formatted PDFs)

2) SemanticChunker
   - Embeds each sentence, splits at large embedding-distance
     jumps between consecutive sentences
   - Produces chunks that match TOPIC boundaries, not just
     character counts
   - Higher quality on prose / reports / policy documents
   - Slower and costs embedding calls during ingestion

3) HYBRID  (semantic-first, size-enforce-second)
   - Step 1: SemanticChunker groups sentences by topic
   - Step 2: RCTS enforces CHUNK_SIZE on any oversized
     semantic chunk
   - Result: chunks that respect topic boundaries AND
     stay within token budget. This is the production
     default.

Diagram of the Hybrid pipeline:

    raw text
       |
       v
    SemanticChunker  (topic-aware splitting)
       |
       v
    semantic chunks (may exceed CHUNK_SIZE)
       |
       v
    RecursiveCharacterTextSplitter (size enforcement + overlap)
       |
       v
    final chunks  -->  embed  -->  FAISS
"""

import uuid
import time
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from app.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_STRATEGY,
    SEMANTIC_BREAKPOINT_TYPE,
    SEMANTIC_BREAKPOINT_AMOUNT,
)

from app.utils.logger import logger


# =========================================================
# Internal helpers
# =========================================================

def _build_recursive_splitter():
    """
    Build a RecursiveCharacterTextSplitter with our defaults.

    Why a helper:
    - the splitter is used by BOTH "recursive" and "hybrid"
      strategies, so we centralize its construction.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )


def _build_semantic_splitter(embedding_model):
    """
    Build a SemanticChunker for topic-aware boundaries.

    SemanticChunker requires an embedding model because it
    computes embeddings for sentences during splitting.
    We pass in the same embedding model used downstream so
    chunk boundaries and retrieval embeddings are aligned.
    """

    return SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
    )


def _enrich_chunk_metadata(
    chunks,
    source_filename,
    doc_id,
    doc_type,
    strategy_used,
):
    """
    Inject enterprise-grade metadata onto every chunk.

    Why this matters:
    -----------------
    Vector search alone returns chunks; without metadata
    you can NOT:
    - cite the source document
    - filter by tenant / department / access level
    - re-index only changed documents
    - debug "why did retrieval return THIS chunk"

    Fields added to chunk.metadata:
    -------------------------------
    - doc_id          : UUID for the parent document
    - chunk_id        : UUID for this chunk
    - chunk_index     : sequential position within the doc
    - total_chunks    : how many chunks the doc became
    - source          : original filename
    - doc_type        : pdf | docx | html | md | txt
    - chunk_strategy  : recursive | semantic | hybrid
    - chunk_size      : actual character length
    - ingested_at     : ISO-8601 UTC timestamp
    """

    total = len(chunks)
    ingested_at = datetime.now(timezone.utc).isoformat()

    for index, chunk in enumerate(chunks):

        # Preserve whatever metadata the loader already set
        # (e.g. PyMuPDFLoader puts "page", "source" etc).
        chunk.metadata.update({
            "doc_id":         doc_id,
            "chunk_id":       str(uuid.uuid4()),
            "chunk_index":    index,
            "total_chunks":   total,
            "source":         source_filename,
            "doc_type":       doc_type,
            "chunk_strategy": strategy_used,
            "chunk_size":     len(chunk.page_content),
            "ingested_at":    ingested_at,
        })

    return chunks


# =========================================================
# Public API
# =========================================================

def split_documents(
    documents,
    source_filename: str,
    doc_type: str = "pdf",
    doc_id: str = None,
    embedding_model=None,
    strategy: str = None,
):
    """
    Split LangChain Documents into chunks using the chosen
    strategy and inject enterprise metadata onto every chunk.

    Args:
        documents          : List[Document] from a loader
        source_filename    : Original filename, used in metadata
        doc_type           : "pdf" | "docx" | ...
        doc_id             : Optional caller-supplied UUID.
                             If None, one is generated here.
        embedding_model    : Required for "semantic" / "hybrid".
                             Ignored for "recursive".
        strategy           : "recursive" | "semantic" | "hybrid".
                             Defaults to settings.CHUNK_STRATEGY.

    Returns:
        List[LangChain Document]: chunks with rich metadata,
        ready for embedding generation and vector DB ingest.
    """

    # ----------------------------------------------------
    # Resolve strategy & doc_id (caller may override)
    # ----------------------------------------------------
    strategy = (strategy or CHUNK_STRATEGY).lower()
    doc_id = doc_id or str(uuid.uuid4())

    logger.info(
        "document_chunking_started",
        strategy=strategy,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        doc_id=doc_id,
        source=source_filename,
        doc_type=doc_type,
        input_documents=len(documents),
    )

    start = time.perf_counter()

    # ----------------------------------------------------
    # Strategy dispatch
    # ----------------------------------------------------
    if strategy == "recursive":

        # Pure size-based recursive splitting.
        # Fast and deterministic. No model calls.
        splitter = _build_recursive_splitter()
        chunks = splitter.split_documents(documents)

    elif strategy == "semantic":

        # Pure topic-based semantic splitting.
        # WARNING: produces chunks of variable size, some may
        # exceed CHUNK_SIZE. Use "hybrid" if downstream
        # consumers (LLM context) need a size guarantee.
        if embedding_model is None:
            raise ValueError(
                "embedding_model is required for 'semantic' "
                "chunking. Pass get_embedding_model() to "
                "split_documents()."
            )

        splitter = _build_semantic_splitter(embedding_model)
        chunks = splitter.split_documents(documents)

    elif strategy == "hybrid":

        # ------------------------------------------------
        # HYBRID STRATEGY (recommended production default)
        # ------------------------------------------------
        # Step 1: split by SEMANTIC similarity (topic boundaries)
        # Step 2: ENFORCE max size on any oversized semantic
        #         chunks using RecursiveCharacterTextSplitter
        #
        # The two-pass approach keeps topic coherence
        # WITHOUT blowing the LLM context budget.
        # ------------------------------------------------
        if embedding_model is None:
            raise ValueError(
                "embedding_model is required for 'hybrid' "
                "chunking. Pass get_embedding_model() to "
                "split_documents()."
            )

        # Step 1: semantic boundaries
        semantic_splitter = _build_semantic_splitter(embedding_model)
        semantic_chunks = semantic_splitter.split_documents(documents)

        logger.info(
            "hybrid_step1_semantic_complete",
            semantic_chunks=len(semantic_chunks),
            doc_id=doc_id,
        )

        # Step 2: size enforcement + overlap
        # We re-run RCTS over the semantic chunks; any chunk
        # already under CHUNK_SIZE will pass through untouched.
        size_splitter = _build_recursive_splitter()
        chunks = size_splitter.split_documents(semantic_chunks)

        logger.info(
            "hybrid_step2_size_enforced",
            final_chunks=len(chunks),
            doc_id=doc_id,
        )

    else:
        raise ValueError(
            f"Unknown chunk strategy: {strategy!r}. "
            f"Use one of: recursive | semantic | hybrid"
        )

    # ----------------------------------------------------
    # Enterprise metadata enrichment
    # ----------------------------------------------------
    chunks = _enrich_chunk_metadata(
        chunks=chunks,
        source_filename=source_filename,
        doc_id=doc_id,
        doc_type=doc_type,
        strategy_used=strategy,
    )

    duration = time.perf_counter() - start

    logger.info(
        "document_chunking_completed",
        strategy=strategy,
        total_chunks=len(chunks),
        doc_id=doc_id,
        duration_seconds=round(duration, 4),
    )

    if chunks:
        logger.info(
            "sample_chunk_metadata",
            metadata=chunks[0].metadata,
        )
        logger.info(
            "sample_chunk_preview",
            content=chunks[0].page_content[:500],
        )

    return chunks
