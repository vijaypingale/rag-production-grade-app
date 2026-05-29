"""
Embedding Generation Module (Bedrock-backed)

Responsibilities:
-----------------
- Generate vector embeddings for a list of chunks
- Hide provider details from the rest of the pipeline
- Batch requests for cost / latency efficiency

Enterprise Principles:
----------------------
- provider abstraction (Bedrock / OpenAI / Cohere are interchangeable)
- reusable embedding layer
- centralized configuration
- structured observability

Provider selection:
-------------------
The active provider is decided in settings.EMBEDDING_PROVIDER.
- "bedrock" (default) -> app/utils/bedrock_client.py
- "openai"  (legacy)  -> app/utils/openai_client.py

Switching providers is a one-line env-var change:
    EMBEDDING_PROVIDER=openai
"""

import time

from app.utils.logger import logger

from app.config.settings import (
    EMBEDDING_PROVIDER,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL as OPENAI_EMBEDDING_MODEL,
)

# Bedrock config is optional (commented out when using OpenAI)
try:
    from app.config.settings import BEDROCK_EMBEDDING_MODEL
except ImportError:
    BEDROCK_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"  # fallback


# =========================================================
# Provider-aware embedding model factory
# =========================================================
# We avoid importing both providers at module load time so
# that switching providers doesn't force users to install
# unused SDKs. Imports happen lazily inside the factory.

def _get_embedding_model():
    """
    Return the embedding model configured for this deployment.

    Returns:
        An object exposing embed_documents(list[str]) -> list[list[float]]
    """

    provider = EMBEDDING_PROVIDER.lower()

    if provider == "bedrock":
        from app.utils.bedrock_client import get_embedding_model
        return get_embedding_model(), BEDROCK_EMBEDDING_MODEL

    if provider == "openai":
        from app.utils.openai_client import get_embedding_model
        return get_embedding_model(), OPENAI_EMBEDDING_MODEL

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: {provider!r}. "
        f"Supported: 'bedrock', 'openai'."
    )


# =========================================================
# Public API
# =========================================================

def get_embedding_model():
    """
    Return ONLY the model (without the model id label).

    Some callers (text_splitter.SemanticChunker) need the
    embedding model itself, not the provider-id pair.
    """

    model, _ = _get_embedding_model()
    return model


def generate_embeddings(chunks):
    """
    Generate embeddings for a batch of chunks.

    Args:
        chunks: List[LangChain Document] - chunks produced by
                app/ingestion/text_splitter.py. Each chunk has
                .page_content (the text) and .metadata.

    Returns:
        List[List[float]]: One vector per chunk, in the same
                           order as the input. Caller is
                           responsible for binding embeddings
                           back to chunks (or use the
                           FAISS.from_documents() shortcut).

    Notes on batching:
    ------------------
    LangChain's embedding wrappers handle batching internally,
    but we still chunk the input list to bound peak memory
    and to log batch-level progress for large ingests.
    """

    embedding_model, model_id = _get_embedding_model()

    total_chunks = len(chunks)

    logger.info(
        "embedding_generation_started",
        provider=EMBEDDING_PROVIDER,
        embedding_model=model_id,
        total_chunks=total_chunks,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    start = time.perf_counter()

    # ----------------------------------------------------
    # Extract texts. Order is preserved, which is critical
    # for binding embeddings back to chunks downstream.
    # ----------------------------------------------------
    texts = [chunk.page_content for chunk in chunks]

    # ----------------------------------------------------
    # Batched embedding generation
    # ----------------------------------------------------
    # We loop in slices of EMBEDDING_BATCH_SIZE so that on
    # large documents we get visible progress logs and we
    # never hold the full token payload in a single request.
    # ----------------------------------------------------

    embeddings: list = []

    for batch_start in range(0, total_chunks, EMBEDDING_BATCH_SIZE):

        batch_end = min(
            batch_start + EMBEDDING_BATCH_SIZE,
            total_chunks
        )

        batch_texts = texts[batch_start:batch_end]

        batch_embeddings = embedding_model.embed_documents(batch_texts)

        embeddings.extend(batch_embeddings)

        logger.info(
            "embedding_batch_completed",
            batch_start=batch_start,
            batch_end=batch_end,
            running_total=len(embeddings),
        )

    duration = time.perf_counter() - start

    logger.info(
        "embedding_generation_completed",

        # Provider + model context
        provider=EMBEDDING_PROVIDER,
        embedding_model=model_id,

        # Quantitative metrics
        total_embeddings=len(embeddings),
        embedding_dimension=len(embeddings[0]) if embeddings else 0,
        source_chunks=total_chunks,

        # Validation preview (first 5 dimensions of first vector)
        sample_embedding_preview=embeddings[0][:5] if embeddings else None,

        # Performance
        duration_seconds=round(duration, 4),
    )

    return embeddings
