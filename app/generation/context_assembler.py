"""
Context Assembler — Section 7

Responsibilities:
-----------------
- Take reranked chunks (sorted best-first by rerank_score)
- Count tokens using tiktoken to stay within the LLM context budget
- Assign citation tags [1], [2], ... to each included chunk
- Return assembled context string + citation map + token stats

Why token budgeting matters:
-----------------------------
gpt-4o-mini has a 128k context window, but sending 50 raw chunks
is expensive and hurts answer quality (LLMs lose focus in very long
contexts). We cap assembled context at max_tokens (default 6000)
to leave room for:
  - System prompt        (~300 tokens)
  - User question        (~100 tokens)
  - LLM answer headroom (~1500 tokens)

Why citation tags:
-------------------
The generation layer will instruct the LLM to cite sources as [N].
The citation_map returned here links each N back to the original
chunk metadata (source file, page, chunk_id) so the API response
can surface exact provenance to the user.
"""

from typing import List, Optional
import tiktoken

from app.utils.logger import logger


# =========================================================
# Constants
# =========================================================

DEFAULT_MAX_TOKENS = 6000   # max tokens for assembled context
DEFAULT_MODEL      = "gpt-4o-mini"

# Separator inserted between chunks in the context string.
# Blank line makes it easier for the LLM to distinguish chunks.
CHUNK_SEPARATOR = "\n\n"


# =========================================================
# Internal helpers
# =========================================================

def _get_encoder(model: str) -> tiktoken.Encoding:
    """
    Return the tiktoken encoder for the given model.

    Falls back to cl100k_base (used by gpt-4, gpt-3.5, gpt-4o
    family) if the model is not found in tiktoken's registry.
    This keeps the code forward-compatible with new model names.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(
            "tiktoken_model_not_found",
            model=model,
            fallback="cl100k_base",
        )
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """Return the number of tokens in text using the given encoder."""
    return len(encoder.encode(text))


# =========================================================
# Public API
# =========================================================

def assemble_context(
    chunks: List[dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Build a token-budgeted, citation-tagged context string from
    reranked chunks.

    Args:
        chunks      : list of dicts from the reranker, each with:
                        - "content"      : chunk text
                        - "metadata"     : dict (source, page, chunk_id, ...)
                        - "rerank_score" : float (already sorted best-first)
        max_tokens  : hard cap on total tokens in the context string.
                      Default 6000. Caller can override per request.
        model       : model name used to pick the right tokenizer.
                      Default "gpt-4o-mini".

    Returns:
        dict with keys:
            - "context"       : str  — assembled context with [N] tags
            - "citation_map"  : dict — { N (int): metadata dict }
            - "chunks_used"   : int  — how many chunks made it in
            - "chunks_total"  : int  — how many chunks were offered
            - "tokens_used"   : int  — total tokens in context string
    """

    if not chunks:
        logger.warning("context_assembler_empty_input")
        return {
            "context":      "",
            "citation_map": {},
            "chunks_used":  0,
            "chunks_total": 0,
            "tokens_used":  0,
        }

    encoder     = _get_encoder(model)
    citation_map: dict  = {}
    parts: List[str]    = []   # assembled text segments
    tokens_used: int    = 0
    citation_num: int   = 1    # starts at 1

    for chunk in chunks:

        content  = chunk.get("content", "").strip()
        metadata = chunk.get("metadata", {})

        if not content:
            continue

        # Format this chunk as: "[N] <content>"
        tagged = f"[{citation_num}] {content}"

        # Count tokens for this tagged chunk + separator
        chunk_tokens = _count_tokens(
            tagged + CHUNK_SEPARATOR, encoder
        )

        # If adding this chunk exceeds budget → stop
        if tokens_used + chunk_tokens > max_tokens:
            logger.info(
                "context_assembler_budget_reached",
                citation_num=citation_num,
                tokens_used=tokens_used,
                max_tokens=max_tokens,
            )
            break

        # Accept this chunk
        parts.append(tagged)
        citation_map[citation_num] = {
            "source":   metadata.get("source", "unknown"),
            "page":     metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "rerank_score": chunk.get("rerank_score"),
        }
        tokens_used  += chunk_tokens
        citation_num += 1

    context = CHUNK_SEPARATOR.join(parts)

    logger.info(
        "context_assembled",
        chunks_used=citation_num - 1,
        chunks_total=len(chunks),
        tokens_used=tokens_used,
        max_tokens=max_tokens,
        model=model,
    )

    return {
        "context":      context,
        "citation_map": citation_map,
        "chunks_used":  citation_num - 1,
        "chunks_total": len(chunks),
        "tokens_used":  tokens_used,
    }
