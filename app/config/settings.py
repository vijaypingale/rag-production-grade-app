"""
Centralized application configuration.

Enterprise systems avoid hardcoded values.
Configurations should be managed centrally.

NOTE:
-----
All knobs that affect cost (embedding provider/model),
quality (chunk strategy, sizes), or storage location
(vector index path) MUST live here, NOT inside modules.
This makes A/B testing and per-environment overrides easy.
"""

import os

# =========================================================
# PDF / Document Path Configuration
# =========================================================
# Default path used by manual debug runs.
# In production flow, files come through the API and this
# default is no longer required.

PDF_PATH = "data/documents/wiser-provider-supplier-guide.pdf"


# =========================================================
# Chunking Configuration
# =========================================================
# CHUNK_SIZE / CHUNK_OVERLAP control the RecursiveCharacter
# splitter and also the SIZE-ENFORCEMENT pass that runs
# AFTER semantic chunking in the Hybrid strategy.
#
# Typical enterprise defaults:
#   chunk_size    : 800 - 1200 chars
#   chunk_overlap : 10 - 20% of chunk_size
#
# Larger chunks  = more context per retrieval, fewer vectors
# Smaller chunks = more precise retrieval, more vectors

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# =========================================================
# Chunk Strategy Selector
# =========================================================
# Three strategies are supported by app/ingestion/text_splitter.py:
#
#   "recursive" : RecursiveCharacterTextSplitter only (fast, deterministic)
#   "semantic"  : SemanticChunker only (embedding-based, slow but smart)
#   "hybrid"    : SemanticChunker FIRST -> RecursiveCharacterTextSplitter
#                 to enforce max size on any oversized semantic chunks.
#                 This is the recommended production default.
#
# Override via env var:  CHUNK_STRATEGY=hybrid

CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "hybrid")


# =========================================================
# Semantic Chunker Configuration
# =========================================================
# SemanticChunker computes embeddings for each sentence,
# then splits where the embedding distance jumps above a
# threshold. "percentile" with 95 is a sane default - it
# splits at the top 5% most-different sentence boundaries.
#
# Other supported threshold types:
#   "percentile" | "standard_deviation" | "interquartile" | "gradient"

SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_AMOUNT = 95.0


# =========================================================
# Embedding Provider Configuration
# =========================================================
# Provider selector for the embedding layer.
# Currently supported: "bedrock" and "openai" (active).
#
# Override via env var:  EMBEDDING_PROVIDER=openai

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")


# =========================================================
# Bedrock Embedding Configuration (DISABLED)
# =========================================================
# AWS Bedrock support is temporarily disabled in favor of OpenAI.
# These configurations will be re-enabled later.
# To re-activate Bedrock:
# 1. Update .env with valid AWS credentials
# 2. Set EMBEDDING_PROVIDER=bedrock in .env
# 3. Uncomment the BEDROCK_EMBEDDING_MODEL and BEDROCK_REGION lines below
#
# Bedrock model options (when re-enabled):
#   amazon.titan-embed-text-v2:0   (1024 dim, $0.00002/1K tokens, multilingual)
#   cohere.embed-english-v3        (1024 dim, $0.0001/1K tokens, high quality)
#   cohere.embed-multilingual-v3   (1024 dim, multilingual variant)

# BEDROCK_EMBEDDING_MODEL = os.getenv(
#     "BEDROCK_EMBEDDING_MODEL",
#     "amazon.titan-embed-text-v2:0"
# )
#
# BEDROCK_REGION = os.getenv(
#     "AWS_REGION",
#     "us-east-1"
# )

# For now, using OpenAI embeddings (see EMBEDDING_MODEL below)


# =========================================================
# OpenAI Embedding Configuration (ACTIVE)
# =========================================================
# Using OpenAI as the primary embedding provider.
# To switch back to Bedrock, set EMBEDDING_PROVIDER=bedrock in .env
# and uncomment the Bedrock config above.

EMBEDDING_MODEL = "text-embedding-3-small"


# =========================================================
# Embedding Batch Configuration
# =========================================================
# Most providers accept batched embed requests. Batching
# reduces per-request overhead drastically. Bedrock Titan
# accepts up to 25 docs per batch internally; LangChain
# handles batching automatically but we still bound it.

EMBEDDING_BATCH_SIZE = 96


# =========================================================
# Vector Store Configuration
# =========================================================
# We persist the FAISS index + a metadata sidecar to disk
# under vector_db/faiss_index/ so the index survives restarts.
# The langchain FAISS wrapper writes:
#   index.faiss     (the raw FAISS binary index)
#   index.pkl       (docstore + metadata)

FAISS_INDEX_DIR = "vector_db/faiss_index"
FAISS_INDEX_NAME = "rag_index"


# =========================================================
# Retrieval Configuration
# =========================================================
# Top-K is the number of chunks returned by vector search.
# Production defaults: K=4..10 before reranking, K=20..100
# if a reranker stage follows.

RETRIEVAL_TOP_K = 5
