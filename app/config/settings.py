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

# =========================================================
# Retrieval Mode Configuration
# =========================================================
# Selects how the retrieval service answers a query:
#   "dense"  : FAISS vector search only         (current default)
#   "hybrid" : BM25 + dense fused via RRF       (production recommended)
#   "mmr"    : FAISS Maximal Marginal Relevance for diversity
#
# Override at request time via the API, or globally via env var.

RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

# MMR diversity-vs-relevance balance.
# 1.0 = pure relevance (same as dense), 0.0 = pure diversity.
# 0.5 is the standard balanced default.

MMR_LAMBDA = 0.5

# How many candidates to pull from each retriever in hybrid
# mode before RRF fusion. Higher = better recall, slower.

HYBRID_FETCH_K = 20

# =========================================================
# Cohere Rerank Configuration
# =========================================================
# Cohere Rerank v3 is the most widely deployed hosted
# cross-encoder reranker in production RAG today. It re-
# scores retrieval candidates by reading the query AND
# each candidate document together in one transformer pass,
# producing significantly more accurate rankings than the
# bi-encoder retrieval stage alone.
#
# Model choices:
#   "rerank-v3.5"                  -- newest, best quality, multilingual
#   "rerank-multilingual-v3.0"     -- multilingual, stable
#   "rerank-english-v3.0"          -- English-only, slightly faster
#
# Pricing: $2 per 1,000 search units (1 unit ≈ 1 query + 100 docs)
# Free tier: 100 calls/minute, no credit card required.

COHERE_RERANK_MODEL = os.getenv(
    "COHERE_RERANK_MODEL",
    "rerank-v3.5"
)

# How many candidates to PULL from the retriever before
# reranking. Higher = better recall going into the reranker,
# more API cost. 50 is the sweet spot in most production
# RAG systems.

RERANK_FETCH_K = 50

# How many candidates to KEEP after reranking. This is
# what ultimately gets passed to the LLM (or returned to
# the caller in our current prototype).

RERANK_TOP_K = 5


# =========================================================
# Grounding Configuration
# =========================================================
# Minimum rerank score required for the top chunk before
# the generation layer is allowed to produce an answer.
#
# If the highest rerank score across all retrieved chunks
# is below this threshold, the system returns a canned
# "I don't have enough information" response instead of
# calling the LLM at all.
#
# Why 0.20:
#   Cohere Rerank v3 scores range roughly 0.0 - 1.0.
#   Empirically, scores below 0.20 indicate the retrieved
#   content is weakly related to the query — generating an
#   answer from such content risks hallucination.
#
# Tune upward (e.g. 0.30) for stricter grounding;
# tune downward (e.g. 0.10) if you see too many false
# "I don't know" responses on valid queries.

GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.05"))
# NOTE: Initial estimate was 0.20 but rerank-v3.5 scores for relevant chunks
# in this corpus land in the 0.05 - 0.40 range. Lowered to 0.05 to avoid
# false "I don't know" responses. Tune upward after observing score
# distributions across more queries (Section 12 — quality dashboard).


# =========================================================
# LLM Provider Configuration
# =========================================================
# Selects which LLM backend handles generation.
# Currently supported: "openai" | "bedrock"
#
# "openai"  : uses ChatOpenAI (langchain-openai)
# "bedrock" : uses ChatBedrock (langchain-aws) — requires
#             valid AWS credentials and region in .env
#
# Override via env var: LLM_PROVIDER=bedrock

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")


# =========================================================
# OpenAI LLM Model Configuration
# =========================================================
# gpt-4o-mini is the recommended default:
#   - 128k context window
#   - Strong instruction-following + citation adherence
#   - ~10x cheaper than gpt-4o for the same task quality
#     on well-structured RAG prompts
#
# Other options:
#   "gpt-4o"          — highest quality, higher cost
#   "gpt-3.5-turbo"   — faster, cheapest, weaker reasoning

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# =========================================================
# AWS Bedrock LLM Configuration (used when LLM_PROVIDER=bedrock)
# =========================================================
# Model IDs follow Bedrock's ARN-style naming.
# Uncomment and set in .env to activate Bedrock generation.
#
# Recommended models:
#   "anthropic.claude-3-5-sonnet-20241022-v2:0"  — best quality
#   "anthropic.claude-3-haiku-20240307-v1:0"     — fast + cheap
#   "amazon.nova-pro-v1:0"                        — Amazon native

BEDROCK_LLM_MODEL = os.getenv(
    "BEDROCK_LLM_MODEL",
    "anthropic.claude-3-5-sonnet-20241022-v2:0"
)

BEDROCK_LLM_REGION = os.getenv("AWS_REGION", "us-east-1")


# =========================================================
# LLM Token Budget
# =========================================================
# MAX_CONTEXT_TOKENS: hard cap on the assembled context
# string passed to the LLM. This is the same value used
# by the context assembler (Section 7).
#
# Budget breakdown for gpt-4o-mini (128k window):
#   ~300  tokens  — system prompt
#   ~100  tokens  — user question
#   6000  tokens  — assembled context  ← this constant
#   ~1500 tokens  — answer headroom
#   ------
#   ~7900 tokens total — well within the 128k window
#
# If you switch to a model with a smaller context window
# (e.g. gpt-3.5-turbo at 16k), lower this value accordingly.

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))

# Maximum tokens the LLM is allowed to generate in its answer.
# 1024 is a safe default — long enough for a thorough answer,
# short enough to keep costs predictable.

MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))


# =========================================================
# LLM Sampling Configuration
# =========================================================
# Temperature controls answer randomness.
# 0.0 = fully deterministic (best for factual RAG answers)
# Higher values introduce creativity — not appropriate for
# grounded Q&A where we want the LLM to stick to context.

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))


# =========================================================
# LLM Retry & Timeout Configuration
# =========================================================
# Production LLM calls can fail transiently (rate limits,
# network blips, provider maintenance windows). We retry
# with exponential backoff before surfacing a 503 to the
# caller.
#
# LLM_TIMEOUT_SECONDS : per-attempt wall-clock timeout.
#                       60s is generous for gpt-4o-mini;
#                       lower to 30s for latency-sensitive apps.
# LLM_MAX_RETRIES     : attempts before giving up.
#                       3 = initial + 2 retries.

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES     = int(os.getenv("LLM_MAX_RETRIES", "3"))


# =========================================================
# Faithfulness / Hallucination Control (Section 9)
# =========================================================
# AFTER the LLM generates an answer, we verify it is actually
# supported by the retrieved context. This is the "verify after"
# layer of the three-tier hallucination defense:
#   1. Grounding gate    (before generation) — GROUNDING_THRESHOLD above
#   2. Faithfulness check (after generation) — this block
#   3. Citation enforcement (programmatic)   — no config needed
#
# Technique (industry standard):
#   Decompose the answer into atomic claims, then use an LLM-as-judge
#   (NLI-style) to classify each claim against the context as
#   supported / unsupported / unverifiable. The faithfulness score is
#   supported_claims / total_claims.
#
# FAITHFULNESS_CHECK_ENABLED:
#   The check costs one extra LLM call (added latency + cost). Enable in
#   production / regulated environments; disable for latency-sensitive or
#   cost-sensitive paths. Toggle via env: FAITHFULNESS_CHECK_ENABLED=false
#
# FAITHFULNESS_THRESHOLD:
#   Minimum score for an answer to be considered trustworthy.
#   Industry guidance: >=0.85 baseline for regulated environments,
#   >0.9 as a stricter target. Below threshold => answer is flagged
#   as low-faithfulness (caller decides whether to suppress).

FAITHFULNESS_CHECK_ENABLED = os.getenv(
    "FAITHFULNESS_CHECK_ENABLED", "true"
).lower() == "true"

FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.85"))


# =========================================================
# Observability — OpenTelemetry (Section 12)
# =========================================================
# We instrument the pipeline with OpenTelemetry (vendor-neutral) so the
# SAME instrumentation can export to ANY backend by changing only the
# exporter — console for local dev, Datadog/Grafana in production.
#
# OTEL_ENABLED:
#   Master switch. When false, instrumentation is a no-op (zero overhead),
#   so tests and offline scripts aren't affected.
# OTEL_EXPORTER:
#   "console" -> print spans to stdout (safe local dev, no account/cost)
#   "otlp"    -> send to an OTLP endpoint (Datadog Agent, Grafana, etc.)
# OTEL_SERVICE_NAME:
#   Logical service name shown in the backend UI.

OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"
OTEL_EXPORTER = os.getenv("OTEL_EXPORTER", "console")   # console | otlp
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "rag-production-app")


# =========================================================
# Model Pricing (for cost-per-query tracking)
# =========================================================
# USD per 1,000,000 tokens. Used to compute cost from token usage so
# observability can report cost-per-query. Update when provider prices
# change. (gpt-4o-mini / text-embedding-3-small list prices.)

MODEL_PRICING_PER_1M = {
    "gpt-4o-mini":            {"input": 0.15,  "output": 0.60},
    "gpt-4o":                 {"input": 2.50,  "output": 10.00},
    "text-embedding-3-small": {"input": 0.02,  "output": 0.00},
}


# =========================================================
# Semantic Cache (Section 13)
# =========================================================
# Cache LLM answers keyed by the query's MEANING (its embedding), not its
# exact text. A new query that is semantically close to a past one (cosine
# similarity >= threshold) returns the cached answer in milliseconds and
# skips retrieval + reranking + the LLM call — the expensive parts.
#
# CACHE_ENABLED:
#   Master switch. When false, every query runs the full pipeline.
# CACHE_BACKEND:
#   "memory" -> in-process cache (works now, no infra; lost on restart,
#               not shared across instances). Good for dev/demo.
#   "redis"  -> RedisVL SemanticCache (shared, persistent) — production.
# SEMANTIC_CACHE_THRESHOLD:
#   Cosine similarity (0-1) a query must reach to count as a cache HIT.
#   Industry guidance for factual/RAG: >= 0.92 (strict) so we never serve a
#   cached answer to a subtly different question. Lower = more hits but more
#   risk of wrong answers.
# CACHE_TTL_SECONDS:
#   How long a cached answer stays valid. 24h default — bounds staleness when
#   source documents change (combined with cache-busting on re-ingestion).
# REDIS_URL:
#   Connection string used only when CACHE_BACKEND="redis".

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_BACKEND = os.getenv("CACHE_BACKEND", "memory")          # memory | redis
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))   # 24 hours
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")