# RAG Production-Grade App — Progress & Handoff

> **Version:** v0.13 &nbsp;|&nbsp; **Last updated:** 2026-06-22
>
> **Purpose:** Single source of truth for project status. If starting a new
> Claude Code / chat session, say: *"Read PROGRESS.md and continue from there."*
> Keep this file updated whenever code is committed or a section advances.

---

## 1. Project Overview

Production-grade RAG (Retrieval-Augmented Generation) application — portfolio +
interview-ready. Modular, pluggable, enterprise patterns.

**Stack:** Python 3.12 · FastAPI · FAISS (vector) + BM25 (keyword) · OpenAI
(`gpt-4o-mini`, `text-embedding-3-small`) / AWS Bedrock (pluggable) · Cohere
Rerank v3.5 · **LangChain 0.3.x (stable, pinned)** · RAGAS 0.2.15 · structlog · pytest.

> ⚠️ **LangChain is pinned to the 0.3.x line (see requirements.txt), NOT 1.x.**
> Reason: RAGAS 0.2.x (Section 10) is built/tested against langchain 0.3.x and is
> incompatible with langchain 1.x (which removed `langchain_community.chat_models.vertexai`
> that RAGAS imports). The app only uses stable langchain APIs, so 0.3.x is the
> safe, enterprise-realistic choice. Do NOT bump to 1.x without re-validating RAGAS.

**Primary endpoint:** `POST /api/v1/ask` → returns answer + citations + token/latency stats.

---

## 2. Section Status (16-section plan)

| # | Section | Status | Notes |
|---|---------|--------|-------|
| 1 | Document Ingestion (PDF, DOCX) | ✅ Done | PyMuPDF + docx loaders, dispatcher |
| 2 | Chunking + Metadata | ✅ Done | Hybrid: Semantic + RecursiveCharacterTextSplitter |
| 3 | Embedding Layer | ✅ Done | OpenAI default, Bedrock pluggable |
| 4 | Vector Store (FAISS) | ✅ Done | Metadata filtering + disk persistence |
| 5 | Retrieval Strategy | ✅ Done | Dense + BM25 hybrid, RRF (k=60), MMR |
| 6 | Reranking (Cohere v3.5) | ✅ Done | RERANK_FETCH_K=50, top_k=5 |
| 7 | Context Assembly | ✅ Done | Token budget (tiktoken) + `[N]` citations. **19 tests pass.** |
| 8 | Generation Layer (LLM) | ✅ Done | `/api/v1/ask` live & tested |
| 9 | Grounding & Hallucination Control | ✅ Done | 3-tier defense: grounding gate + claim-level faithfulness judge + citation enforcement. **14 tests.** |
| 10 | Evaluation Framework (RAGAS) | ✅ Done | Real RAGAS 0.2.15 on stable langchain 0.3.x. **Stratified 70-Q gold set** across 6 categories + routing harness (RAGAS for answerable, behavioral checks for out-of-scope/adversarial/ambiguous). `scripts/eval.py` with `--max`/`--category` flags, CI gate |
| 11 | Security & Access Control | 🔴 Not started | ACL, PII redaction, prompt-injection defense |
| 12 | Observability | ✅ Done | structlog + **OpenTelemetry tracing**: per-request `rag.ask` span (cost-per-query, tokens, faithfulness, trust, latency). Vendor-neutral OTLP — **verified live in Datadog APM** (app → OTel → Datadog Agent → cloud). Flips to console/Grafana/Jaeger via one env var |
| 13 | Caching & Performance | 🔴 Not started | Semantic query cache, embedding cache |
| 14 | Feedback Loop | 🔴 Not started | `/api/v1/feedback` endpoint |
| 15 | Infrastructure & Deployment | 🟡 ~10% | Folder skeleton only. Need: Docker, CI, .env.example |
| 16 | Agentic Capabilities (optional) | 🔴 Not started | Web-search fallback, multi-step reasoning |

---

## 3. Key Constants (decided)

```
RRF_K               = 60
RERANK_FETCH_K      = 50
RERANK_TOP_K        = 5
GROUNDING_THRESHOLD = 0.05   # lowered from 0.20 — rerank-v3.5 relevant scores land 0.05–0.40
LLM_MODEL           = gpt-4o-mini
MAX_CONTEXT_TOKENS  = 6000
MAX_COMPLETION_TOKENS = 1024
LLM_TEMPERATURE     = 0.0
FAITHFULNESS_CHECK_ENABLED = true
FAITHFULNESS_THRESHOLD     = 0.85   # >=0.85 baseline for regulated envs, >0.9 target

# Section 10 eval CI-gate thresholds (scripts/eval.py)
EVAL faithfulness      >= 0.85
EVAL answer_relevancy  >= 0.85
EVAL context_precision >= 0.80
EVAL context_recall    =  report-only (corpus-dependent)
```

---

## 4. Files Built This Session (Section 7 & 8)

**Section 7 (committed):**
- `app/generation/__init__.py`
- `app/generation/context_assembler.py`
- `tests/generation/test_context_assembler.py` (19 tests)

**Section 8 (committed):**
- `app/generation/llm.py` — provider-swappable LLM wrapper, retry/backoff, typed errors, token tracking
- `app/services/ask_service.py` — pipeline orchestrator (retrieve → ground → assemble → generate)
- `app/api/ask_api.py` — `POST /api/v1/ask` endpoint, HTTP error mapping
- `app/config/settings.py` — LLM + grounding config
- `app/main.py` — registered ask_router

**Section 9 (committed):**
- `app/generation/grounding.py` — `enforce_citations()` (programmatic) + `check_faithfulness()` (claim-level LLM-as-judge)
- `app/generation/llm.py` — added `run_judge()` provider-agnostic helper (modified)
- `app/services/ask_service.py` — wired Stage 5 verification + trustworthy verdict (modified)
- `app/api/ask_api.py` — added faithfulness/trust response fields (modified)
- `app/config/settings.py` — FAITHFULNESS_* config (modified)
- `tests/generation/test_grounding.py` — 14 deterministic tests

**Section 10:**
- `app/evaluation/ragas_eval.py` — RAGAS wrapper: builds EvaluationDataset, runs
  4 metrics via our own LLM/embeddings (LangchainLLMWrapper)
- `app/evaluation/behavioral.py` — behavioral checks (abstain / resist / no-hallucination)
- `data/eval/wiser_eval_set.json` — **70 stratified gold questions** across 6 categories
  (30 happy_path, 10 edge, 7 multi_hop, 8 out_of_scope, 8 adversarial, 7 ambiguous)
- `scripts/eval.py` — routing harness + CI gate; `--max N` (per-category cap) and
  `--category X` flags for quick runs
- `app/services/ask_service.py` — added `retrieved_contexts` to AskResult (modified)
- `requirements.txt` — pinned langchain 0.3.x stack + ragas (modified)

Category eval-type routing:
- happy_path / edge / multi_hop  -> RAGAS (needs reference answer)
- out_of_scope -> behavioral: must abstain
- adversarial  -> behavioral: must resist (canary string absent from answer)
- ambiguous    -> behavioral: must not hallucinate (abstain or stay faithful)

**Section 12 (Step 1):**
- `app/observability/tracing.py` — OTel TracerProvider setup, console/otlp exporter
  switch via OTEL_EXPORTER, no-op when OTEL_ENABLED=false
- `app/observability/cost.py` — token→USD cost via MODEL_PRICING_PER_1M
- `app/services/ask_service.py` — `ask()` wraps `_ask_impl()` in a `rag.ask` span (modified)
- `app/config/settings.py` — OTEL_* + MODEL_PRICING_PER_1M config (modified)
- `requirements.txt` — opentelemetry-api/sdk (modified)

---

## 5. Open Threads / Known Issues

- **"List all" coverage gap:** Exhaustive queries (e.g. "list all procedure codes")
  return incomplete results. Root cause: vector top-k retrieval under-fetches dense
  tables. The WISeR doc has ~395 codes in a table (pages 36–56); standard RAG returns
  only the ~5 discussed in prose. **Decision pending:** note as known limitation vs.
  build a structured/query-router retrieval path.
- **Retrieval latency:** ~3–10s per query (cold BM25 + Cohere API). Candidate for
  Section 13 (caching).
- **`.env` contains live API keys** — confirm it's gitignored, never commit it.

---

## 6. Next Step

> **Section order was re-prioritized:** 12 Observability ✅ → **13 Caching (next)** →
> 14 Feedback → 15 Deployment → 11 Security (deferred — "owned by a dedicated
> security engineer") → 16 Agentic (optional).

**Section 13 — Caching & Performance.** Build (simple but real patterns):
1. Semantic query cache — embed the query; if cosine sim > ~0.97 to a cached
   query, return the cached answer (skip retrieval+LLM)
2. Embedding cache — hash chunk content → reuse embeddings across re-ingestions
3. Redis (preferred) or in-memory dict for dev
4. Cache TTL: queries ~24h, embeddings ~forever
5. Emit cache-hit/miss as OTel attributes (feeds Section 12 dashboards)

### Datadog (Section 12) — how to run it locally
- Datadog Agent installed on Windows; OTLP receiver enabled in `datadog.yaml`
  (ports 4317/4318); valid API key in the AGENT (not in `.env`).
- App sends when `OTEL_EXPORTER=otlp` (now set in `.env`). Traces appear in
  Datadog **APM → Traces** as service `rag-production-app`, operation `rag.ask`.
- Optional future enhancement: child spans per stage for a waterfall trace.

After Section 13: 14 Feedback → 15 Deployment → 11 Security → 16 Agentic.

---

## 7. Update Log

| Date | Version | Change |
|------|---------|--------|
| 2026-06-22 | v0.13 | Section 12 COMPLETE — OTLP export verified live in Datadog APM (app → OTel → Datadog Agent → cloud); 4 `rag.ask` spans confirmed in UI. requirements: uncommented opentelemetry-exporter-otlp; .env: OTEL_EXPORTER=otlp. (Datadog API key lives in the agent, not .env.) |
| 2026-06-21 | v0.12 | Section 12 Step 1 (Observability) — OpenTelemetry tracing via `app/observability/` (tracing.py + cost.py); `ask()` wrapped in `rag.ask` span with cost-per-query/tokens/faithfulness/latency attrs; console exporter (vendor-neutral, flips to Datadog by one env var); OTEL_ENABLED no-op keeps 48 tests green. Reordered roadmap (Security deferred) |
| 2026-06-21 | v0.11 | Section 10 expanded — stratified 70-Q gold set across 6 categories + routing harness (`app/evaluation/behavioral.py`): RAGAS for answerable, behavioral checks (abstain/resist/no-hallucination) for out-of-scope/adversarial/ambiguous; eval.py `--max`/`--category` flags. Smoke run (1/category) passed |
| 2026-06-21 | v0.10 | Section 10 (Evaluation, RAGAS) — pinned langchain to stable 0.3.x to enable RAGAS 0.2.15; 4 metrics + gold set (8 Q incl. exhaustive-recall demo) + `scripts/eval.py` CI gate; added `retrieved_contexts` to AskResult. Exhaustive "list all codes" Q demonstrates context_recall drop (1.0→0.875) |
| 2026-06-19 | v0.9 | Section 9 (Grounding & Hallucination Control) — 3-tier defense: grounding gate + claim-level faithfulness judge + citation enforcement; +14 tests (33 total); new `trustworthy`/faithfulness fields on /ask |
| 2026-06-19 | v0.8 | Section 8 (Generation Layer) built & live-tested; grounding threshold tuned to 0.05; PROGRESS.md created |
| (earlier) | v0.7 | Section 7 (Context Assembler) + 19 tests committed |
| (earlier) | v0.6 | Sections 1–6 complete (ingestion → retrieval → rerank) |

<!-- When updating: bump version, change "Last updated" at top, add a row here. -->
