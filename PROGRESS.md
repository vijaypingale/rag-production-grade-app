# RAG Production-Grade App — Progress & Handoff

> **Version:** v0.8 &nbsp;|&nbsp; **Last updated:** 2026-06-19
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
Rerank v3.5 · LangChain · structlog · pytest.

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
| 8 | Generation Layer (LLM) | ✅ Done | `/api/v1/ask` live & tested. **NOT yet committed.** |
| 9 | Grounding & Hallucination Control | 🟡 Partial | Grounding **gate** live (threshold). Need: faithfulness judge + citation enforcement |
| 10 | Evaluation Framework (RAGAS) | 🔴 Not started | |
| 11 | Security & Access Control | 🔴 Not started | ACL, PII redaction, prompt-injection defense |
| 12 | Observability | 🟡 ~40% | structlog only. Need: Langfuse/LangSmith, cost, dashboards |
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
```

---

## 4. Files Built This Session (Section 7 & 8)

**Section 7 (committed):**
- `app/generation/__init__.py`
- `app/generation/context_assembler.py`
- `tests/generation/test_context_assembler.py` (19 tests)

**Section 8 (⚠️ NOT committed — in working tree):**
- `app/generation/llm.py` — provider-swappable LLM wrapper, retry/backoff, typed errors, token tracking
- `app/services/ask_service.py` — pipeline orchestrator (retrieve → ground → assemble → generate)
- `app/api/ask_api.py` — `POST /api/v1/ask` endpoint, HTTP error mapping
- `app/config/settings.py` — added LLM + grounding config (modified)
- `app/main.py` — registered ask_router (modified)

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

**Section 9 — complete Grounding & Hallucination Control.** Builds on the grounding
gate already in `ask_service.py`. Add:
1. Faithfulness check (LLM-as-judge: does the answer match cited chunks?)
2. Citation enforcement (parse `[N]` tags, verify each maps to a retrieved chunk)
3. Track "unanswerable" rate as a metric

---

## 7. Update Log

| Date | Version | Change |
|------|---------|--------|
| 2026-06-19 | v0.8 | Section 8 (Generation Layer) built & live-tested; grounding threshold tuned to 0.05; PROGRESS.md created |
| (earlier) | v0.7 | Section 7 (Context Assembler) + 19 tests committed |
| (earlier) | v0.6 | Sections 1–6 complete (ingestion → retrieval → rerank) |

<!-- When updating: bump version, change "Last updated" at top, add a row here. -->
