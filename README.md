# Production RAG LangChain — v2

Enterprise-grade RAG backend with **hybrid chunking**, **AWS Bedrock embeddings**, **persistent FAISS with metadata filtering**, and **multi-format ingestion**.

This is `v2` of the prototype. It covers Sections 1–4 of the production-RAG checklist:

| # | Area | Status |
|---|------|--------|
| 1 | Document ingestion (PDF + DOCX; HTML/MD/TXT/CSV sketched) | ✅ |
| 2 | Hybrid chunking (Semantic + Recursive) + rich metadata | ✅ |
| 3 | Embedding model — AWS Bedrock (Titan v2) | ✅ |
| 4 | Vector DB — FAISS w/ metadata filtering (OpenSearch sketch) | ✅ |

---

## Architecture

```
Client / UI
    |
    v
FastAPI Routes              app/api/ingestion_api.py
                            app/api/retrieval_api.py
    |
    v
Service Layer               app/services/ingestion_service.py
                            app/services/retrieval_service.py
    |
    v
Pipeline Components
    |
    +-- Ingestion           app/ingestion/document_loader.py   (format dispatcher)
    |                       app/ingestion/pdf_loader.py
    |                       app/ingestion/docx_loader.py
    |                       app/ingestion/text_splitter.py     (Hybrid: Semantic + Recursive)
    |
    +-- Embeddings          app/embeddings/embedding_generator.py
    |                       app/utils/bedrock_client.py        (AWS Bedrock — default)
    |                       app/utils/openai_client.py         (legacy fallback)
    |
    +-- Vector Store        app/vectorstores/faiss_store.py    (active)
                            app/vectorstores/opensearch_store.py (migration sketch)
    |
    v
Persistent storage          vector_db/faiss_index/rag_index.{faiss,pkl}
```

---

## Setup

```bash
git clone <repo>
cd RAG_v2

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

# Configure env (Bedrock by default — see .env)
cp .env .env.local   # then edit
```

### AWS Bedrock setup

1. Open the AWS Bedrock console in the region you want to use (default `us-east-1`).
2. Go to **Model access** → enable **Amazon Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`).
3. Either:
   - put `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `.env` for local dev, OR
   - attach an IAM role to your EC2/ECS/EKS task with `bedrock:InvokeModel` on the embedding model ARN.

---

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

OpenAPI docs: <http://localhost:8000/docs>

---

## API

### 1. Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest/upload \
     -F "file=@data/documents/wiser-provider-supplier-guide.pdf"
```

Response:
```json
{
  "status": "success",
  "doc_id": "f3c1...",
  "file_name": "wiser-provider-supplier-guide.pdf",
  "doc_type": "pdf",
  "total_documents": 42,
  "total_chunks": 187,
  "total_embeddings": 187,
  "index_total_vectors": 187
}
```

Supported formats: `.pdf`, `.docx`. Add HTML/MD/TXT/CSV by uncommenting the corresponding branches in `app/ingestion/document_loader.py`.

### 2. Semantic search (with optional metadata filter)

```bash
curl -X POST http://localhost:8000/api/v1/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the supplier onboarding process?",
       "top_k": 5,
       "metadata_filter": { "doc_type": "pdf" }
     }'
```

### 3. Index stats

```bash
curl http://localhost:8000/api/v1/index/stats
```

---

## Chunking strategies

Set the strategy via `CHUNK_STRATEGY` in `.env` (or in `app/config/settings.py`):

| Strategy | When to use |
|----------|-------------|
| `recursive` | Fast, deterministic, no embedding cost during chunking. Good for well-structured documents. |
| `semantic` | Topic-aware boundaries via SemanticChunker. Slower, higher quality on prose. |
| `hybrid` *(default)* | SemanticChunker first → RecursiveCharacterTextSplitter for size enforcement. **Recommended for production.** |

The hybrid pipeline:

```
raw text → SemanticChunker (topic boundaries) → RecursiveCharacterTextSplitter (size cap + overlap) → final chunks
```

---

## Metadata on every chunk

Every chunk emitted by `text_splitter.split_documents()` carries:

```json
{
  "doc_id":         "<UUID for the parent document>",
  "chunk_id":       "<UUID for this chunk>",
  "chunk_index":    0,
  "total_chunks":   187,
  "source":         "wiser-provider-supplier-guide.pdf",
  "doc_type":       "pdf",
  "chunk_strategy": "hybrid",
  "chunk_size":     934,
  "ingested_at":    "2025-05-29T15:42:11.234567+00:00",
  "page":           3
}
```

This metadata is what makes **tenant isolation**, **per-source filtering**, and **re-indexing** possible.

---

## Project layout

```
app/
├── api/
│   ├── ingestion_api.py        # POST /ingest/upload  (multi-format)
│   └── retrieval_api.py        # POST /search, GET /index/stats
├── config/
│   └── settings.py             # all knobs (chunking, embedding, FAISS, retrieval)
├── ingestion/
│   ├── pdf_loader.py
│   ├── docx_loader.py
│   ├── document_loader.py      # format dispatcher (HTML/MD/TXT in comments)
│   └── text_splitter.py        # Hybrid chunking + enterprise metadata
├── embeddings/
│   └── embedding_generator.py  # provider-agnostic, batched
├── vectorstores/
│   ├── faiss_store.py          # FAISS persistence + metadata filtering
│   └── opensearch_store.py     # migration sketch (commented)
├── services/
│   ├── ingestion_service.py    # orchestrates the ingestion pipeline
│   └── retrieval_service.py    # orchestrates search
├── utils/
│   ├── logger.py               # structlog config
│   ├── bedrock_client.py       # AWS Bedrock provider
│   └── openai_client.py        # legacy provider
└── main.py
data/
vector_db/
    └── faiss_index/            # persisted index lives here
```

---

## Roadmap (Sections 5–16 of the production-RAG checklist)

- 5  Hybrid retrieval (BM25 + dense + RRF)
- 6  Reranker (Cohere Rerank / BGE / Cross-encoder)
- 7  Context assembly + lost-in-the-middle mitigation
- 8  Generation layer (Bedrock Claude / GPT-4 routing)
- 9  Grounding + citation-aware answers
- 10 Eval framework (RAGAS / golden set)
- 11 Access control + PII redaction
- 12 Observability (LangSmith / Langfuse traces)
- 13 Semantic query cache
- 14 Feedback loop (thumbs / retraining)
- 15 IaC + horizontal scale
- 16 Agentic capabilities
