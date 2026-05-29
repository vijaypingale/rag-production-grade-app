"""
Main FastAPI Application Entry Point

Responsibilities:
-----------------
- Initialize FastAPI application
- Register API routers (ingestion + retrieval)
- Configure application metadata
- Serve as backend entry point for the RAG platform

Enterprise Architecture:
------------------------
Client/UI
    |
    v
FastAPI Routes  (app/api/*)
    |
    v
Service Layer   (app/services/*)
    |
    v
RAG Pipeline Components
    |  ingestion: pdf/docx loader -> text_splitter (hybrid) -> embeddings (Bedrock) -> FAISS
    |  retrieval: FAISS similarity_search (with metadata filtering)
    v
Persistent storage  (vector_db/faiss_index/)
"""

from fastapi import FastAPI

from app.api.ingestion_api import router as ingestion_router
from app.api.retrieval_api import router as retrieval_router

from app.utils.logger import logger


# ============================================================================
# Initialize FastAPI Application
# ============================================================================

app = FastAPI(
    title="Production RAG LangChain API",
    description=(
        "Enterprise-grade RAG backend using LangChain, "
        "FAISS (with metadata filtering), FastAPI, and AWS Bedrock embeddings. "
        "Supports hybrid chunking (Semantic + Recursive) and multi-format ingestion."
    ),
    version="2.0.0",
)


# ============================================================================
# Register API Routers
# ============================================================================
# All routers are mounted under /api/v1 to keep room for future
# versioning (/api/v2 etc) without breaking existing clients.

app.include_router(
    ingestion_router,
    prefix="/api/v1",
)

app.include_router(
    retrieval_router,
    prefix="/api/v1",
)


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health")
async def health_check():

    logger.info("health_check_called")

    return {
        "status": "healthy",
        "application": "production-rag-langchain",
        "version": "2.0.0",
    }


# ============================================================================
# Application Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():

    logger.info("FASTAPI APPLICATION STARTED")


# ============================================================================
# Application Shutdown Event
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event():

    logger.info("fastapi_application_stopped")
