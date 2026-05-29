"""
Ingestion Service Layer

Responsibilities:
-----------------
- Receive uploaded file from API
- Dispatch to format-specific loader (PDF/DOCX/...)
- Run hybrid chunking with enterprise metadata
- Generate embeddings via Bedrock
- Upsert chunks into the persistent FAISS index
- Return a structured ingestion summary

Enterprise Principle:
---------------------
Service layer orchestrates workflows.
APIs stay thin; business logic stays here.

Pipeline:
---------
    Upload (bytes)
        |
        v
    Temp file write
        |
        v
    document_loader.load_document()      <-- format dispatcher
        |
        v
    text_splitter.split_documents()      <-- hybrid chunking + metadata
        |
        v
    embedding_generator.generate_embeddings()   (sanity log; FAISS re-embeds internally)
        |
        v
    faiss_store.upsert_chunks()          <-- persistent vector store
        |
        v
    JSON response

Notes on legacy disk-based flow:
--------------------------------
The original prototype kept save_uploaded_file() and a
process_uploaded_pdf(file_path) helper. We keep both for
backward compatibility but the API route uses the in-memory
flow which is faster and avoids leaving files in data/documents.
"""

from pathlib import Path
import shutil
import tempfile
import time
import uuid

from app.ingestion.document_loader import load_document
from app.ingestion.text_splitter import split_documents
from app.embeddings.embedding_generator import (
    generate_embeddings,
    get_embedding_model,
)
from app.vectorstores.faiss_store import upsert_chunks

from app.utils.logger import logger


# ============================================================================
# Configuration
# ============================================================================

DOCUMENTS_DIR = "data/documents"


# ============================================================================
# File Persistence Layer (legacy / optional)
# ============================================================================

def save_uploaded_file(upload_file):
    """
    Save uploaded file from memory to disk under data/documents/.
    Kept for cases where you DO want to persist the raw upload
    (e.g. audit, reprocessing). The API route currently does NOT
    call this -- it uses the in-memory flow.
    """

    upload_path = Path(DOCUMENTS_DIR) / upload_file.filename

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    logger.info(
        "file_uploaded_successfully",
        file_name=upload_file.filename
    )

    return str(upload_path)


# ============================================================================
# In-Memory Upload Processing Pipeline (PRODUCTION FLOW)
# ============================================================================

async def process_uploaded_file_in_memory(upload_file):
    """
    Full ingestion pipeline for one uploaded file.

    Steps:
    ------
    1. Read upload bytes
    2. Spool to a NamedTemporaryFile (loaders need a path)
    3. Dispatch to the right loader (PDF / DOCX / ...)
    4. Hybrid chunk + inject metadata (doc_id, source, ...)
    5. Generate embeddings (logged; FAISS re-embeds for index)
    6. Upsert into persistent FAISS index
    7. Delete temp file
    8. Return structured summary

    Returns:
        dict with status, doc_id, counts and per-stage timings.
    """

    logger.info(
        "ingestion_in_memory_started",
        file_name=upload_file.filename,
        content_type=getattr(upload_file, "content_type", None),
    )

    overall_start = time.perf_counter()

    # Generate a single doc_id up front; same id is stamped on
    # every chunk's metadata and returned in the response so the
    # caller can later reference / delete / re-ingest the document.
    doc_id = str(uuid.uuid4())

    tmp_path = None

    try:

        # =====================================================================
        # Step 1: Read uploaded file bytes
        # =====================================================================

        read_start = time.perf_counter()

        file_bytes = await upload_file.read()

        read_time = time.perf_counter() - read_start

        logger.info(
            "file_bytes_read_from_memory",
            file_name=upload_file.filename,
            size_bytes=len(file_bytes),
            read_duration_seconds=round(read_time, 4),
        )

        # =====================================================================
        # Step 2: Spool to a temp file (loaders need an on-disk path)
        # =====================================================================

        # Preserve the original extension so the dispatcher can
        # detect format from the suffix.
        suffix = Path(upload_file.filename).suffix.lower() or ".bin"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = Path(tmp.name)

        try:
            tmp.write(file_bytes)
            tmp.flush()
        finally:
            tmp.close()

        logger.info(
            "temp_file_written",
            temp_path=str(tmp_path),
            temp_file_size_bytes=len(file_bytes),
        )

        # =====================================================================
        # Step 3: Format-aware document loading
        # =====================================================================

        load_start = time.perf_counter()

        documents, doc_type = load_document(str(tmp_path))

        load_time = time.perf_counter() - load_start

        logger.info(
            "document_loaded",
            file_name=upload_file.filename,
            doc_type=doc_type,
            total_documents=len(documents),
            load_duration_seconds=round(load_time, 4),
        )

        # =====================================================================
        # Step 4: Hybrid chunking with rich metadata
        # =====================================================================
        # SemanticChunker requires the same embedding model the
        # rest of the pipeline uses. We resolve it ONCE here and
        # reuse it for embedding generation below.
        # =====================================================================

        embedding_model = get_embedding_model()

        chunk_start = time.perf_counter()

        chunks = split_documents(
            documents=documents,
            source_filename=upload_file.filename,
            doc_type=doc_type,
            doc_id=doc_id,
            embedding_model=embedding_model,
            # strategy resolved from settings.CHUNK_STRATEGY
        )

        chunk_time = time.perf_counter() - chunk_start

        logger.info(
            "documents_chunked",
            file_name=upload_file.filename,
            doc_id=doc_id,
            total_chunks=len(chunks),
            chunk_duration_seconds=round(chunk_time, 4),
        )

        # =====================================================================
        # Step 5: Generate embeddings (observability / sanity log)
        # =====================================================================
        # NOTE: FAISS.from_documents() inside upsert_chunks() will
        # call embed_documents() again internally. The explicit
        # call here is a deliberate observability hook so that
        # log readers can see embedding dimensions, batch progress,
        # and provider details independently of the vector store.
        # In a cost-sensitive deployment you can remove this call
        # and let FAISS handle embedding inline.
        # =====================================================================

        embed_start = time.perf_counter()

        embeddings = generate_embeddings(chunks)

        embed_time = time.perf_counter() - embed_start

        logger.info(
            "embeddings_generated",
            file_name=upload_file.filename,
            doc_id=doc_id,
            total_embeddings=len(embeddings),
            embedding_dimension=len(embeddings[0]) if embeddings else 0,
            embed_duration_seconds=round(embed_time, 4),
        )

        # =====================================================================
        # Step 6: Upsert into persistent FAISS index
        # =====================================================================

        store_start = time.perf_counter()

        store_result = upsert_chunks(chunks)

        store_time = time.perf_counter() - store_start

        logger.info(
            "chunks_stored_in_vector_db",
            file_name=upload_file.filename,
            doc_id=doc_id,
            store_result=store_result,
            store_duration_seconds=round(store_time, 4),
        )

        # =====================================================================
        # Step 7: Cleanup temp file
        # =====================================================================

        cleanup_start = time.perf_counter()

        try:
            tmp_path.unlink()
            cleanup_time = time.perf_counter() - cleanup_start
            logger.info(
                "temp_file_deleted",
                temp_path=str(tmp_path),
                cleanup_duration_seconds=round(cleanup_time, 4),
            )
        except Exception as ex:
            cleanup_time = time.perf_counter() - cleanup_start
            logger.warning(
                "temp_file_delete_failed",
                temp_path=str(tmp_path),
                error=str(ex),
                cleanup_duration_seconds=round(cleanup_time, 4),
            )

        # =====================================================================
        # Final summary log
        # =====================================================================

        total_time = time.perf_counter() - overall_start

        logger.info(
            "ingestion_in_memory_completed",
            file_name=upload_file.filename,
            doc_id=doc_id,
            doc_type=doc_type,
            total_documents=len(documents),
            total_chunks=len(chunks),
            total_embeddings=len(embeddings),
            index_total_vectors=store_result.get("index_total_vectors"),
            total_duration_seconds=round(total_time, 4),
            read_duration_seconds=round(read_time, 4),
            load_duration_seconds=round(load_time, 4),
            chunk_duration_seconds=round(chunk_time, 4),
            embed_duration_seconds=round(embed_time, 4),
            store_duration_seconds=round(store_time, 4),
            cleanup_duration_seconds=round(cleanup_time, 4),
        )

        return {
            "status": "success",
            "doc_id": doc_id,
            "file_name": upload_file.filename,
            "doc_type": doc_type,
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "total_embeddings": len(embeddings),
            "index_total_vectors": store_result.get("index_total_vectors"),
        }

    except Exception as ex:

        logger.error(
            "ingestion_in_memory_failed",
            file_name=upload_file.filename,
            error=str(ex),
        )

        # Best-effort temp file cleanup on failure path.
        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

        raise


# ============================================================================
# Legacy Disk-Based Processing Flow
# ============================================================================

def process_uploaded_pdf(file_path: str):
    """
    Synchronous disk-based ingestion pipeline.

    Kept for backward compatibility with the original prototype
    so that any debug scripts that called this entrypoint keep
    working. The in-memory async flow above is the production path.
    """

    logger.info(
        "ingestion_disk_pipeline_started",
        file_path=file_path,
    )

    doc_id = str(uuid.uuid4())

    documents, doc_type = load_document(file_path)

    embedding_model = get_embedding_model()

    chunks = split_documents(
        documents=documents,
        source_filename=Path(file_path).name,
        doc_type=doc_type,
        doc_id=doc_id,
        embedding_model=embedding_model,
    )

    embeddings = generate_embeddings(chunks)

    store_result = upsert_chunks(chunks)

    logger.info(
        "ingestion_disk_pipeline_completed",
        doc_id=doc_id,
        total_documents=len(documents),
        total_chunks=len(chunks),
        total_embeddings=len(embeddings),
        index_total_vectors=store_result.get("index_total_vectors"),
    )

    return {
        "status": "success",
        "doc_id": doc_id,
        "doc_type": doc_type,
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "total_embeddings": len(embeddings),
        "index_total_vectors": store_result.get("index_total_vectors"),
    }
