"""
Ingestion API Routes

Responsibilities:
-----------------
- Validate uploaded files (extension + presence)
- Delegate processing to the ingestion service
- Translate service exceptions into HTTP responses

The route is intentionally THIN.
All business logic lives in app/services/ingestion_service.py.
"""

from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.document_loader import SUPPORTED_EXTENSIONS
from app.services.ingestion_service import (
    process_uploaded_pdf,
    process_uploaded_file_in_memory,
)

from app.utils.logger import logger


router = APIRouter(
    tags=["Document Ingestion"]
)


# ============================================================================
# Smoke test
# ============================================================================

@router.get("/ingest/test")
async def test_ingestion_api():
    """Liveness check for the ingestion router."""
    return {
        "message": "Ingestion API is working successfully"
    }


# ============================================================================
# Multi-format upload endpoint
# ============================================================================

@router.post("/ingest/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    """
    Upload a single document for ingestion.

    Accepted formats:
    -----------------
    Currently: PDF, DOCX.
    To add more (HTML, MD, TXT, CSV), uncomment the corresponding
    branches in app/ingestion/document_loader.py.

    Pipeline (handled in the service layer):
    ----------------------------------------
      load -> chunk (hybrid + metadata) -> embed (Bedrock) -> FAISS upsert

    Returns:
        JSON ingestion summary with doc_id, counts, and timings.
    """

    logger.info(
        "upload_api_invoked",
        file_name=file.filename,
        content_type=file.content_type,
    )

    try:

        # =====================================================
        # Validate filename present
        # =====================================================
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file has no filename"
            )

        # =====================================================
        # Validate extension is supported by the dispatcher
        # =====================================================
        extension = Path(file.filename).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:

            logger.warning(
                "upload_rejected_unsupported_format",
                file_name=file.filename,
                extension=extension,
                supported=list(SUPPORTED_EXTENSIONS),
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: {extension}. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                ),
            )

        logger.info(
            "upload_validated",
            file_name=file.filename,
            extension=extension,
        )

        # =====================================================
        # Trigger ingestion pipeline (in-memory production flow)
        # =====================================================
        result = await process_uploaded_file_in_memory(file)

        return result

    except HTTPException:
        # Already a structured HTTP error -- propagate as-is.
        raise

    except Exception as ex:

        logger.error(
            "upload_api_failed",
            file_name=getattr(file, "filename", None),
            error=str(ex),
        )

        raise HTTPException(
            status_code=500,
            detail=str(ex),
        )
