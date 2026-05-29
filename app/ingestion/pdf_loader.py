"""
PDF Ingestion Module

Responsibilities:
-----------------
- Load PDF documents
- Extract diagnostic metadata
- Return LangChain Document objects

This module represents the first layer
of the enterprise RAG pipeline for PDF files.

Library notes:
--------------
- PyMuPDF (fitz):  direct access to PDF internals
                   (metadata, image detection, encryption)
- PyMuPDFLoader:   LangChain wrapper that returns Documents
                   ready for chunking
"""

from pathlib import Path
import fitz

from langchain_community.document_loaders import PyMuPDFLoader

from app.config.settings import PDF_PATH
from app.utils.logger import logger


def extract_pdf_metadata(pdf_path: str):
    """
    Extract PDF metadata using PyMuPDF (fitz).

    Diagnostic only -- the values are logged for the
    audit trail. The chunking layer adds its own
    enterprise metadata (doc_id, chunk_id, etc).

    Args:
        pdf_path (str): Full path to PDF file
    """

    pdf_document = fitz.open(pdf_path)

    metadata = pdf_document.metadata

    total_pages = pdf_document.page_count

    has_images = False

    # Scan pages to detect images.
    # Useful as a signal that downstream OCR may be needed
    # for scanned PDFs (current pipeline does NOT OCR).
    for page_num in range(total_pages):

        page = pdf_document.load_page(page_num)

        if page.get_images():
            has_images = True
            break

    logger.info(
        "pdf_metadata_extracted",
        file_name=Path(pdf_path).name,
        total_pages=total_pages,
        contains_images=has_images,
        encrypted=pdf_document.is_encrypted,
        metadata=metadata,
    )

    pdf_document.close()


def load_pdf(pdf_path: str):
    """
    Load PDF and return LangChain Document objects.

    Process:
    --------
    1. Validate file exists
    2. Extract diagnostic metadata using PyMuPDF
    3. Use LangChain's PyMuPDFLoader to produce Documents
       (one Document per page, each carrying loader metadata
        such as "page" and "source")
    4. Log preview + sample metadata

    Returns:
        List[LangChain Document]: one Document per page.
    """

    path = Path(pdf_path)

    if not path.exists():

        logger.error(
            "pdf_file_not_found",
            file_path=pdf_path,
        )

        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(
        "pdf_loading_started",
        file_name=path.name,
    )

    # Step 1: diagnostic metadata
    extract_pdf_metadata(pdf_path)

    # Step 2: standard loader -> Documents
    loader = PyMuPDFLoader(str(path))
    documents = loader.load()

    logger.info(
        "pdf_loaded_successfully",
        total_documents=len(documents),
        sample_metadata=documents[0].metadata if documents else None,
    )

    if documents:
        logger.info(
            "sample_document_preview",
            content=documents[0].page_content[:500],
        )

    return documents


# ============================================================================
# NOTE:
# ----------------------------------------------------------------------------
# Previously this module was executed directly using:
#
#     python -m app.ingestion.pdf_loader
#
# That standalone execution was useful during the initial development
# and debugging phase to independently validate PDF ingestion,
# metadata extraction and logging.
#
# In the enterprise FastAPI architecture, execution flows through:
#
#     Client/UI -> FastAPI Endpoint -> Service Layer -> Ingestion Modules
#
# We intentionally keep the old standalone block commented out below
# for future debugging / local validation.
# ============================================================================

# if __name__ == "__main__":
#     load_pdf(PDF_PATH)
