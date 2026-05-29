"""
DOCX Ingestion Module

Responsibilities:
-----------------
- Load Microsoft Word (.docx) documents
- Extract metadata (title, author, paragraphs)
- Return LangChain Document objects

Why a separate loader per format:
---------------------------------
Each file format has different parsing concerns:
- PDF needs page extraction and image detection
- DOCX needs paragraph + table structure preservation
- HTML needs tag stripping and link handling
Keeping them separate keeps each module small and testable.
The unified dispatcher lives in app/ingestion/document_loader.py.

Library choice:
---------------
We use LangChain's Docx2txtLoader which wraps the docx2txt
package. It is fast, dependency-light and produces a single
Document per file (vs UnstructuredWordDocumentLoader which
splits more aggressively but pulls in heavy deps).
"""

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader

from app.utils.logger import logger


def load_docx(docx_path: str):
    """
    Load a DOCX file and return LangChain Document objects.

    Process:
    --------
    1. Validate file exists
    2. Build Docx2txtLoader for the path
    3. Load -> returns a single Document containing the whole file's text
    4. Log preview + metadata for observability

    Args:
        docx_path (str): Full path to .docx file

    Returns:
        List[LangChain Document]: Typically 1 Document; downstream
                                  chunking will split it into many.
    """

    path = Path(docx_path)

    if not path.exists():

        logger.error(
            "docx_file_not_found",
            file_path=docx_path
        )

        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    logger.info(
        "docx_loading_started",
        file_name=path.name
    )

    # Docx2txtLoader returns a single Document per file.
    # If you need paragraph-level granularity, switch to
    # UnstructuredWordDocumentLoader(path, mode="elements").
    loader = Docx2txtLoader(str(path))

    documents = loader.load()

    logger.info(
        "docx_loaded_successfully",
        total_documents=len(documents),
        sample_metadata=documents[0].metadata if documents else None
    )

    if documents:
        logger.info(
            "sample_document_preview",
            content=documents[0].page_content[:500]
        )

    return documents
