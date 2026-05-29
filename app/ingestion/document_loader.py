"""
Unified Document Loader Dispatcher

Responsibilities:
-----------------
- Inspect file extension
- Dispatch to the correct format-specific loader
- Return LangChain Document objects regardless of format

Why a dispatcher exists:
------------------------
The service layer should NOT know whether a file is a PDF,
DOCX, HTML, etc. The dispatcher gives the rest of the system
ONE function to call: `load_document(path)`. Adding a new
format becomes a one-line change here, not a refactor of
the service / API layers.

Current formats:
----------------
- PDF   : full implementation -> app/ingestion/pdf_loader.py
- DOCX  : full implementation -> app/ingestion/docx_loader.py
- HTML  : sketch in comments below (BSHTMLLoader / Unstructured)
- MD    : sketch in comments below (UnstructuredMarkdownLoader)
- TXT   : sketch in comments below (TextLoader)
"""

from pathlib import Path

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx

from app.utils.logger import logger


# =========================================================
# Supported Extensions
# =========================================================
# Centralized so the API layer can validate uploads up front.

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def load_document(file_path: str):
    """
    Detect file type and dispatch to the matching loader.

    Args:
        file_path (str): Full path to the file on disk.

    Returns:
        Tuple[List[Document], str]:
            (documents, doc_type)
            doc_type is one of: "pdf", "docx", (later: "html", "md", "txt")

    Raises:
        ValueError if the extension is unsupported.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    logger.info(
        "document_loader_dispatch",
        file_name=path.name,
        detected_extension=extension
    )

    # ----------------------------------------------------
    # PDF -- primary supported format
    # ----------------------------------------------------
    if extension == ".pdf":
        documents = load_pdf(str(path))
        return documents, "pdf"

    # ----------------------------------------------------
    # DOCX -- Microsoft Word
    # ----------------------------------------------------
    if extension == ".docx":
        documents = load_docx(str(path))
        return documents, "docx"

    # ----------------------------------------------------
    # HTML -- sketch / not enabled yet
    # ----------------------------------------------------
    # from langchain_community.document_loaders import BSHTMLLoader
    #
    # if extension in (".html", ".htm"):
    #     loader = BSHTMLLoader(str(path))   # bs4-based parser
    #     documents = loader.load()
    #     return documents, "html"
    #
    # For URLs (not local files), use:
    # from langchain_community.document_loaders import WebBaseLoader
    # loader = WebBaseLoader(["https://example.com/page"])
    # documents = loader.load()

    # ----------------------------------------------------
    # MARKDOWN -- sketch / not enabled yet
    # ----------------------------------------------------
    # from langchain_community.document_loaders import UnstructuredMarkdownLoader
    #
    # if extension in (".md", ".markdown"):
    #     # mode="elements" keeps headings as Document boundaries,
    #     # mode="single"   returns one Document for the whole file.
    #     loader = UnstructuredMarkdownLoader(str(path), mode="elements")
    #     documents = loader.load()
    #     return documents, "md"

    # ----------------------------------------------------
    # PLAIN TEXT -- sketch / not enabled yet
    # ----------------------------------------------------
    # from langchain_community.document_loaders import TextLoader
    #
    # if extension == ".txt":
    #     loader = TextLoader(str(path), encoding="utf-8")
    #     documents = loader.load()
    #     return documents, "txt"

    # ----------------------------------------------------
    # CSV -- sketch / not enabled yet
    # ----------------------------------------------------
    # from langchain_community.document_loaders import CSVLoader
    #
    # if extension == ".csv":
    #     # CSVLoader produces one Document per row, with each row's
    #     # columns serialized as "col1: val1\ncol2: val2\n...".
    #     loader = CSVLoader(file_path=str(path))
    #     documents = loader.load()
    #     return documents, "csv"

    # ----------------------------------------------------
    # Unsupported -> explicit failure
    # ----------------------------------------------------
    logger.error(
        "unsupported_file_type",
        file_name=path.name,
        extension=extension,
        supported_extensions=list(SUPPORTED_EXTENSIONS)
    )

    raise ValueError(
        f"Unsupported file type: {extension}. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )
