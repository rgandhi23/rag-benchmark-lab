import hashlib
import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ragbench.models import DocumentInput


class PDFIngestionError(ValueError):
    """Raised when a PDF cannot be safely converted into searchable text."""


def _document_id(filename: str, content: bytes) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-") or "document"
    digest = hashlib.sha256(content).hexdigest()[:10]
    return f"{stem}-{digest}"


def extract_pdf_documents(
    filename: str,
    content: bytes,
    *,
    document_id: str | None = None,
    title: str | None = None,
) -> list[DocumentInput]:
    """Extract one searchable document per text-bearing page while retaining page citations."""
    if not content.startswith(b"%PDF-"):
        raise PDFIngestionError("The uploaded file is not a valid PDF")
    try:
        reader = PdfReader(BytesIO(content))
    except (PdfReadError, OSError, ValueError) as exc:
        raise PDFIngestionError("The PDF could not be read") from exc
    if reader.is_encrypted:
        raise PDFIngestionError("Encrypted PDFs are not supported")

    resolved_id = document_id or _document_id(filename, content)
    resolved_title = title or Path(filename).stem or "Uploaded PDF"
    pages: list[DocumentInput] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pages.append(
            DocumentInput(
                document_id=resolved_id,
                title=resolved_title,
                text=text,
                source=filename,
                page_number=page_number,
            )
        )
    if not pages:
        raise PDFIngestionError(
            "No extractable text was found; image-only PDFs require OCR before ingestion"
        )
    return pages
