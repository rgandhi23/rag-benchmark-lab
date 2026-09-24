from io import BytesIO

import pytest
from pypdf import PdfWriter

from ragbench.ingestion import PDFIngestionError, extract_pdf_documents


def test_rejects_non_pdf_content() -> None:
    with pytest.raises(PDFIngestionError, match="not a valid PDF"):
        extract_pdf_documents("notes.pdf", b"plain text")


def test_rejects_image_only_or_empty_pdf() -> None:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(output)

    with pytest.raises(PDFIngestionError, match="No extractable text"):
        extract_pdf_documents("scan.pdf", output.getvalue())
