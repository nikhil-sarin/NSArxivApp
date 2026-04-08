"""PDF text extraction module."""

from pathlib import Path
from pypdf import PdfReader
from typing import Optional


class PDFExtractor:
    """Extract text from PDF files."""

    def __init__(self):
        pass

    def extract_text(self, pdf_path: Path) -> str:
        """Extract all text from a PDF file."""
        try:
            reader = PdfReader(str(pdf_path))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def extract_first_n_pages(self, pdf_path: Path, n_pages: int = 3) -> str:
        """Extract text from the first N pages of a PDF."""
        try:
            reader = PdfReader(str(pdf_path))
            text = ""
            for i, page in enumerate(reader.pages):
                if i >= n_pages:
                    break
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
