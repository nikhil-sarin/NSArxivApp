"""Helpers for retrieving full paper text for summaries and chat."""

from pathlib import Path
from typing import Optional

from app.arxiv_client import ArxivClient
from app.pdf_extractor import PDFExtractor
from app.tex_extractor import fetch_html_text


def get_paper_text(
    arxiv_id: str,
    arxiv_client: ArxivClient,
    pdf_extractor: PDFExtractor,
    result: Optional[object] = None,
    title: Optional[str] = None,
    pdf_url: Optional[str] = None,
    cache_dir: Path = Path("data/papers"),
) -> str:
    """Return the fullest available text for a paper: HTML first, then full PDF."""
    text = fetch_html_text(arxiv_id, cache_dir)
    if text:
        return text

    pdf_path = arxiv_client.get_pdf_path_by_id(arxiv_id)
    if pdf_path is None or not pdf_path.exists():
        if result is not None:
            pdf_path = arxiv_client.get_pdf_path(result)
        else:
            pdf_path = arxiv_client.get_pdf_path_for_id(arxiv_id, title=title, pdf_url=pdf_url)

    return pdf_extractor.extract_text(pdf_path)
