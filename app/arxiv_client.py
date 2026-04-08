"""ArXiv API client for fetching paper metadata and PDFs."""

import arxiv
import os
import re
import hashlib
from datetime import datetime, timezone
from typing import Optional, List
from pathlib import Path


class ArxivClient:
    """Client for interacting with ArXiv API."""

    def __init__(self, download_dir: str = "data/papers"):
        self.client = arxiv.Client()
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def search(
        self,
        query: str = "",
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
    ) -> List[arxiv.Result]:
        """Search for papers on ArXiv, optionally filtered by category and date."""
        # ArXiv query syntax uses field prefixes:
        #   ti/abs for text search, cat: for category, submittedDate for date range
        parts = []

        if query:
            # Search in title and abstract
            parts.append(f"(ti:{query} OR abs:{query})")

        if categories:
            parts.append("(" + " OR ".join(f"cat:{c}" for c in categories) + ")")

        if date_from:
            date_str = date_from.strftime("%Y%m%d")
            parts.append(f"submittedDate:[{date_str}000000 TO 99991231235959]")

        if not parts:
            # Nothing specified — refuse to return everything
            return []

        combined_query = " AND ".join(parts)
        search = arxiv.Search(
            query=combined_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        return list(self.client.results(search))

    def get_pdf_path(self, result: arxiv.Result) -> Path:
        """Get or download the PDF for a paper result."""
        # Create a safe filename from the paper ID
        pdf_hash = hashlib.md5(result.pdf_url.encode()).hexdigest()[:8]
        safe_title = re.sub(r"[^\w\-]", "_", result.title)[:50]
        filename = f"{safe_title}_{pdf_hash}.pdf"
        pdf_path = self.download_dir / filename

        if not pdf_path.exists():
            print(f"Downloading: {result.title}")
            result.download_pdf(dirpath=self.download_dir, filename=filename)

        return pdf_path

    def get_pdf_path_by_id(self, arxiv_id: str):
        """Find an already-downloaded PDF by arxiv ID hash suffix, or return None."""
        # The hash is based on pdf_url which we don't have here, so scan by arxiv_id prefix
        clean_id = arxiv_id.split("v")[0]  # strip version
        for pdf in self.download_dir.glob("*.pdf"):
            if clean_id.replace(".", "_") in pdf.stem or clean_id in pdf.stem:
                return pdf
        return None

    def get_paper_metadata(self, result: arxiv.Result) -> dict:
        """Extract useful metadata from a paper result."""
        return {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": result.published.isoformat(),
            "pdf_url": result.pdf_url,
            "arxiv_id": result.entry_id.split("/")[-1],
            "categories": result.categories,
            "comment": result.comment,
        }
