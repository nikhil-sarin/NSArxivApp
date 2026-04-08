"""ArXiv API client for fetching paper metadata and PDFs."""

import arxiv
import os
import hashlib
from typing import Optional, List
from pathlib import Path


class ArxivClient:
    """Client for interacting with ArXiv API."""

    def __init__(self, download_dir: str = "papers"):
        self.client = arxiv.Client()
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def search(
        self,
        query: str = "",
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        sort_by: str = "submittedDate",
    ) -> List[arxiv.Result]:
        """Search for papers on ArXiv."""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        if categories:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                entry_query=" OR ".join([f"cat:{cat}" for cat in categories]),
            )
        return list(self.client.results(search))

    def get_pdf_path(self, result: arxiv.Result) -> Path:
        """Get or download the PDF for a paper result."""
        # Create a safe filename from the paper ID
        pdf_hash = hashlib.md5(result.pdf_url.encode()).hexdigest()[:8]
        filename = f"{result.title.replace(' ', '_')[:50]}_{pdf_hash}.pdf"
        pdf_path = self.download_dir / filename

        if not pdf_path.exists():
            print(f"Downloading: {result.title}")
            result.download_pdf(dirpath=self.download_dir, filename=filename)

        return pdf_path

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
