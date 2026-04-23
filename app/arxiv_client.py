"""ArXiv API client for fetching paper metadata and PDFs."""

import arxiv
import os
import re
from datetime import datetime, timezone
from typing import Optional, List
from pathlib import Path


class ArxivClient:
    """Client for interacting with ArXiv API."""

    def __init__(self, download_dir: str = "data/papers"):
        # delay_seconds: wait between requests to avoid 429 rate limiting
        # num_retries: retry on transient failures (arxiv.Client default is 3)
        self.client = arxiv.Client(
            page_size=25,
            delay_seconds=3.0,
            num_retries=5,
        )
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def search(
        self,
        query: str = "",
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        author: str = "",
    ) -> List[arxiv.Result]:
        """Search for papers on ArXiv, optionally filtered by category, date, and author."""
        parts = []

        if query:
            parts.append(f"(ti:{query} OR abs:{query})")

        if author:
            # ArXiv author search: quote multi-word names so they match as a phrase,
            # not as separate tokens (which would match "Nikhil X" and "Y Sarin" separately).
            # Also support lastname_firstinitial format e.g. sarin_n.
            clean_author = author.strip().strip('"').strip("'")
            if " " in clean_author and "_" not in clean_author:
                # Convert "Nikhil Sarin" -> "sarin_n" for most precise ArXiv matching,
                # but also keep the quoted form as a fallback OR clause.
                parts_name = clean_author.split()
                lastname = parts_name[-1].lower()
                firstinit = parts_name[0][0].lower()
                parts.append(f'(au:"{clean_author}" OR au:{lastname}_{firstinit})')
            else:
                parts.append(f'au:"{clean_author}"')

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
        # Embed the arxiv ID in the filename so get_pdf_path_by_id can find it reliably.
        arxiv_id = result.entry_id.split("/")[-1].split("v")[0]  # e.g. 2301.12345
        safe_id = arxiv_id.replace(".", "_")
        safe_title = re.sub(r"[^\w\-]", "_", result.title)[:40]
        filename = f"{safe_id}_{safe_title}.pdf"
        pdf_path = self.download_dir / filename

        if not pdf_path.exists():
            # Also check old hash-based naming for backwards compatibility
            old = self.get_pdf_path_by_id(arxiv_id)
            if old is not None and old.exists():
                return old
            print(f"Downloading: {result.title}")
            result.download_pdf(dirpath=self.download_dir, filename=filename)

        return pdf_path

    def get_pdf_path_by_id(self, arxiv_id: str) -> Optional[Path]:
        """Find an already-downloaded PDF by arxiv ID embedded in the filename."""
        clean_id = arxiv_id.split("v")[0]  # strip version suffix
        safe_id = clean_id.replace(".", "_")
        # New naming: {safe_id}_*.pdf
        matches = list(self.download_dir.glob(f"{safe_id}_*.pdf"))
        if matches:
            return matches[0]
        # Legacy naming: title contained the id tokens
        for pdf in self.download_dir.glob("*.pdf"):
            if safe_id in pdf.stem or clean_id in pdf.stem:
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
