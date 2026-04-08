"""CLI script for scheduled daily ArXiv fetch.

Usage:
    python -m app.fetch_job --query "neutron star" --categories astro-ph.HE --max-results 10
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.arxiv_client import ArxivClient
from app.pdf_extractor import PDFExtractor
from app.summarizer import PaperSummarizer
from app.vector_db import PaperVectorDB
from app import paper_store


def run(query: str, categories: list[str], max_results: int, days_back: int = 1):
    load_dotenv()

    date_from = datetime.now(timezone.utc) - timedelta(days=days_back)

    client = ArxivClient()
    extractor = PDFExtractor()
    summarizer = PaperSummarizer()
    vdb = PaperVectorDB()

    print(f"[{datetime.now()}] Fetching papers: query={repr(query)} cats={categories} since={date_from.date()}")

    papers = client.search(
        query=query,
        max_results=max_results,
        categories=categories if categories else None,
        date_from=date_from,
    )

    print(f"Found {len(papers)} papers.")
    new_count = 0

    for result in papers:
        metadata = client.get_paper_metadata(result)
        pid = metadata["arxiv_id"]

        if paper_store.paper_exists(pid):
            print(f"  [skip] {pid} already stored.")
            continue

        print(f"  [new]  {pid}: {metadata['title'][:70]}")
        pdf_path = client.get_pdf_path(result)
        text = extractor.extract_first_n_pages(pdf_path, n_pages=3)
        summary = summarizer.summarize(text)

        paper_store.save_paper(pid, metadata, summary)
        chroma_meta = {
            k: v for k, v in {
                **metadata,
                "summary": summary,
                "authors": ", ".join(metadata.get("authors", [])),
                "categories": ", ".join(metadata.get("categories", [])),
            }.items()
            if v is not None and not isinstance(v, list)
        }
        vdb.add_paper(
            paper_id=pid,
            title=metadata["title"],
            summary=summary,
            metadata=chroma_meta,
        )
        new_count += 1

    print(f"[{datetime.now()}] Done. Added {new_count} new papers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily ArXiv fetch job")
    parser.add_argument("--query", default="", help="Search query")
    parser.add_argument("--categories", nargs="*", default=[], help="ArXiv categories")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--days-back", type=int, default=1, help="Fetch papers from last N days")
    args = parser.parse_args()

    run(
        query=args.query,
        categories=args.categories,
        max_results=args.max_results,
        days_back=args.days_back,
    )
