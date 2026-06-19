"""CLI script for scheduled daily ArXiv fetch.

Usage:
    python -m app.fetch_job --query "neutron star" --categories astro-ph.HE --max-results 10
    python -m app.fetch_job --mode new-submissions --categories astro-ph.HE gr-qc --days-back 1
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.arxiv_announcements import fetch_papers as fetch_announcement_papers
from app.arxiv_announcements import paper_to_metadata
from app.arxiv_client import ArxivClient
from app import paper_store
from app.paper_text import get_paper_text
from app.pdf_extractor import PDFExtractor
from app.summarizer import PaperSummarizer
from app.summary_workflow import summarize_with_fallback
from app.vector_db import PaperVectorDB

MAX_EMPTY_DATE_LOOKBACK_DAYS = 7


def _save_new_papers(
    papers: list[dict],
    *,
    client: ArxivClient,
    extractor: PDFExtractor,
    summarizer: PaperSummarizer,
    vdb: PaperVectorDB,
) -> int:
    new_count = 0
    for metadata in papers:
        pid = metadata["arxiv_id"]
        if paper_store.paper_exists(pid):
            print(f"  [skip] {pid} already stored.")
            continue

        print(f"  [new]  {pid}: {metadata['title'][:70]}")
        try:
            text = get_paper_text(
                pid,
                client,
                extractor,
                title=metadata.get("title"),
                pdf_url=metadata.get("pdf_url"),
            )
        except Exception as exc:
            print(f"  [warn] full text unavailable for {pid}: {exc}")
            text = ""

        summary = summarize_with_fallback(summarizer, text, metadata.get("abstract", ""))
        if not summary.strip():
            print(f"  [skip] {pid} had no readable full text or abstract.")
            continue

        paper_store.save_paper(pid, metadata, summary)
        chroma_meta = {
            key: value for key, value in {
                **metadata,
                "summary": summary,
                "authors": ", ".join(metadata.get("authors", [])),
                "categories": ", ".join(metadata.get("categories", [])),
            }.items()
            if value is not None and not isinstance(value, list)
        }
        vdb.add_paper(
            paper_id=pid,
            title=metadata["title"],
            summary=summary,
            metadata=chroma_meta,
        )
        new_count += 1
    return new_count


def run_query_search(query: str, categories: list[str], max_results: int, days_back: int = 1):
    load_dotenv()

    date_from = datetime.now(timezone.utc) - timedelta(days=days_back)

    client = ArxivClient()
    extractor = PDFExtractor()
    summarizer = PaperSummarizer()
    vdb = PaperVectorDB()

    print(f"[{datetime.now()}] Fetching papers: query={repr(query)} cats={categories} since={date_from.date()}")

    results = client.search(
        query=query,
        max_results=max_results,
        categories=categories if categories else None,
        date_from=date_from,
    )

    print(f"Found {len(results)} papers.")
    new_count = _save_new_papers(
        [client.get_paper_metadata(result) for result in results],
        client=client,
        extractor=extractor,
        summarizer=summarizer,
        vdb=vdb,
    )
    print(f"[{datetime.now()}] Done. Added {new_count} new papers.")


def _fetch_latest_nonempty_submissions(
    categories: list[str],
    start_date,
    *,
    max_lookback_days: int = MAX_EMPTY_DATE_LOOKBACK_DAYS,
) -> tuple[object, dict[str, list]]:
    """Walk backwards from start_date until any category has announcements."""
    for offset in range(max_lookback_days + 1):
        candidate_date = start_date - timedelta(days=offset)
        by_category: dict[str, list] = {}
        total = 0
        for category in categories:
            papers = fetch_announcement_papers(category, candidate_date)
            by_category[category] = papers
            total += len(papers)
        if total > 0:
            return candidate_date, by_category
    return start_date, {category: [] for category in categories}


def run_new_submissions(categories: list[str], max_results: int, days_back: int = 1):
    load_dotenv()
    if not categories:
        raise ValueError("new-submissions mode requires at least one category")

    requested_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).date()
    client = ArxivClient()
    extractor = PDFExtractor()
    summarizer = PaperSummarizer()
    vdb = PaperVectorDB()

    print(f"[{datetime.now()}] Fetching new submissions: cats={categories} requested_announcement_date={requested_date}")

    try:
        target_date, papers_by_category = _fetch_latest_nonempty_submissions(categories, requested_date)
    except Exception as exc:
        raise RuntimeError(f"failed to resolve latest non-empty announcement date: {exc}") from exc

    if target_date != requested_date:
        print(f"  [info] no new submissions on {requested_date}; using latest non-empty announcement date {target_date}")

    seen_ids: set[str] = set()
    metadata_to_save: list[dict] = []
    for category in categories:
        papers = papers_by_category.get(category, [])
        print(f"  [info] {category}: {len(papers)} announced papers")
        for paper in papers:
            if paper.arxiv_id in seen_ids:
                continue
            seen_ids.add(paper.arxiv_id)
            metadata_to_save.append(paper_to_metadata(paper))

    if max_results > 0:
        metadata_to_save = metadata_to_save[:max_results]

    print(f"Found {len(metadata_to_save)} unique announcement-day papers.")
    new_count = _save_new_papers(
        metadata_to_save,
        client=client,
        extractor=extractor,
        summarizer=summarizer,
        vdb=vdb,
    )
    print(f"[{datetime.now()}] Done. Added {new_count} new papers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily ArXiv fetch job")
    parser.add_argument(
        "--mode",
        choices=["query-search", "new-submissions"],
        default="query-search",
        help="query-search uses the API query flow; new-submissions mirrors ArXivSelaa's category/day fetch.",
    )
    parser.add_argument("--query", default="", help="Search query")
    parser.add_argument("--categories", nargs="*", default=[], help="ArXiv categories")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="In query-search mode, look back N days; in new-submissions mode, start from the UTC announcement day N days ago and back up to the latest non-empty date.",
    )
    args = parser.parse_args()

    if args.mode == "new-submissions":
        run_new_submissions(
            categories=args.categories,
            max_results=args.max_results,
            days_back=args.days_back,
        )
    else:
        run_query_search(
            query=args.query,
            categories=args.categories,
            max_results=args.max_results,
            days_back=args.days_back,
        )
