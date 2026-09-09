"""Backfill citation discovery across a bounded slice of the stored library."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from app import citation_discovery, contribution_catalogue, paper_store
from app.arxiv_client import ArxivClient
from app.paper_text import get_paper_text
from app.pdf_extractor import PDFExtractor


def _published(paper: dict) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(paper.get("published", "")).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


def eligible_papers(papers: list[dict], *, owner: str, days: int, limit: int, now=None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    owner_lower = owner.strip().lower()
    eligible = []
    seen: set[str] = set()
    for paper in sorted(
        papers,
        key=lambda item: _published(item) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    ):
        paper_id = str(paper.get("arxiv_id", "")).split("v")[0]
        published = _published(paper)
        first_author = str((paper.get("authors") or [""])[0]).strip().lower()
        if not paper_id or paper_id in seen or not published or published < cutoff or first_author == owner_lower:
            continue
        seen.add(paper_id)
        eligible.append(paper)
        if len(eligible) >= limit:
            break
    return eligible


def run(*, days: int, max_papers: int, force: bool = False) -> dict:
    catalogue = contribution_catalogue.load()
    papers = eligible_papers(
        paper_store.load_all_papers(),
        owner=catalogue["owner"],
        days=days,
        limit=max_papers,
    )
    client = ArxivClient()
    extractor = PDFExtractor()
    totals = {
        "papers_selected": len(papers),
        "papers_processed": 0,
        "papers_failed": 0,
        "contributions_checked": 0,
        "model_judgements": 0,
        "reviewable": 0,
    }
    for index, paper in enumerate(papers, start=1):
        paper_id = str(paper.get("arxiv_id", ""))
        try:
            text = get_paper_text(
                paper_id,
                client,
                extractor,
                title=paper.get("title"),
                pdf_url=paper.get("pdf_url"),
                cache_dir=Path("data/papers"),
            )
            result = citation_discovery.discover_paper(paper, text, force=force)
        except Exception as exc:
            totals["papers_failed"] += 1
            print(f"[{index}/{len(papers)}] {paper_id}: failed: {exc}", flush=True)
            continue
        totals["papers_processed"] += 1
        totals["contributions_checked"] += result["checked"]
        totals["model_judgements"] += result["model_judgements"]
        totals["reviewable"] += result["reviewable"]
        print(
            f"[{index}/{len(papers)}] {paper_id}: checked={result['checked']} "
            f"judged={result['model_judgements']} reviewable={result['reviewable']}",
            flush=True,
        )
    return totals


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Backfill evidence-first citation discovery")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--max-papers", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(days=args.days, max_papers=args.max_papers, force=args.force), sort_keys=True))
