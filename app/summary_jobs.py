"""Durable background summary regeneration."""

from __future__ import annotations

from pathlib import Path

from app import job_queue, paper_store
from app.paper_text import get_paper_text
from app.summary_workflow import completeness_issues, summarize_with_provenance


def _safe_metadata(paper: dict, summary: str) -> dict:
    output = {}
    for key, value in {**paper, "summary": summary}.items():
        if value is None:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        elif not isinstance(value, (str, int, float, bool)):
            value = str(value)
        output[key] = value
    return output


def _services(runtime: dict) -> dict:
    if all(runtime.get(key) is not None for key in ("arxiv_client", "pdf_extractor", "vector_db", "summarizer")):
        return runtime
    from app.arxiv_client import ArxivClient
    from app.pdf_extractor import PDFExtractor
    from app.summarizer import PaperSummarizer
    from app.vector_db import PaperVectorDB
    return {
        **runtime,
        "arxiv_client": ArxivClient(),
        "pdf_extractor": PDFExtractor(),
        "vector_db": PaperVectorDB(),
        "summarizer": PaperSummarizer(),
    }


def _run(payload: dict, runtime: dict) -> dict:
    services = _services(runtime)
    requested = payload.get("paper_ids")
    papers = paper_store.get_papers(requested) if requested is not None else paper_store.load_reading_papers()
    detailed = bool(payload.get("detailed"))
    completed = failed = 0
    errors: list[str] = []
    for index, paper in enumerate(papers, start=1):
        paper_id = str(paper.get("arxiv_id", ""))
        try:
            text = get_paper_text(
                paper_id,
                services["arxiv_client"],
                services["pdf_extractor"],
                title=paper.get("title"),
                pdf_url=paper.get("pdf_url"),
                cache_dir=Path("data/papers"),
            )
            summary, provenance = summarize_with_provenance(
                services["summarizer"],
                text,
                str(paper.get("abstract", "")),
                max_length=500 if detailed else 300,
                detailed=detailed,
            )
            if not summary:
                raise ValueError("no readable full text or abstract")
            paper["summary_provenance"] = provenance
            paper_store.save_paper(
                paper_id,
                paper,
                summary,
                summary_provenance=provenance,
            )
            vector_db = services["vector_db"]
            vector_db.collection.delete(ids=[paper_id])
            vector_db.add_paper(
                paper_id=paper_id,
                title=str(paper.get("title", paper_id)),
                summary=summary,
                metadata=_safe_metadata(paper, summary),
            )
            completed += 1
        except Exception as exc:
            failed += 1
            errors.append(f"{paper_id}: {type(exc).__name__}: {exc}")
        job_queue.update_progress(runtime["job_id"], {
            "papers_examined": len(papers),
            "papers_completed": completed,
            "current": index,
            "failed": failed,
            "errors": errors[-20:],
        })
    return {
        "papers_examined": len(papers),
        "papers_completed": completed,
        "failed": failed,
        "errors": errors[-20:],
    }


def enqueue(
    paper_ids: list[str] | None,
    *,
    detailed: bool,
    arxiv_client=None,
    pdf_extractor=None,
    vector_db=None,
    summarizer=None,
) -> str:
    runtime = {
        "arxiv_client": arxiv_client,
        "pdf_extractor": pdf_extractor,
        "vector_db": vector_db,
        "summarizer": summarizer,
    }
    return job_queue.enqueue(
        "summary_regeneration",
        {"paper_ids": paper_ids, "detailed": detailed},
        runtime=runtime,
    )


def incomplete_paper_ids() -> list[str]:
    return [
        str(paper.get("arxiv_id"))
        for paper in paper_store.load_reading_papers()
        if completeness_issues(str(paper.get("summary", "")))
    ]


job_queue.register("summary_regeneration", _run)
