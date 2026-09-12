"""Persistent background indexing jobs with repairable state."""

from __future__ import annotations

from app import corpus_index, job_queue, paper_store


def _run_index(payload: dict, services: dict) -> dict:
    paper_ids = payload.get("paper_ids")
    force = bool(payload.get("force"))
    if not services:
        from app.arxiv_client import ArxivClient
        from app.pdf_extractor import PDFExtractor
        from app.vector_db import PaperVectorDB
        services = {
            "arxiv_client": ArxivClient(),
            "pdf_extractor": PDFExtractor(),
            "vector_db": PaperVectorDB(),
        }
    indexed = failed = chunks = 0
    errors = []
    papers = paper_store.load_reading_papers()
    if paper_ids is not None:
        wanted = set(paper_ids)
        papers = [paper for paper in papers if paper.get("arxiv_id") in wanted]
    for paper in papers:
        try:
            chunks += corpus_index.index_paper(
                paper,
                arxiv_client=services["arxiv_client"],
                pdf_extractor=services["pdf_extractor"],
                vector_db=services["vector_db"],
                force=force,
            )
            indexed += 1
        except Exception as exc:
            failed += 1
            errors.append(f"{paper.get('arxiv_id', 'unknown')}: {exc}")
    return {
        "papers_examined": len(papers), "papers_indexed": indexed, "chunks_written": chunks,
        "failed": failed, "errors": errors[:20],
    }


def enqueue_library_index(*, arxiv_client, pdf_extractor, vector_db, paper_ids=None, force=False) -> str:
    """Queue one process-local worker while persisting lifecycle and results in SQLite."""
    payload = {"paper_ids": paper_ids, "force": force}
    services = {"arxiv_client": arxiv_client, "pdf_extractor": pdf_extractor, "vector_db": vector_db}
    return job_queue.enqueue("library_index", payload, runtime=services)


def list_jobs(limit: int = 20) -> list[dict]:
    return job_queue.list_jobs(limit)


def repair_stale_jobs() -> int:
    """Requeue interrupted work and wake the durable worker."""
    repaired = job_queue.recover_interrupted()
    job_queue.start()
    return repaired


job_queue.register("library_index", _run_index)
