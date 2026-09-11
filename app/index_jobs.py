"""Persistent background indexing jobs with repairable state."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app import corpus_index, paper_store, research_db


_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-index")
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status(job_id: str, status: str, *, result: dict | None = None, error: str = "") -> None:
    with research_db.transaction() as db:
        db.execute(
            "UPDATE jobs SET status=?, result_json=?, error=?, updated_at=? WHERE job_id=?",
            (status, json.dumps(result or {}), error, _now(), job_id),
        )


def _run_index(job_id: str, services: dict, paper_ids: list[str] | None, force: bool) -> None:
    _status(job_id, "running")
    indexed = failed = chunks = 0
    errors = []
    try:
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
        _status(job_id, "completed" if not failed else "completed_with_errors", result={
            "papers_examined": len(papers), "papers_indexed": indexed, "chunks_written": chunks,
            "failed": failed, "errors": errors[:20],
        })
    except Exception as exc:
        _status(job_id, "failed", error=str(exc))


def enqueue_library_index(*, arxiv_client, pdf_extractor, vector_db, paper_ids=None, force=False) -> str:
    """Queue one process-local worker while persisting lifecycle and results in SQLite."""
    job_id = str(uuid.uuid4())
    payload = {"paper_ids": paper_ids, "force": force}
    with research_db.transaction() as db:
        db.execute(
            "INSERT INTO jobs(job_id, kind, status, payload_json, created_at, updated_at) VALUES(?, 'library_index', 'queued', ?, ?, ?)",
            (job_id, json.dumps(payload), _now(), _now()),
        )
    services = {"arxiv_client": arxiv_client, "pdf_extractor": pdf_extractor, "vector_db": vector_db}
    with _LOCK:
        _EXECUTOR.submit(_run_index, job_id, services, paper_ids, force)
    return job_id


def list_jobs(limit: int = 20) -> list[dict]:
    db = research_db.connect()
    try:
        rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            item["result"] = json.loads(item.pop("result_json") or "{}")
            result.append(item)
        return result
    finally:
        db.close()


def repair_stale_jobs() -> int:
    """Mark interrupted queued/running jobs so operators can safely rerun them."""
    with research_db.transaction() as db:
        cursor = db.execute(
            "UPDATE jobs SET status='failed', error='worker interrupted; rerun requested', updated_at=? "
            "WHERE status IN ('queued', 'running')",
            (_now(),),
        )
        return cursor.rowcount
