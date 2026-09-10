"""Automatic, bounded citation-opportunity discovery for newly stored papers."""

from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app import citation_evidence, citation_opportunities, citation_opportunity_store
from app import contribution_catalogue, privacy, research_db
from app.summarizer import PaperSummarizer


_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="citation-discovery")
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def enabled() -> bool:
    return os.getenv("AUTO_CITATION_DISCOVERY", "true").strip().lower() in {"1", "true", "yes", "on"}


def discover_paper(paper: dict, paper_text: str, *, force: bool = False) -> dict:
    """Analyse one public paper, calling the model only after deterministic filtering."""
    paper_id = str(paper.get("arxiv_id", "")).strip()
    if not paper_id or not paper_text.strip():
        return {"paper_id": paper_id, "checked": 0, "model_judgements": 0, "reviewable": 0}

    catalogue = contribution_catalogue.load()
    provider = privacy.choose_provider(
        os.getenv("PUBLIC_LLM_PROVIDER", os.getenv("SUMMARIZER_PROVIDER", "ollama")),
        contains_private_data=False,
    )
    judge_model = PaperSummarizer(provider=provider)
    checked = judged = reviewable = 0
    for contribution in catalogue["contributions"]:
        if not contribution.get("enabled", True):
            continue
        if not force and citation_opportunity_store.has_analysis(
            paper_id,
            contribution["id"],
            citation_opportunities.ANALYSIS_VERSION,
            catalogue["schema_version"],
        ):
            continue
        citation_opportunity_store.delete_unreviewed_analyses(paper_id, contribution["id"])
        checked += 1
        packet = citation_evidence.build_evidence_packet(
            paper_id,
            paper_text,
            contribution,
            owner_name_variants=catalogue.get("owner_name_variants", [catalogue["owner"]]),
            corpus_chunks=research_db.documents_for_owner("paper_content", paper_id),
        )
        if not packet["candidate"]:
            continue
        result = citation_opportunities.judge(
            packet,
            contribution,
            catalogue_version=catalogue["schema_version"],
            complete=lambda system, user: judge_model.complete(system, user, max_length=900),
            provider=provider,
            model_name=judge_model._active_model(),
        )
        judged += 1
        if result["classification"] in {"strong_citation_opportunity", "potentially_useful"}:
            reviewable += 1
    return {"paper_id": paper_id, "checked": checked, "model_judgements": judged, "reviewable": reviewable}


def _set_job(job_id: str, status: str, *, result: dict | None = None, error: str = "") -> None:
    with research_db.transaction() as db:
        db.execute(
            "UPDATE jobs SET status=?, result_json=?, error=?, updated_at=? WHERE job_id=?",
            (status, json.dumps(result or {}), error[:2000], _now(), job_id),
        )


def _run(job_id: str, paper: dict, paper_text: str, force: bool) -> None:
    _set_job(job_id, "running")
    try:
        result = discover_paper(paper, paper_text, force=force)
    except Exception as exc:
        _set_job(job_id, "failed", error=str(exc))
    else:
        _set_job(job_id, "completed", result=result)


def enqueue_paper(paper: dict, paper_text: str, *, force: bool = False) -> str | None:
    """Queue automatic discovery without delaying an interactive paper ingest."""
    if not enabled() or not paper_text.strip():
        return None
    job_id = str(uuid.uuid4())
    payload = {"paper_id": paper.get("arxiv_id", ""), "force": force}
    with research_db.transaction() as db:
        db.execute(
            "INSERT INTO jobs(job_id, kind, status, payload_json, created_at, updated_at) "
            "VALUES(?, 'citation_discovery', 'queued', ?, ?, ?)",
            (job_id, json.dumps(payload), _now(), _now()),
        )
    with _LOCK:
        _EXECUTOR.submit(_run, job_id, dict(paper), paper_text, force)
    return job_id
