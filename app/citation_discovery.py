"""Automatic, bounded citation-opportunity discovery for newly stored papers."""

from __future__ import annotations

import os

from app import citation_evidence, citation_opportunities, citation_opportunity_store
from app import contribution_catalogue, job_queue, paper_store, privacy, research_db
from app.summarizer import PaperSummarizer


def enabled() -> bool:
    return os.getenv("AUTO_CITATION_DISCOVERY", "false").strip().lower() in {"1", "true", "yes", "on"}


def discover_paper(
    paper: dict,
    paper_text: str,
    *,
    force: bool = False,
    contribution_ids: set[str] | None = None,
) -> dict:
    """Analyse one public paper, calling the model only after deterministic filtering."""
    paper_id = str(paper.get("arxiv_id", "")).strip()
    if not paper_id or not paper_text.strip():
        return {"paper_id": paper_id, "checked": 0, "model_judgements": 0, "reviewable": 0}
    if citation_opportunity_store.paper_is_dismissed(paper_id):
        return {
            "paper_id": paper_id,
            "checked": 0,
            "model_judgements": 0,
            "reviewable": 0,
            "ineligible_reason": "dismissed_by_user",
        }

    catalogue = contribution_catalogue.load()
    provider = privacy.choose_provider(
        os.getenv("PUBLIC_LLM_PROVIDER", os.getenv("SUMMARIZER_PROVIDER", "ollama")),
        contains_private_data=False,
    )
    judge_model = PaperSummarizer(provider=provider)
    checked = judged = reviewable = 0
    contributions = [
        contribution for contribution in catalogue["contributions"]
        if contribution.get("enabled", True)
        and (contribution_ids is None or contribution["id"] in contribution_ids)
    ]
    if citation_evidence.paper_is_already_published(paper):
        removed = citation_opportunity_store.delete_active_for_paper(paper_id)
        return {
            "paper_id": paper_id,
            "checked": len(contributions),
            "model_judgements": 0,
            "reviewable": 0,
            "removed": removed,
            "ineligible_reason": "accepted_or_published",
        }
    analysed = set() if force else citation_opportunity_store.analyzed_contribution_ids(
        paper_id,
        citation_opportunities.ANALYSIS_VERSION,
        catalogue["schema_version"],
    )
    contributions = [item for item in contributions if item["id"] not in analysed]
    citation_opportunity_store.delete_unreviewed_analyses(
        paper_id,
        [item["id"] for item in contributions],
    )
    for contribution in contributions:
        checked += 1
        if not citation_evidence.contribution_predates_paper(paper_id, contribution):
            continue
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


def refresh_active_eligibility(arxiv_client) -> dict:
    """Refresh active matches from arXiv and remove accepted/published papers."""
    active = citation_opportunity_store.list_actionable(limit=500)
    paper_ids = sorted({
        item["paper_id"]
        for item in active
        if item.get("status") in {"proposed", "needs_review", "confirmed"}
    })
    checked = removed = failed = 0
    for paper_id in paper_ids:
        try:
            result = arxiv_client.get_result_by_id(paper_id)
            metadata = arxiv_client.get_paper_metadata(result)
            checked += 1
            if citation_evidence.paper_is_already_published(metadata):
                removed += citation_opportunity_store.delete_active_for_paper(paper_id)
                existing = paper_store.get_paper(paper_id) or {}
                if existing:
                    paper_store.save_paper(
                        paper_id,
                        {**existing, **metadata},
                        str(existing.get("summary", "")),
                        summary_provenance=existing.get("summary_provenance"),
                    )
        except Exception:
            failed += 1
    return {
        "papers_checked": checked,
        "opportunities_removed": removed,
        "failed": failed,
    }


def _run(payload: dict, runtime: dict) -> dict:
    paper_id = str(payload.get("paper_id", ""))
    paper = runtime.get("paper") or paper_store.get_paper(paper_id)
    if not paper:
        raise ValueError(f"Paper {paper_id} is no longer available")
    paper_text = str(runtime.get("paper_text", ""))
    if not paper_text:
        from pathlib import Path
        from app.arxiv_client import ArxivClient
        from app.paper_text import get_paper_text
        from app.pdf_extractor import PDFExtractor
        paper_text = get_paper_text(
            paper_id,
            ArxivClient(),
            PDFExtractor(),
            title=paper.get("title"),
            pdf_url=paper.get("pdf_url"),
            cache_dir=Path("data/papers"),
        )
    return discover_paper(paper, paper_text, force=bool(payload.get("force")))


def enqueue_paper(paper: dict, paper_text: str, *, force: bool = False) -> str | None:
    """Queue automatic discovery without delaying an interactive paper ingest."""
    if not enabled() or not paper_text.strip():
        return None
    payload = {"paper_id": paper.get("arxiv_id", ""), "force": force}
    return job_queue.enqueue(
        "citation_discovery",
        payload,
        runtime={"paper": dict(paper), "paper_text": paper_text},
    )


job_queue.register("citation_discovery", _run)
