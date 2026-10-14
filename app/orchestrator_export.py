"""Optional artifact-only handoff of relevant papers to LocalOrchestrator."""

from __future__ import annotations

import os
from typing import Any

import requests


TRUE_VALUES = {"1", "true", "yes", "on"}
MAX_SUMMARY_CHARS = 12000
MAX_ABSTRACT_CHARS = 6000
MAX_REASON_CHARS = 4000
PUBLIC_PROVENANCE_FIELDS = (
    "status",
    "provider",
    "model",
    "input_source",
    "detailed",
    "workflow_version",
)


def enabled() -> bool:
    """Require a dedicated opt-in in addition to the citation-email endpoint."""
    return (
        os.getenv("LOCAL_ORCHESTRATOR_EXPORT_RELEVANT_PAPERS", "").strip().casefold()
        in TRUE_VALUES
    )


def _bounded(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _public_summary_provenance(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key]
        for key in PUBLIC_PROVENANCE_FIELDS
        if key in value and isinstance(value[key], (str, int, float, bool, type(None)))
    }


def build_paper_bundle(paper: dict, summary: str) -> dict:
    arxiv_id = _bounded(paper.get("arxiv_id"), 100)
    title = _bounded(paper.get("title"), 1000)
    if not arxiv_id or not title:
        raise ValueError("A paper export requires arxiv_id and title")
    authors = [
        _bounded(author, 300)
        for author in paper.get("authors", [])
        if _bounded(author, 300)
    ][:100]
    categories = [
        _bounded(category, 100)
        for category in paper.get("categories", [])
        if _bounded(category, 100)
    ][:50]
    abs_url = _bounded(paper.get("abs_url"), 2000) or f"https://arxiv.org/abs/{arxiv_id}"
    author = ", ".join(authors[:3])
    if len(authors) > 3:
        author += ", et al."
    return {
        "schema_version": "1.1",
        "source": {
            "kind": "paper",
            "source_system": "nsarxivapp.daily-relevance",
            "external_id": arxiv_id,
            "title": title,
            "author": author[:500] or None,
            "permalink": abs_url,
            "trust": "public",
            "metadata": {
                "arxiv_id": arxiv_id,
                "authors": authors,
                "categories": categories,
                "abstract": _bounded(paper.get("abstract"), MAX_ABSTRACT_CHARS),
                "summary": _bounded(summary, MAX_SUMMARY_CHARS),
                "relevance_score": float(paper.get("tracking_relevance_score", 0.0)),
                "relevance_reason": _bounded(
                    paper.get("tracking_relevance_reason"), MAX_REASON_CHARS
                ),
                "summary_provenance": _public_summary_provenance(
                    paper.get("summary_provenance")
                ),
                "triage_status": "inbox",
                "export_reason": "tracking_relevance_threshold",
            },
        },
        "candidates": [],
        "update_existing_source": True,
    }


def export_relevant_paper(
    paper: dict,
    summary: str,
    *,
    url: str | None = None,
    timeout: float = 15,
) -> dict:
    if not enabled():
        raise ValueError("Relevant-paper export is not enabled")
    endpoint = (url or os.getenv("LOCAL_ORCHESTRATOR_URL", "")).strip()
    if not endpoint:
        raise ValueError(
            "LOCAL_ORCHESTRATOR_URL is required when relevant-paper export is enabled"
        )
    token = os.getenv("LOCAL_ORCHESTRATOR_API_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    bundle = build_paper_bundle(paper, summary)
    try:
        response = requests.post(endpoint, json=bundle, headers=headers, timeout=timeout)
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as exc:
        detail = str(exc)
        response = getattr(exc, "response", None)
        if response is not None and response.text.strip():
            detail = f"{detail}: {response.text.strip()[:1500]}"
        raise RuntimeError(f"LocalOrchestrator paper export failed: {detail}") from exc
    if not isinstance(result, dict) or not result.get("artifact_id"):
        raise RuntimeError("LocalOrchestrator paper export returned an invalid response")
    return result
