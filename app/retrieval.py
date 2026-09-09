"""Hybrid retrieval and citation formatting for research documents."""

from __future__ import annotations

import json
import re

from app import research_db


def _rrf(rank: int, weight: float = 1.0) -> float:
    return weight / (60 + rank)


def hybrid_search(
    query: str,
    *,
    vector_db,
    paper_ids: list[str] | None = None,
    limit: int = 8,
    kinds: list[str] | None = None,
) -> list[dict]:
    """Fuse SQLite keyword and embedding ranks using reciprocal-rank fusion."""
    keyword = research_db.search_documents(
        query,
        limit=max(limit * 3, 20),
        kinds=kinds,
        paper_ids=paper_ids,
    )
    semantic = vector_db.search_documents(query, top_k=max(limit * 3, 20), paper_ids=paper_ids)
    fused: dict[str, dict] = {}
    for rank, item in enumerate(keyword, start=1):
        document_id = item["document_id"]
        fused[document_id] = {
            "document_id": document_id,
            "text": item["text"],
            "paper_id": item.get("paper_id"),
            "title": item.get("title", ""),
            "kind": item.get("kind", ""),
            "locator": item.get("locator", {}),
            "score": _rrf(rank, 1.15),
        }
    for rank, item in enumerate(semantic, start=1):
        metadata = item.get("metadata", {})
        document_id = item["document_id"]
        locator = json.loads(metadata.get("locator_json", "{}"))
        entry = fused.setdefault(
            document_id,
            {
                "document_id": document_id,
                "text": item.get("text", ""),
                "paper_id": metadata.get("paper_id"),
                "title": metadata.get("title", ""),
                "kind": metadata.get("kind", ""),
                "locator": locator,
                "score": 0.0,
            },
        )
        entry["score"] += _rrf(rank)
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:limit]


def citation_label(item: dict) -> str:
    locator = item.get("locator", {})
    paper_id = item.get("paper_id") or locator.get("arxiv_id", "unknown")
    if locator.get("page"):
        return f"arXiv:{paper_id} p.{locator['page']} chunk {locator.get('chunk', 1)}"
    if locator.get("figure"):
        return f"arXiv:{paper_id} fig.{locator['figure']}"
    section = locator.get("section")
    if section:
        return f"arXiv:{paper_id} section {section} chunk {locator.get('chunk', 1)}"
    return f"arXiv:{paper_id} {item.get('kind', 'source')}"


def source_url(item: dict) -> str:
    """Return a stable source URL when the record originates from an ArXiv paper."""
    paper_id = item.get("paper_id") or item.get("locator", {}).get("arxiv_id")
    if not paper_id:
        return ""
    page = item.get("locator", {}).get("page")
    suffix = f"#page={page}" if page else ""
    return f"https://arxiv.org/pdf/{paper_id}{suffix}"


def citation_markdown(item: dict) -> str:
    label = citation_label(item)
    url = source_url(item)
    return f"[{label}]({url})" if url else f"[{label}]"


def context_block(items: list[dict]) -> str:
    return "\n\n".join(f"{citation_markdown(item)}\n{item['text']}" for item in items)


def linkify_citations(answer: str, items: list[dict]) -> str:
    """Turn only citations present in retrieved evidence into clickable links."""
    for item in items:
        label = citation_label(item)
        url = source_url(item)
        if url:
            answer = re.sub(
                rf"\[{re.escape(label)}\](?!\()",
                f"[{label}]({url})",
                answer,
            )
    return answer
