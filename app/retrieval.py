"""Hybrid retrieval and citation formatting for research documents."""

from __future__ import annotations

import json

from app import research_db


def _rrf(rank: int, weight: float = 1.0) -> float:
    return weight / (60 + rank)


def hybrid_search(query: str, *, vector_db, paper_ids: list[str] | None = None, limit: int = 8) -> list[dict]:
    """Fuse SQLite keyword and embedding ranks using reciprocal-rank fusion."""
    keyword = research_db.search_documents(
        query,
        limit=max(limit * 3, 20),
        kinds=["paper_chunk", "notes", "report", "summary", "abstract"],
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
    section = locator.get("section")
    if section:
        return f"arXiv:{paper_id} section {section} chunk {locator.get('chunk', 1)}"
    return f"arXiv:{paper_id} {item.get('kind', 'source')}"


def context_block(items: list[dict]) -> str:
    return "\n\n".join(f"[{citation_label(item)}]\n{item['text']}" for item in items)
