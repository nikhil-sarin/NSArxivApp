"""Persistent JSON store for paper metadata and summaries."""

import json
from pathlib import Path
from typing import Dict, List, Optional


STORE_PATH = Path("data/papers.json")


def _load() -> Dict[str, Dict]:
    if not STORE_PATH.exists():
        return {}
    with open(STORE_PATH) as f:
        return json.load(f)


def _save(data: Dict[str, Dict]):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_paper(paper_id: str, metadata: Dict, summary: str):
    """Persist a paper's metadata and summary."""
    data = _load()
    data[paper_id] = {**metadata, "summary": summary}
    _save(data)


def load_all_papers() -> List[Dict]:
    """Return all stored papers as a list."""
    return list(_load().values())


def paper_exists(paper_id: str) -> bool:
    return paper_id in _load()


def delete_paper(paper_id: str):
    """Remove a paper from the JSON store."""
    data = _load()
    data.pop(paper_id, None)
    _save(data)


def delete_paper(paper_id: str):
    """Remove a paper from the JSON store."""
    data = _load()
    data.pop(paper_id, None)
    _save(data)


def search_by_author(author_query: str) -> List[Dict]:
    """Return papers where any author name contains the query (case-insensitive)."""
    q = author_query.lower().strip()
    results = []
    for paper in _load().values():
        authors = paper.get("authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",")]
        if any(q in a.lower() for a in authors):
            results.append(paper)
    return results


def search_by_text(text_query: str) -> List[Dict]:
    """Full-text search across title, summary, and abstract (case-insensitive)."""
    q = text_query.lower().strip()
    results = []
    for paper in _load().values():
        haystack = " ".join([
            paper.get("title", ""),
            paper.get("summary", ""),
            paper.get("abstract", ""),
        ]).lower()
        if q in haystack:
            results.append(paper)
    return results
