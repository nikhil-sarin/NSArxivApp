"""Persistent JSON store for paper metadata, summaries, and research notes."""

import json
from pathlib import Path
from typing import Dict, List, Optional


STORE_PATH = Path("data/papers.json")

DEFAULT_NOTES = {
    "key_result": "",
    "why_i_care": "",
    "cite_for": "",
    "caveats": "",
    "follow_up": "",
}


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
    existing = data.get(paper_id, {})
    data[paper_id] = {**existing, **metadata, "summary": summary}
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


def update_paper(paper_id: str, updates: Dict):
    """Update stored fields for one paper."""
    data = _load()
    if paper_id in data:
        data[paper_id].update(updates)
        _save(data)


def get_paper(paper_id: str) -> Optional[Dict]:
    """Return one stored paper by ID."""
    return _load().get(paper_id)


def get_notes(paper_id: str) -> Dict:
    """Return normalized research notes for one paper."""
    paper = get_paper(paper_id) or {}
    notes = paper.get("research_notes", {})
    if not isinstance(notes, dict):
        notes = {}
    return {**DEFAULT_NOTES, **notes}


def save_notes(paper_id: str, notes: Dict):
    """Persist structured research notes for one paper."""
    clean_notes = {key: str(notes.get(key, "")) for key in DEFAULT_NOTES}
    data = _load()
    if paper_id in data:
        data[paper_id]["research_notes"] = clean_notes
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
