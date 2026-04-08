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
