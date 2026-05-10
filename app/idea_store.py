"""Persistent store for paper and grant ideas at data/ideas.json."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

STORE_PATH = Path("data/ideas.json")


def _load() -> Dict:
    if not STORE_PATH.exists():
        return {"paper": {}, "grant": {}}
    try:
        data = json.loads(STORE_PATH.read_text())
        # Ensure both keys exist
        data.setdefault("paper", {})
        data.setdefault("grant", {})
        return data
    except Exception:
        return {"paper": {}, "grant": {}}


def _save(data: Dict):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(data, indent=2, default=str))


def save_idea(idea_type: str, title: str, description: str, extra: Optional[Dict] = None) -> str:
    """Save a new idea. Returns the idea ID."""
    data = _load()
    idea_id = str(uuid.uuid4())[:8]
    data[idea_type][idea_id] = {
        "id": idea_id,
        "title": title,
        "description": description,
        "status": "draft",
        "created": datetime.now().isoformat(),
        "chat_history": [],
        "linked_papers": [],
        "notes": "",
        **(extra or {}),
    }
    _save(data)
    return idea_id


def load_ideas(idea_type: str) -> List[Dict]:
    return list(_load()[idea_type].values())


def get_idea(idea_type: str, idea_id: str) -> Optional[Dict]:
    return _load()[idea_type].get(idea_id)


def update_idea(idea_type: str, idea_id: str, updates: Dict):
    data = _load()
    if idea_id in data[idea_type]:
        data[idea_type][idea_id].update(updates)
        _save(data)


def delete_idea(idea_type: str, idea_id: str):
    data = _load()
    data[idea_type].pop(idea_id, None)
    _save(data)


def append_chat(idea_type: str, idea_id: str, role: str, content: str):
    data = _load()
    if idea_id in data[idea_type]:
        data[idea_type][idea_id].setdefault("chat_history", [])
        data[idea_type][idea_id]["chat_history"].append({"role": role, "content": content})
        _save(data)


def set_linked_papers(idea_type: str, idea_id: str, paper_ids: List[str]):
    """Replace the papers linked to an idea."""
    data = _load()
    if idea_id in data[idea_type]:
        seen = set()
        clean_ids = []
        for paper_id in paper_ids:
            if paper_id and paper_id not in seen:
                clean_ids.append(paper_id)
                seen.add(paper_id)
        data[idea_type][idea_id]["linked_papers"] = clean_ids
        _save(data)
