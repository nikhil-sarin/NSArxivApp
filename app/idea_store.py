"""Transactional paper and grant idea persistence with legacy JSON import."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app import research_db

STORE_PATH = Path("data/ideas.json")


def _load() -> Dict:
    _migrate_legacy_if_needed()
    data = {"paper": {}, "grant": {}}
    db = research_db.connect()
    try:
        for row in db.execute("SELECT idea_type, idea_id, data_json FROM ideas ORDER BY created_at"):
            data.setdefault(row["idea_type"], {})[row["idea_id"]] = json.loads(row["data_json"])
    finally:
        db.close()
    return data


def _save(data: Dict):
    now = datetime.now(timezone.utc).isoformat()
    with research_db.transaction() as db:
        db.execute("DELETE FROM ideas")
        for idea_type, ideas in data.items():
            for idea_id, idea in ideas.items():
                created = str(idea.get("created") or now)
                db.execute(
                    "INSERT INTO ideas(idea_type, idea_id, data_json, created_at, updated_at) VALUES(?, ?, ?, ?, ?)",
                    (idea_type, idea_id, json.dumps(idea, default=str), created, now),
                )


def _migrate_legacy_if_needed() -> None:
    db = research_db.connect()
    try:
        if db.execute("SELECT 1 FROM ideas LIMIT 1").fetchone() or not STORE_PATH.exists():
            return
    finally:
        db.close()
    try:
        legacy = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    now = datetime.now(timezone.utc).isoformat()
    with research_db.transaction() as db:
        if db.execute("SELECT 1 FROM ideas LIMIT 1").fetchone():
            return
        for idea_type in ("paper", "grant"):
            for idea_id, idea in legacy.get(idea_type, {}).items():
                db.execute(
                    "INSERT INTO ideas(idea_type, idea_id, data_json, created_at, updated_at) VALUES(?, ?, ?, ?, ?)",
                    (idea_type, idea_id, json.dumps(idea, default=str), str(idea.get("created") or now), now),
                )


def save_idea(idea_type: str, title: str, description: str, extra: Optional[Dict] = None) -> str:
    """Save a new idea. Returns the idea ID."""
    data = _load()
    idea_id = str(uuid.uuid4())[:8]
    data[idea_type][idea_id] = {
        "id": idea_id,
        "title": title,
        "description": description,
        "status": "draft",
        "created": datetime.now(timezone.utc).isoformat(),
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
