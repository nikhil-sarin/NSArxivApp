"""Transactional researcher profile with legacy JSON import."""

import json
from pathlib import Path
from typing import Dict

from app import research_db

PROFILE_PATH = Path("data/profile.json")

DEFAULT_PROFILE = {
    "name": "",
    "position": "",
    "institution": "",
    "research_areas": "",
    "methods_and_tools": "",
    "bio": "",
}


def load() -> Dict:
    db = research_db.connect()
    try:
        row = db.execute("SELECT value_json FROM settings WHERE key='researcher_profile'").fetchone()
    finally:
        db.close()
    if row:
        return {**DEFAULT_PROFILE, **json.loads(row[0])}
    if PROFILE_PATH.exists():
        try:
            profile = {**DEFAULT_PROFILE, **json.loads(PROFILE_PATH.read_text())}
            save(profile)
            return profile
        except (OSError, json.JSONDecodeError):
            pass
    return dict(DEFAULT_PROFILE)


def save(profile: Dict):
    with research_db.transaction() as db:
        db.execute(
            "INSERT INTO settings(key, value_json, updated_at) VALUES('researcher_profile', ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
            (json.dumps({**DEFAULT_PROFILE, **profile}),),
        )


def is_empty(profile: Dict) -> bool:
    return not any(profile.get(k, "").strip() for k in DEFAULT_PROFILE)


def to_context_string(profile: Dict) -> str:
    """Format profile as a context block for LLM prompts."""
    if is_empty(profile):
        return ""
    lines = ["## Researcher Profile"]
    if profile.get("name"):
        lines.append(f"Name: {profile['name']}")
    if profile.get("position") or profile.get("institution"):
        lines.append(f"Position: {profile.get('position', '')} at {profile.get('institution', '')}".strip(" at"))
    if profile.get("research_areas"):
        lines.append(f"Research areas: {profile['research_areas']}")
    if profile.get("methods_and_tools"):
        lines.append(f"Methods & tools: {profile['methods_and_tools']}")
    if profile.get("bio"):
        lines.append(f"Bio: {profile['bio']}")
    return "\n".join(lines)
