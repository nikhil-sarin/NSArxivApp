"""Persistent researcher profile stored at data/profile.json."""

import json
from pathlib import Path
from typing import Dict

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
    if not PROFILE_PATH.exists():
        return dict(DEFAULT_PROFILE)
    try:
        return {**DEFAULT_PROFILE, **json.loads(PROFILE_PATH.read_text())}
    except Exception:
        return dict(DEFAULT_PROFILE)


def save(profile: Dict):
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))


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
