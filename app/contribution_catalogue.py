"""Versioned, user-maintained catalogue of personal research contributions."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


DEFAULT_PATH = Path("config/contributions.json")
EXAMPLE_PATH = Path("config/contributions.example.json")


class CatalogueError(ValueError):
    pass


def normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _validate_entry(entry: dict, seen: set[str]) -> dict:
    entry_id = str(entry.get("id", "")).strip()
    prefix = f"contribution {entry_id or '<missing>'}"
    if not entry_id:
        raise CatalogueError(f"{prefix}: id is required")
    if entry_id in seen:
        raise CatalogueError(f"{prefix}: duplicate id")
    seen.add(entry_id)
    for field in ("name", "kind", "contact_framing"):
        if not str(entry.get(field, "")).strip():
            raise CatalogueError(f"{prefix}: {field} is required")
    citations = entry.get("canonical_citations", [])
    if not entry.get("unpublished") and not any(
        citation.get(key) for citation in citations for key in ("doi", "arxiv_id", "bibcode", "title")
    ):
        raise CatalogueError(f"{prefix}: canonical citation identifier or unpublished=true is required")
    signals = list(entry.get("strong_signals", [])) + list(entry.get("weak_signals", []))
    if not signals:
        raise CatalogueError(f"{prefix}: at least one strong or weak signal is required")
    aliases = [normalize(alias) for alias in entry.get("aliases", []) if normalize(alias)]
    if len(aliases) != len(set(aliases)):
        raise CatalogueError(f"{prefix}: aliases must be unique after normalization")
    for rule in entry.get("discovery_rules", []):
        groups = rule.get("all", [])
        if rule.get("strength") not in {"strong", "weak"} or not str(rule.get("name", "")).strip():
            raise CatalogueError(f"{prefix}: discovery rule requires a name and strong/weak strength")
        if len(groups) < 2 or any(not isinstance(group, list) or not any(normalize(term) for term in group) for group in groups):
            raise CatalogueError(f"{prefix}: discovery rule requires at least two non-empty term groups")
    return {**entry, "aliases": aliases}


def validate(data: dict) -> dict:
    if not re.fullmatch(r"1\.\d+", str(data.get("schema_version", ""))):
        raise CatalogueError("catalogue schema_version must be 1.x")
    if not str(data.get("owner", "")).strip():
        raise CatalogueError("catalogue owner is required")
    seen: set[str] = set()
    contributions = [_validate_entry(entry, seen) for entry in data.get("contributions", [])]
    if not contributions:
        raise CatalogueError("catalogue must contain at least one contribution")
    return {**data, "contributions": contributions}


def configured_path() -> Path:
    return Path(os.getenv("NSARXIV_CONTRIBUTIONS_PATH", str(DEFAULT_PATH))).expanduser()


def load(path: Path | None = None) -> dict:
    path = path or configured_path()
    if not path.exists() and path == DEFAULT_PATH:
        path = EXAMPLE_PATH
    try:
        return validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogueError(f"could not read contribution catalogue {path}: {exc}") from exc


def get(data: dict, contribution_id: str) -> dict:
    for contribution in data["contributions"]:
        if contribution["id"] == contribution_id:
            return contribution
    raise KeyError(contribution_id)
