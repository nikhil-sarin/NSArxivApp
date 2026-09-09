"""Approval-gated action exchange contract for meetings and agentic tools."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from app import research_db


VALID_STATUSES = {"proposed", "approved", "rejected", "executed", "failed"}
VALID_ACTION_TYPES = {"task.create", "calendar.create", "notion.update", "agent.invoke"}


def propose(source: str, action_type: str, title: str, payload: dict, project_id: Optional[str] = None) -> str:
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"Unsupported action type: {action_type}")
    if not source.strip() or not title.strip() or not isinstance(payload, dict):
        raise ValueError("source, title, and an object payload are required")
    action_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    with research_db.transaction() as db:
        db.execute(
            "INSERT INTO action_proposals VALUES(?, ?, ?, ?, ?, ?, 'proposed', ?, ?)",
            (action_id, source, project_id, action_type, title, json.dumps(payload, default=str), timestamp, timestamp),
        )
    return action_id


def import_proposals(source: str, items: list[dict], project_id: Optional[str] = None) -> list[str]:
    """Validate an external agent's JSON handoff and import proposals without executing them."""
    if not isinstance(items, list):
        raise ValueError("action handoff must be a JSON list")
    action_ids = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each action proposal must be an object")
        action_ids.append(propose(
            source,
            str(item.get("action_type", "")),
            str(item.get("title", "")),
            item.get("payload", {}),
            project_id=item.get("project_id") or project_id,
        ))
    return action_ids


def update_status(action_id: str, status: str) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Unknown action status: {status}")
    with research_db.transaction() as db:
        current = db.execute("SELECT status FROM action_proposals WHERE action_id=?", (action_id,)).fetchone()
        if current is None:
            raise KeyError(action_id)
        if status == "executed" and current["status"] != "approved":
            raise ValueError("Only approved actions may be marked executed")
        db.execute(
            "UPDATE action_proposals SET status=?, updated_at=? WHERE action_id=?",
            (status, datetime.now(timezone.utc).isoformat(), action_id),
        )


def list_actions(status: Optional[str] = None, limit: int = 100) -> list[dict]:
    db = research_db.connect()
    try:
        where = "WHERE status=?" if status else ""
        params: list[object] = [status] if status else []
        params.append(limit)
        rows = db.execute(
            f"SELECT * FROM action_proposals {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        actions = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            actions.append(item)
        return actions
    finally:
        db.close()
