"""Approval-gated action exchange contract for meetings and agentic tools."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from app import research_db


VALID_STATUSES = {"proposed", "approved", "rejected", "executed", "failed"}


def propose(source: str, action_type: str, title: str, payload: dict, project_id: Optional[str] = None) -> str:
    action_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    with research_db.transaction() as db:
        db.execute(
            "INSERT INTO action_proposals VALUES(?, ?, ?, ?, ?, ?, 'proposed', ?, ?)",
            (action_id, source, project_id, action_type, title, json.dumps(payload, default=str), timestamp, timestamp),
        )
    return action_id


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
