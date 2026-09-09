"""Transactional persistence for stable citation-opportunity analyses."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app import research_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(opportunity: dict) -> None:
    now = _now()
    with research_db.transaction() as db:
        existing = db.execute(
            "SELECT created_at, status, export_status, export_error FROM citation_opportunities WHERE opportunity_id=?",
            (opportunity["opportunity_id"],),
        ).fetchone()
        db.execute(
            """
            INSERT OR REPLACE INTO citation_opportunities(
                opportunity_id, paper_id, contribution_id, analysis_version, catalogue_version,
                classification, confidence, status, data_json, export_status, export_error,
                created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                opportunity["opportunity_id"], opportunity["paper_id"], opportunity["contribution_id"],
                opportunity["analysis_version"], opportunity["catalogue_version"],
                opportunity["classification"], opportunity["confidence"],
                existing["status"] if existing else opportunity.get("status", "proposed"),
                json.dumps(opportunity, default=str), existing["export_status"] if existing else "",
                existing["export_error"] if existing else "", existing["created_at"] if existing else now, now,
            ),
        )


def list_opportunities(status: str | None = None, limit: int = 100) -> list[dict]:
    db = research_db.connect()
    try:
        where = "WHERE status=?" if status else ""
        params: list[object] = [status] if status else []
        params.append(limit)
        rows = db.execute(f"SELECT * FROM citation_opportunities {where} ORDER BY updated_at DESC LIMIT ?", params).fetchall()
        result = []
        for row in rows:
            item = json.loads(row["data_json"])
            item.update({"status": row["status"], "export_status": row["export_status"], "export_error": row["export_error"]})
            result.append(item)
        return result
    finally:
        db.close()


def has_analysis(paper_id: str, contribution_id: str, analysis_version: str, catalogue_version: str) -> bool:
    db = research_db.connect()
    try:
        return db.execute(
            "SELECT 1 FROM citation_opportunities WHERE paper_id=? AND contribution_id=? "
            "AND analysis_version=? AND catalogue_version=? LIMIT 1",
            (paper_id, contribution_id, analysis_version, catalogue_version),
        ).fetchone() is not None
    finally:
        db.close()


def update_status(opportunity_id: str, status: str) -> None:
    if status not in {"proposed", "confirmed", "not_relevant", "needs_review", "exported"}:
        raise ValueError(f"invalid opportunity status: {status}")
    with research_db.transaction() as db:
        if not db.execute("SELECT 1 FROM citation_opportunities WHERE opportunity_id=?", (opportunity_id,)).fetchone():
            raise KeyError(opportunity_id)
        db.execute("UPDATE citation_opportunities SET status=?, updated_at=? WHERE opportunity_id=?", (status, _now(), opportunity_id))


def record_export(opportunity_id: str, *, success: bool, error: str = "") -> None:
    with research_db.transaction() as db:
        db.execute(
            "UPDATE citation_opportunities SET status=CASE WHEN ? THEN 'exported' ELSE status END, export_status=?, export_error=?, updated_at=? WHERE opportunity_id=?",
            (success, "exported" if success else "failed", error[:2000], _now(), opportunity_id),
        )
