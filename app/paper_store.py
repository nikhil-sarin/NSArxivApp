"""Transactional paper, note, triage, and search persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app import research_db


STORE_PATH = research_db.DB_PATH
LEGACY_STORE_PATH = Path("data/papers.json")

DEFAULT_NOTES = {
    "key_result": "",
    "why_i_care": "",
    "cite_for": "",
    "caveats": "",
    "follow_up": "",
}

TRIAGE_STATUSES = ("inbox", "read_later", "skimmed", "read", "saved", "irrelevant")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_notes(paper: Dict) -> Dict:
    notes = paper.get("research_notes", {})
    if not isinstance(notes, dict):
        notes = {}
    return {**DEFAULT_NOTES, **notes}


def _notes_text(paper: Dict) -> str:
    notes = _normalized_notes(paper)
    return "\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in notes.items() if value.strip())


def _index_paper(db, paper_id: str, paper: Dict) -> None:
    title = str(paper.get("title", paper_id))
    authors = paper.get("authors", [])
    categories = paper.get("categories", [])
    if isinstance(authors, list):
        authors = " ".join(str(author) for author in authors)
    if isinstance(categories, list):
        categories = " ".join(str(category) for category in categories)
    documents = {
        "metadata": " ".join([title, str(authors), str(categories)]),
        "summary": str(paper.get("summary", "") or ""),
        "abstract": str(paper.get("abstract", "") or ""),
        "notes": _notes_text(paper),
    }
    for kind, text in documents.items():
        document_id = f"paper:{paper_id}:{kind}"
        if text.strip():
            research_db.upsert_document(
                document_id,
                kind=kind,
                owner_type="paper",
                owner_id=paper_id,
                paper_id=paper_id,
                title=title,
                text=text,
                locator={"arxiv_id": paper_id, "kind": kind},
                connection=db,
            )
        else:
            db.execute("DELETE FROM document_fts WHERE document_id = ?", (document_id,))
            db.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))


def _migrate_legacy_if_needed() -> None:
    db = research_db.connect()
    try:
        count = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    finally:
        db.close()
    if count or not LEGACY_STORE_PATH.exists():
        return

    try:
        legacy = json.loads(LEGACY_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(legacy, dict):
        return

    timestamp = _now()
    with research_db.transaction() as db:
        if db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]:
            return
        for paper_id, paper in legacy.items():
            if not isinstance(paper, dict):
                continue
            db.execute(
                "INSERT INTO papers(paper_id, data_json, created_at, updated_at) VALUES(?, ?, ?, ?)",
                (paper_id, json.dumps(paper, default=str), timestamp, timestamp),
            )
            db.execute(
                "INSERT INTO triage(paper_id, status, updated_at) VALUES(?, 'saved', ?)",
                (paper_id, timestamp),
            )
            _index_paper(db, paper_id, paper)


def _load() -> Dict[str, Dict]:
    """Compatibility helper returning the historical ID-to-paper mapping."""
    _migrate_legacy_if_needed()
    db = research_db.connect()
    try:
        rows = db.execute("SELECT paper_id, data_json FROM papers ORDER BY created_at").fetchall()
        return {row["paper_id"]: json.loads(row["data_json"]) for row in rows}
    finally:
        db.close()


def _save(data: Dict[str, Dict]):
    """Compatibility bulk replacement used by older maintenance scripts."""
    timestamp = _now()
    with research_db.transaction() as db:
        db.execute("DELETE FROM document_fts")
        db.execute("DELETE FROM documents WHERE owner_type='paper'")
        db.execute("DELETE FROM triage")
        db.execute("DELETE FROM papers")
        for paper_id, paper in data.items():
            db.execute(
                "INSERT INTO papers(paper_id, data_json, created_at, updated_at) VALUES(?, ?, ?, ?)",
                (paper_id, json.dumps(paper, default=str), timestamp, timestamp),
            )
            db.execute(
                "INSERT INTO triage(paper_id, status, updated_at) VALUES(?, 'saved', ?)",
                (paper_id, timestamp),
            )
            _index_paper(db, paper_id, paper)


def save_paper(paper_id: str, metadata: Dict, summary: str):
    """Persist a paper atomically and place newly discovered papers in the inbox."""
    _migrate_legacy_if_needed()
    timestamp = _now()
    with research_db.transaction() as db:
        row = db.execute("SELECT data_json FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        existing = json.loads(row["data_json"]) if row else {}
        paper = {**existing, **metadata, "summary": summary}
        db.execute(
            """
            INSERT INTO papers(paper_id, data_json, created_at, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET data_json=excluded.data_json, updated_at=excluded.updated_at
            """,
            (paper_id, json.dumps(paper, default=str), timestamp, timestamp),
        )
        if row is None:
            db.execute(
                "INSERT INTO triage(paper_id, status, updated_at) VALUES(?, 'inbox', ?)",
                (paper_id, timestamp),
            )
        _index_paper(db, paper_id, paper)


def load_all_papers() -> List[Dict]:
    return list(_load().values())


def load_reading_papers() -> List[Dict]:
    """Return papers admitted to the reading workflow, excluding monitoring-only records."""
    _migrate_legacy_if_needed()
    db = research_db.connect()
    try:
        rows = db.execute(
            """
            SELECT p.data_json
            FROM papers p JOIN triage t USING(paper_id)
            WHERE t.status != 'irrelevant'
            ORDER BY p.updated_at DESC
            """
        ).fetchall()
        return [json.loads(row["data_json"]) for row in rows]
    finally:
        db.close()


def rebuild_search_index() -> int:
    """Rebuild paper metadata, summary, abstract, and note FTS records."""
    papers = _load()
    with research_db.transaction() as db:
        rows = db.execute("SELECT document_id FROM documents WHERE owner_type='paper'").fetchall()
        for row in rows:
            db.execute("DELETE FROM document_fts WHERE document_id = ?", (row["document_id"],))
        db.execute("DELETE FROM documents WHERE owner_type='paper'")
        for paper_id, paper in papers.items():
            _index_paper(db, paper_id, paper)
    return len(papers)


def paper_exists(paper_id: str) -> bool:
    _migrate_legacy_if_needed()
    db = research_db.connect()
    try:
        return db.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)).fetchone() is not None
    finally:
        db.close()


def delete_paper(paper_id: str):
    _migrate_legacy_if_needed()
    with research_db.transaction() as db:
        research_db.delete_documents("paper", paper_id, connection=db)
        db.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))


def update_paper(paper_id: str, updates: Dict):
    paper = get_paper(paper_id)
    if paper is None:
        return
    paper.update(updates)
    save_paper(paper_id, paper, str(paper.get("summary", "")))


def get_paper(paper_id: str) -> Optional[Dict]:
    _migrate_legacy_if_needed()
    db = research_db.connect()
    try:
        row = db.execute("SELECT data_json FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return json.loads(row["data_json"]) if row else None
    finally:
        db.close()


def resolve_paper_id(arxiv_id: str) -> Optional[str]:
    """Resolve a versionless arXiv ID to the stored paper key, if present."""
    canonical = str(arxiv_id).split("v")[0]
    _migrate_legacy_if_needed()
    db = research_db.connect()
    try:
        row = db.execute(
            "SELECT paper_id FROM papers WHERE paper_id=? OR paper_id LIKE ? "
            "ORDER BY paper_id DESC LIMIT 1",
            (canonical, f"{canonical}v%"),
        ).fetchone()
        return row["paper_id"] if row else None
    finally:
        db.close()


def get_triage(paper_id: str) -> Optional[Dict]:
    """Return the current triage state for one stored paper."""
    _migrate_legacy_if_needed()
    db = research_db.connect()
    try:
        row = db.execute(
            "SELECT status, feedback, relevance_score, relevance_reason "
            "FROM triage WHERE paper_id=?",
            (paper_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        db.close()


def get_notes(paper_id: str) -> Dict:
    return _normalized_notes(get_paper(paper_id) or {})


def save_notes(paper_id: str, notes: Dict):
    paper = get_paper(paper_id)
    if paper is None:
        return
    paper["research_notes"] = {key: str(notes.get(key, "")) for key in DEFAULT_NOTES}
    save_paper(paper_id, paper, str(paper.get("summary", "")))


def update_triage(
    paper_id: str,
    *,
    status: Optional[str] = None,
    feedback: Optional[int] = None,
    relevance_score: Optional[float] = None,
    relevance_reason: Optional[str] = None,
) -> None:
    if status is not None and status not in TRIAGE_STATUSES:
        raise ValueError(f"Unknown triage status: {status}")
    if feedback is not None and feedback not in (-1, 0, 1):
        raise ValueError("feedback must be -1, 0, or 1")
    _migrate_legacy_if_needed()
    fields = ["updated_at = ?"]
    params: list[object] = [_now()]
    for name, value in (
        ("status", status),
        ("feedback", feedback),
        ("relevance_score", relevance_score),
        ("relevance_reason", relevance_reason),
    ):
        if value is not None:
            fields.append(f"{name} = ?")
            params.append(value)
    params.append(paper_id)
    with research_db.transaction() as db:
        db.execute(f"UPDATE triage SET {', '.join(fields)} WHERE paper_id = ?", params)


def load_triage(status: Optional[str] = "inbox", limit: int = 100) -> List[Dict]:
    _migrate_legacy_if_needed()
    db = research_db.connect()
    try:
        where = "WHERE t.status = ?" if status else ""
        params: list[object] = [status] if status else []
        params.append(limit)
        rows = db.execute(
            f"""
            SELECT p.data_json, t.status, t.feedback, t.relevance_score, t.relevance_reason
            FROM triage t JOIN papers p USING(paper_id)
            {where}
            ORDER BY COALESCE(t.relevance_score, -1) DESC, p.updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        output = []
        for row in rows:
            paper = json.loads(row["data_json"])
            paper["triage"] = {
                "status": row["status"],
                "feedback": row["feedback"],
                "relevance_score": row["relevance_score"],
                "relevance_reason": row["relevance_reason"],
            }
            output.append(paper)
        return output
    finally:
        db.close()


def search_by_author(author_query: str) -> List[Dict]:
    query = author_query.lower().strip()
    results = []
    for paper in load_reading_papers():
        authors = paper.get("authors", [])
        if isinstance(authors, str):
            authors = [author.strip() for author in authors.split(",")]
        if any(query in author.lower() for author in authors):
            results.append(paper)
    return results


def search_by_text(text_query: str) -> List[Dict]:
    """Full-text search across summaries, abstracts, and structured notes."""
    try:
        matches = research_db.search_documents(text_query, limit=200)
    except Exception:
        query = text_query.lower().strip()
        return [
            paper for paper in load_reading_papers()
            if query in " ".join(
                [
                    str(paper.get("title", "")),
                    str(paper.get("summary", "")),
                    str(paper.get("abstract", "")),
                    _notes_text(paper),
                ]
            ).lower()
        ]
    reading_ids = {
        paper.get("arxiv_id") for paper in load_reading_papers()
    }
    seen = set()
    papers = []
    for match in matches:
        paper_id = match.get("paper_id")
        if not paper_id or paper_id in seen:
            continue
        paper = get_paper(paper_id)
        if paper and paper_id in reading_ids:
            papers.append(paper)
            seen.add(paper_id)
    return papers
