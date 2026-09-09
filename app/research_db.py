"""Transactional storage and full-text indexing for research workflow data."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


DB_PATH = Path("data/research.db")
SCHEMA_VERSION = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open an initialized SQLite connection suitable for concurrent app/jobs."""
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    _ensure_schema(connection)
    return connection


@contextmanager
def transaction(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            data_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS triage (
            paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
            status TEXT NOT NULL DEFAULT 'inbox',
            relevance_score REAL,
            relevance_reason TEXT NOT NULL DEFAULT '',
            feedback INTEGER NOT NULL DEFAULT 0 CHECK(feedback IN (-1, 0, 1)),
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            owner_type TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            paper_id TEXT,
            title TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL,
            locator_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
            document_id UNINDEXED,
            title,
            text,
            tokenize='porter unicode61'
        );

        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT NOT NULL DEFAULT '{}',
            error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'research',
            status TEXT NOT NULL DEFAULT 'active',
            description TEXT NOT NULL DEFAULT '',
            methods TEXT NOT NULL DEFAULT '',
            repository_paths_json TEXT NOT NULL DEFAULT '[]',
            notion_url TEXT NOT NULL DEFAULT '',
            notion_export_path TEXT NOT NULL DEFAULT '',
            data_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS project_papers (
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
            relationship TEXT NOT NULL DEFAULT 'relevant',
            created_at TEXT NOT NULL,
            PRIMARY KEY(project_id, paper_id)
        );

        CREATE TABLE IF NOT EXISTS project_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS action_proposals (
            action_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            project_id TEXT REFERENCES projects(project_id) ON DELETE SET NULL,
            action_type TEXT NOT NULL,
            title TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'proposed',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evaluation_runs (
            run_id TEXT PRIMARY KEY,
            suite TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ideas (
            idea_type TEXT NOT NULL,
            idea_id TEXT NOT NULL,
            data_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(idea_type, idea_id)
        );

        CREATE TABLE IF NOT EXISTS citation_opportunities (
            opportunity_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            contribution_id TEXT NOT NULL,
            analysis_version TEXT NOT NULL,
            catalogue_version TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'proposed',
            data_json TEXT NOT NULL,
            export_status TEXT NOT NULL DEFAULT '',
            export_error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    connection.commit()


def upsert_document(
    document_id: str,
    *,
    kind: str,
    owner_type: str,
    owner_id: str,
    text: str,
    title: str = "",
    paper_id: Optional[str] = None,
    locator: Optional[dict] = None,
    connection: Optional[sqlite3.Connection] = None,
) -> None:
    """Store one attributable text record and keep the FTS index synchronized."""
    owns_connection = connection is None
    db = connection or connect()
    try:
        timestamp = _now()
        db.execute(
            """
            INSERT INTO documents(
                document_id, kind, owner_type, owner_id, paper_id, title,
                text, locator_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                kind=excluded.kind,
                owner_type=excluded.owner_type,
                owner_id=excluded.owner_id,
                paper_id=excluded.paper_id,
                title=excluded.title,
                text=excluded.text,
                locator_json=excluded.locator_json,
                updated_at=excluded.updated_at
            """,
            (
                document_id,
                kind,
                owner_type,
                owner_id,
                paper_id,
                title,
                text,
                json.dumps(locator or {}, default=str),
                timestamp,
            ),
        )
        db.execute("DELETE FROM document_fts WHERE document_id = ?", (document_id,))
        db.execute(
            "INSERT INTO document_fts(document_id, title, text) VALUES(?, ?, ?)",
            (document_id, title, text),
        )
        if owns_connection:
            db.commit()
    finally:
        if owns_connection:
            db.close()


def delete_documents(owner_type: str, owner_id: str, connection: Optional[sqlite3.Connection] = None) -> None:
    owns_connection = connection is None
    db = connection or connect()
    try:
        rows = db.execute(
            "SELECT document_id FROM documents WHERE owner_type = ? AND owner_id = ?",
            (owner_type, owner_id),
        ).fetchall()
        for row in rows:
            db.execute("DELETE FROM document_fts WHERE document_id = ?", (row["document_id"],))
        db.execute(
            "DELETE FROM documents WHERE owner_type = ? AND owner_id = ?",
            (owner_type, owner_id),
        )
        if owns_connection:
            db.commit()
    finally:
        if owns_connection:
            db.close()


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[\w.-]+", query, flags=re.UNICODE)
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def search_documents(
    query: str,
    limit: int = 20,
    kinds: Optional[list[str]] = None,
    paper_ids: Optional[list[str]] = None,
) -> list[dict]:
    """Search attributable records with SQLite FTS5/BM25 ranking."""
    match_query = _fts_query(query)
    if not match_query:
        return []
    db = connect()
    try:
        clauses = ["document_fts MATCH ?"]
        params: list[object] = [match_query]
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            clauses.append(f"d.kind IN ({placeholders})")
            params.extend(kinds)
        if paper_ids:
            placeholders = ",".join("?" for _ in paper_ids)
            clauses.append(f"d.paper_id IN ({placeholders})")
            params.extend(paper_ids)
        params.append(limit)
        rows = db.execute(
            f"""
            SELECT d.*, bm25(document_fts, 0.0, 1.0, 2.0) AS rank
            FROM document_fts
            JOIN documents d USING(document_id)
            WHERE {' AND '.join(clauses)}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["locator"] = json.loads(item.pop("locator_json") or "{}")
            results.append(item)
        return results
    finally:
        db.close()


def has_documents(owner_type: str, owner_id: str, kind: Optional[str] = None) -> bool:
    db = connect()
    try:
        sql = "SELECT 1 FROM documents WHERE owner_type=? AND owner_id=?"
        params: list[object] = [owner_type, owner_id]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        return db.execute(sql + " LIMIT 1", params).fetchone() is not None
    finally:
        db.close()


def documents_for_owner(owner_type: str, owner_id: str, limit: int = 200) -> list[dict]:
    db = connect()
    try:
        rows = db.execute(
            "SELECT * FROM documents WHERE owner_type=? AND owner_id=? ORDER BY document_id LIMIT ?",
            (owner_type, owner_id, limit),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["locator"] = json.loads(item.pop("locator_json") or "{}")
            result.append(item)
        return result
    finally:
        db.close()


def health_snapshot() -> dict:
    """Return inexpensive consistency counts for the UI and deployment checks."""
    db = connect()
    try:
        return {
            "schema_version": int(
                db.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
            ),
            "papers": db.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            "inbox": db.execute("SELECT COUNT(*) FROM triage WHERE status='inbox'").fetchone()[0],
            "documents": db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "fts_documents": db.execute("SELECT COUNT(*) FROM document_fts").fetchone()[0],
            "failed_jobs": db.execute("SELECT COUNT(*) FROM jobs WHERE status='failed'").fetchone()[0],
            "pending_jobs": db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchone()[0],
            "projects": db.execute("SELECT COUNT(*) FROM projects").fetchone()[0],
            "evaluations": db.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0],
        }
    finally:
        db.close()
