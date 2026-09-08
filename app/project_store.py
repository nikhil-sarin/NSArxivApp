"""Persistent projects, evidence links, and weekly source snapshots."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app import idea_store, research_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_project(
    *,
    title: str,
    description: str = "",
    kind: str = "research",
    status: str = "active",
    methods: str = "",
    repository_paths: Optional[list[str]] = None,
    notion_url: str = "",
    notion_export_path: str = "",
    project_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    project_id = project_id or str(uuid.uuid4())[:12]
    timestamp = _now()
    with research_db.transaction() as db:
        db.execute(
            """
            INSERT INTO projects(
                project_id, title, kind, status, description, methods,
                repository_paths_json, notion_url, notion_export_path,
                data_json, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                title=excluded.title,
                kind=excluded.kind,
                status=excluded.status,
                description=excluded.description,
                methods=excluded.methods,
                repository_paths_json=excluded.repository_paths_json,
                notion_url=excluded.notion_url,
                notion_export_path=excluded.notion_export_path,
                data_json=excluded.data_json,
                updated_at=excluded.updated_at
            """,
            (
                project_id,
                title.strip(),
                kind,
                status,
                description.strip(),
                methods.strip(),
                json.dumps(repository_paths or []),
                notion_url.strip(),
                notion_export_path.strip(),
                json.dumps(extra or {}, default=str),
                timestamp,
                timestamp,
            ),
        )
    research_db.upsert_document(
        f"project:{project_id}:description",
        kind="project",
        owner_type="project",
        owner_id=project_id,
        title=title,
        text="\n".join(value for value in [description, methods] if value),
        locator={"project_id": project_id},
    )
    return project_id


def _row_to_project(row) -> dict:
    project = dict(row)
    project["repository_paths"] = json.loads(project.pop("repository_paths_json") or "[]")
    project.update(json.loads(project.pop("data_json") or "{}"))
    return project


def load_projects(status: Optional[str] = None) -> list[dict]:
    migrate_ideas()
    db = research_db.connect()
    try:
        if status:
            rows = db.execute("SELECT * FROM projects WHERE status=? ORDER BY updated_at DESC", (status,)).fetchall()
        else:
            rows = db.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [_row_to_project(row) for row in rows]
    finally:
        db.close()


def get_project(project_id: str) -> Optional[dict]:
    db = research_db.connect()
    try:
        row = db.execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
        return _row_to_project(row) if row else None
    finally:
        db.close()


def migrate_ideas() -> int:
    """Import legacy paper/grant ideas as projects without changing ideas.json."""
    imported = 0
    db = research_db.connect()
    try:
        existing = {row[0] for row in db.execute("SELECT project_id FROM projects").fetchall()}
    finally:
        db.close()
    for idea_type in ("paper", "grant"):
        for idea in idea_store.load_ideas(idea_type):
            project_id = f"idea-{idea_type}-{idea['id']}"
            if project_id in existing:
                continue
            save_project(
                project_id=project_id,
                title=idea.get("title", "Untitled idea"),
                description=idea.get("description", ""),
                kind=idea_type,
                status="archived" if idea.get("status") == "archived" else "active",
                extra={"legacy_idea_id": idea["id"], "legacy_idea_type": idea_type},
            )
            for paper_id in idea.get("linked_papers", []):
                link_paper(project_id, paper_id)
            existing.add(project_id)
            imported += 1
    return imported


def link_paper(project_id: str, paper_id: str, relationship: str = "relevant") -> None:
    with research_db.transaction() as db:
        db.execute(
            """
            INSERT INTO project_papers(project_id, paper_id, relationship, created_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(project_id, paper_id) DO UPDATE SET relationship=excluded.relationship
            """,
            (project_id, paper_id, relationship, _now()),
        )


def linked_paper_ids(project_id: str) -> list[str]:
    db = research_db.connect()
    try:
        return [
            row[0]
            for row in db.execute(
                "SELECT paper_id FROM project_papers WHERE project_id=? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        ]
    finally:
        db.close()


def set_linked_papers(project_id: str, paper_ids: list[str]) -> None:
    with research_db.transaction() as db:
        db.execute("DELETE FROM project_papers WHERE project_id=?", (project_id,))
        for paper_id in dict.fromkeys(paper_ids):
            db.execute(
                "INSERT INTO project_papers(project_id, paper_id, relationship, created_at) VALUES(?, ?, 'relevant', ?)",
                (project_id, paper_id, _now()),
            )


def _store_snapshot(project_id: str, source: str, content: str) -> bool:
    if not content.strip():
        return False
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    db = research_db.connect()
    try:
        exists = db.execute(
            "SELECT 1 FROM project_snapshots WHERE project_id=? AND source=? AND content_hash=?",
            (project_id, source, digest),
        ).fetchone()
    finally:
        db.close()
    if exists:
        return False
    snapshot_id = str(uuid.uuid4())
    timestamp = _now()
    with research_db.transaction() as db:
        db.execute(
            "INSERT INTO project_snapshots VALUES(?, ?, ?, ?, ?, ?)",
            (snapshot_id, project_id, source, content, digest, timestamp),
        )
    research_db.upsert_document(
        f"snapshot:{snapshot_id}",
        kind="project_snapshot",
        owner_type="project",
        owner_id=project_id,
        title=f"{source} snapshot",
        text=content,
        locator={"project_id": project_id, "source": source, "created_at": timestamp},
    )
    return True


def capture_project_sources(project: dict, since: str = "7 days ago") -> dict:
    """Capture read-only Git and exported-Notion snapshots for a project."""
    captured = {"git": 0, "notion_export": 0, "errors": []}
    for raw_path in project.get("repository_paths", []):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir() or not (path / ".git").exists():
            captured["errors"].append(f"Not a Git repository: {path}")
            continue
        result = subprocess.run(
            ["git", "-C", str(path), "log", f"--since={since}", "--date=short", "--pretty=format:%h %ad %s", "--stat"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            captured["errors"].append(result.stderr.strip() or f"Could not read {path}")
            continue
        if _store_snapshot(project["project_id"], f"git:{path}", result.stdout):
            captured["git"] += 1

    export_path = project.get("notion_export_path", "").strip()
    if export_path:
        path = Path(export_path).expanduser().resolve()
        if path.is_file():
            if _store_snapshot(project["project_id"], f"notion-export:{path}", path.read_text(errors="replace")):
                captured["notion_export"] += 1
        else:
            captured["errors"].append(f"Notion export not found: {path}")
    return captured


def latest_snapshots(project_id: str, limit: int = 10) -> list[dict]:
    db = research_db.connect()
    try:
        return [
            dict(row)
            for row in db.execute(
                "SELECT * FROM project_snapshots WHERE project_id=? ORDER BY created_at DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        ]
    finally:
        db.close()
