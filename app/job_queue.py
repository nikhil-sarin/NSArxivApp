"""Restart-safe SQLite job queue with one process-local execution thread."""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from app import research_db


Handler = Callable[[dict, dict], dict]
_HANDLERS: dict[str, Handler] = {}
_RUNTIME: dict[str, dict] = {}
_THREAD: threading.Thread | None = None
_LOCK = threading.Lock()
_WAKE = threading.Event()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register(kind: str, handler: Handler) -> None:
    _HANDLERS[kind] = handler


def enqueue(kind: str, payload: dict, *, runtime: dict | None = None) -> str:
    job_id = str(uuid.uuid4())
    with research_db.transaction() as db:
        db.execute(
            "INSERT INTO jobs(job_id, kind, status, payload_json, created_at, updated_at) "
            "VALUES(?, ?, 'queued', ?, ?, ?)",
            (job_id, kind, json.dumps(payload), _now(), _now()),
        )
    if runtime:
        _RUNTIME[job_id] = runtime
    start()
    _WAKE.set()
    return job_id


def recover_interrupted() -> int:
    """Return jobs claimed by a terminated process to the queue."""
    with research_db.transaction() as db:
        cursor = db.execute(
            "UPDATE jobs SET status='queued', error='', updated_at=? WHERE status='running'",
            (_now(),),
        )
        return cursor.rowcount


def retry_failed() -> int:
    """Requeue failed jobs while preserving their durable payloads."""
    with research_db.transaction() as db:
        cursor = db.execute(
            "UPDATE jobs SET status='queued', result_json='{}', error='', updated_at=? "
            "WHERE status IN ('failed', 'completed_with_errors')",
            (_now(),),
        )
    start()
    _WAKE.set()
    return cursor.rowcount


def cancel_queued() -> int:
    """Cancel work that has not yet been claimed."""
    with research_db.transaction() as db:
        cursor = db.execute(
            "UPDATE jobs SET status='cancelled', updated_at=? WHERE status='queued'",
            (_now(),),
        )
        return cursor.rowcount


def _claim() -> dict | None:
    with research_db.transaction() as db:
        row = db.execute(
            "SELECT job_id, kind, payload_json FROM jobs "
            "WHERE status='queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        changed = db.execute(
            "UPDATE jobs SET status='running', updated_at=? "
            "WHERE job_id=? AND status='queued'",
            (_now(), row["job_id"]),
        ).rowcount
        if not changed:
            return None
        return {
            "job_id": row["job_id"],
            "kind": row["kind"],
            "payload": json.loads(row["payload_json"] or "{}"),
        }


def _finish(job_id: str, status: str, *, result: dict | None = None, error: str = "") -> None:
    with research_db.transaction() as db:
        db.execute(
            "UPDATE jobs SET status=?, result_json=?, error=?, updated_at=? WHERE job_id=?",
            (status, json.dumps(result or {}), error[:4000], _now(), job_id),
        )
    _RUNTIME.pop(job_id, None)


def update_progress(job_id: str, result: dict) -> None:
    """Persist intermediate progress without changing the running state."""
    with research_db.transaction() as db:
        db.execute(
            "UPDATE jobs SET result_json=?, updated_at=? WHERE job_id=? AND status='running'",
            (json.dumps(result), _now(), job_id),
        )


def run_pending(*, max_jobs: int | None = None) -> int:
    """Claim and execute queued jobs; exposed for service processes and tests."""
    completed = 0
    while max_jobs is None or completed < max_jobs:
        job = _claim()
        if job is None:
            break
        handler = _HANDLERS.get(job["kind"])
        if handler is None:
            _finish(job["job_id"], "failed", error=f"No handler registered for {job['kind']}")
            completed += 1
            continue
        try:
            runtime = {**_RUNTIME.get(job["job_id"], {}), "job_id": job["job_id"]}
            result = handler(job["payload"], runtime)
        except Exception as exc:
            _finish(job["job_id"], "failed", error=f"{type(exc).__name__}: {exc}")
        else:
            status = "completed_with_errors" if result.get("failed") else "completed"
            _finish(job["job_id"], status, result=result)
        completed += 1
    return completed


def _worker() -> None:
    while True:
        if not run_pending():
            _WAKE.wait(timeout=5)
            _WAKE.clear()
        else:
            time.sleep(0.01)


def start(*, recover: bool = False) -> None:
    global _THREAD
    with _LOCK:
        needs_start = _THREAD is None or not _THREAD.is_alive()
        if needs_start and recover:
            recover_interrupted()
        if needs_start:
            _THREAD = threading.Thread(target=_worker, name="nsarxiv-jobs", daemon=True)
            _THREAD.start()


def list_jobs(limit: int = 20) -> list[dict]:
    db = research_db.connect()
    try:
        rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            item["result"] = json.loads(item.pop("result_json") or "{}")
            result.append(item)
        return result
    finally:
        db.close()
