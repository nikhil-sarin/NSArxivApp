"""Self-hosted maintenance commands for diagnostics and SQLite backups."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from app import research_db


def backup_database(destination: Path | None = None, *, keep: int = 14) -> Path:
    source_path = research_db.DB_PATH
    if not source_path.exists():
        raise FileNotFoundError(f"Database does not exist: {source_path}")
    backup_dir = source_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = destination or backup_dir / f"{source_path.stem}-{stamp}.db"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(source_path)
    target = sqlite3.connect(destination)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    backups = sorted(backup_dir.glob(f"{source_path.stem}-*.db"), key=lambda path: path.stat().st_mtime)
    for expired in backups[:-max(keep, 1)]:
        expired.unlink()
    return destination


def restore_database(source_path: Path) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"Backup does not exist: {source_path}")
    if research_db.DB_PATH.exists():
        backup_database()
    source = sqlite3.connect(source_path)
    target = sqlite3.connect(research_db.DB_PATH)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def doctor() -> dict:
    checks: dict[str, dict] = {}
    checks["python"] = {
        "ok": sys.version_info >= (3, 10),
        "detail": sys.version.split()[0],
    }
    data_dir = research_db.DB_PATH.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    checks["data_directory"] = {
        "ok": os.access(data_dir, os.W_OK),
        "detail": str(data_dir.resolve()),
    }
    try:
        health = research_db.health_snapshot()
        checks["database"] = {
            "ok": health["schema_version"] == research_db.SCHEMA_VERSION,
            "detail": health,
        }
        checks["full_text_index"] = {
            "ok": health["documents"] == health["fts_documents"],
            "detail": {"documents": health["documents"], "fts_documents": health["fts_documents"]},
        }
    except Exception as exc:
        checks["database"] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    provider = os.getenv("SUMMARIZER_PROVIDER", "ollama").lower()
    required_key = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider)
    checks["summarizer_configuration"] = {
        "ok": not required_key or bool(os.getenv(required_key)),
        "detail": {
            "provider": provider,
            "model": os.getenv("LLM_MODEL") or os.getenv(f"{provider.upper()}_MODEL", "default"),
            "credential": "configured" if required_key and os.getenv(required_key) else "not required" if not required_key else "missing",
        },
    }
    orchestrator = os.getenv("LOCAL_ORCHESTRATOR_URL", "").strip()
    checks["local_orchestrator"] = {
        "ok": True,
        "detail": "optional; configured" if orchestrator else "optional; not configured",
    }
    return {"ok": all(check["ok"] for check in checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    backup_parser = subparsers.add_parser("backup")
    backup_parser.add_argument("--destination", type=Path)
    backup_parser.add_argument("--keep", type=int, default=14)
    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--from", dest="source", type=Path, required=True)
    restore_parser.add_argument("--yes", action="store_true", help="Confirm replacement of the active database")
    subparsers.add_parser("doctor")
    args = parser.parse_args()
    if args.command == "backup":
        print(backup_database(args.destination, keep=args.keep))
        return 0
    if args.command == "restore":
        if not args.yes:
            parser.error("restore requires --yes")
        restore_database(args.source)
        print(research_db.DB_PATH)
        return 0
    result = doctor()
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
