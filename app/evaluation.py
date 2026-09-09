"""Persisted model evaluation and evidence-aware routing."""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from app import research_db


DEFAULT_CASES = [
    {"question": "What is the central contribution?", "expected_terms": ["contribution"]},
    {"question": "What evidence supports the result?", "expected_terms": ["evidence"]},
    {"question": "What limitations remain?", "expected_terms": ["limitation"]},
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def score_answer(answer: str, expected_terms: list[str], valid_citations: set[str] | None = None) -> dict:
    """Score lexical task coverage and whether cited labels are from supplied evidence."""
    lower = answer.lower()
    covered = sum(term.lower() in lower for term in expected_terms)
    citations = set(re.findall(r"\[(arXiv:[^\]]+)\]", answer))
    valid = valid_citations or set()
    citation_precision = (len(citations & valid) / len(citations)) if citations else 0.0
    return {
        "task_coverage": covered / max(len(expected_terms), 1),
        "citation_precision": citation_precision,
        "citation_count": len(citations),
    }


def run_suite(
    *,
    suite: str,
    provider: str,
    model: str,
    complete: Callable[[str], str],
    cases: list[dict] | None = None,
) -> dict:
    """Run a small reproducible evaluation suite through an injected model callable."""
    outcomes = []
    for case in cases or DEFAULT_CASES:
        started = time.monotonic()
        answer = complete(case["question"])
        metrics = score_answer(answer, case.get("expected_terms", []), set(case.get("valid_citations", [])))
        outcomes.append({**metrics, "latency_seconds": time.monotonic() - started})
    aggregate = {
        "cases": len(outcomes),
        "task_coverage": sum(row["task_coverage"] for row in outcomes) / max(len(outcomes), 1),
        "citation_precision": sum(row["citation_precision"] for row in outcomes) / max(len(outcomes), 1),
        "mean_latency_seconds": sum(row["latency_seconds"] for row in outcomes) / max(len(outcomes), 1),
        "outcomes": outcomes,
    }
    run_id = str(uuid.uuid4())
    with research_db.transaction() as db:
        db.execute(
            "INSERT INTO evaluation_runs(run_id, suite, provider, model, metrics_json, created_at) VALUES(?, ?, ?, ?, ?, ?)",
            (run_id, suite, provider, model, json.dumps(aggregate), _now()),
        )
    return {"run_id": run_id, **aggregate}


def list_runs(limit: int = 50) -> list[dict]:
    db = research_db.connect()
    try:
        rows = db.execute("SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json"))
            result.append(item)
        return result
    finally:
        db.close()


def recommend_route(runs: list[dict], *, contains_private_data: bool = False) -> dict | None:
    """Choose a measured route, excluding cloud providers for private content."""
    candidates = [run for run in runs if not contains_private_data or run.get("provider") in {"ollama", "local"}]
    if not candidates:
        return None
    def utility(run: dict) -> float:
        metrics = run.get("metrics", {})
        return (
            float(metrics.get("task_coverage", 0))
            + float(metrics.get("citation_precision", 0))
            - min(float(metrics.get("mean_latency_seconds", 0)) / 120, 0.5)
        )
    return max(candidates, key=utility)
