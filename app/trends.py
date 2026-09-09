"""Temporal emerging-theme detection over a paper library."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone


STOPWORDS = {
    "about", "after", "also", "among", "based", "been", "between", "from", "have", "into",
    "more", "paper", "results", "show", "that", "their", "these", "this", "using", "which", "with",
}


def _date(value: object) -> datetime | None:
    text = str(value or "").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)
    except ValueError:
        return None


def _terms(paper: dict) -> set[str]:
    text = f"{paper.get('title', '')} {paper.get('summary', '')} {paper.get('abstract', '')}".lower()
    words = [word for word in re.findall(r"[a-z][a-z0-9-]{3,}", text) if word not in STOPWORDS]
    return set(words + [f"{a} {b}" for a, b in zip(words, words[1:]) if a != b])


def emerging_themes(papers: list[dict], *, recent_days: int = 90, limit: int = 20, now=None) -> list[dict]:
    """Rank terms by recent prevalence relative to the prior comparison window."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=recent_days)
    older_cutoff = cutoff - timedelta(days=recent_days * 3)
    recent, older = Counter(), Counter()
    recent_n = older_n = 0
    for paper in papers:
        published = _date(paper.get("published"))
        if not published or published > now or published < older_cutoff:
            continue
        if published >= cutoff:
            recent.update(_terms(paper)); recent_n += 1
        else:
            older.update(_terms(paper)); older_n += 1
    rows = []
    for term, count in recent.items():
        if count < 2:
            continue
        recent_rate = count / max(recent_n, 1)
        older_rate = older[term] / max(older_n, 1)
        lift = (recent_rate + 1 / max(recent_n, 1)) / (older_rate + 1 / max(older_n, 1))
        score = recent_rate * math.log2(1 + lift)
        rows.append({"theme": term, "recent_papers": count, "previous_papers": older[term], "lift": round(lift, 2), "score": round(score, 4)})
    return sorted(rows, key=lambda row: (row["score"], row["recent_papers"]), reverse=True)[:limit]
