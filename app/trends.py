"""Conservative temporal theme detection over paper titles and abstracts."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone


STOPWORDS = {
    "about", "after", "also", "among", "approach", "based", "been", "between", "data",
    "from", "have", "into", "method", "methods", "model", "models", "more", "paper",
    "present", "results", "show", "study", "that", "their", "these", "this", "using",
    "which", "with", "work", "analysis", "observations", "observed", "properties",
    "across", "consistent", "first", "including", "investigate", "provide", "remains",
    "source", "sources", "through", "while", "within",
    "cosmic", "evidence", "exploring", "gamma", "implications", "light",
}


def _date(value: object) -> datetime | None:
    text = str(value or "").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)
    except ValueError:
        return None


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9-]{3,}", text.lower())


def _valid(word: str) -> bool:
    return word not in STOPWORDS


def _terms(paper: dict) -> tuple[set[str], set[str]]:
    """Return abstract terms and stricter title-supported terms; ignore generated summaries."""
    title_words = _words(str(paper.get("title", "")))
    all_words = _words(f"{paper.get('title', '')} {paper.get('abstract', '')}")

    def terms(words: list[str]) -> set[str]:
        unigrams = {word for word in words if _valid(word) and len(word) >= 5}
        bigrams = {
            f"{first} {second}" for first, second in zip(words, words[1:])
            if first != second and _valid(first) and _valid(second)
            and len(first) >= 4 and len(second) >= 4
        }
        return unigrams | bigrams

    return terms(all_words), terms(title_words)


def emerging_themes(papers: list[dict], *, recent_days: int = 90, limit: int = 20, now=None) -> list[dict]:
    """Return supported themes that have grown against a three-window baseline."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=recent_days)
    older_cutoff = cutoff - timedelta(days=recent_days * 3)
    recent, recent_context, older = Counter(), Counter(), Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    recent_n = older_n = 0
    seen_papers: set[str] = set()
    for paper in papers:
        arxiv_id = re.sub(r"v\d+$", "", str(paper.get("arxiv_id", "")).lower())
        identity = arxiv_id or re.sub(r"\W+", " ", str(paper.get("title", "")).lower()).strip()
        if not identity or identity in seen_papers:
            continue
        seen_papers.add(identity)
        published = _date(paper.get("published"))
        if not published or published > now or published < older_cutoff:
            continue
        context_terms, title_terms = _terms(paper)
        if published >= cutoff:
            recent.update(title_terms)
            recent_context.update(context_terms)
            recent_n += 1
            for term in title_terms:
                if len(examples[term]) < 3:
                    examples[term].append(str(paper.get("title", paper.get("arxiv_id", "Untitled"))))
        else:
            older.update(title_terms)
            older_n += 1

    minimum_support = max(3, math.ceil(recent_n * 0.01))
    rows = []
    for term, count in recent.items():
        if count < minimum_support:
            continue
        recent_rate = (count + 0.5) / (recent_n + 1)
        older_rate = (older[term] + 0.5) / (older_n + 1)
        lift = recent_rate / older_rate
        if lift < 1.5:
            continue
        score = recent_rate * math.log2(lift) * math.log1p(count)
        rows.append({
            "theme": term,
            "recent_papers": count,
            "previous_papers": older[term],
            "title_support": count,
            "context_support": recent_context[term],
            "lift": round(lift, 2),
            "score": round(score, 4),
            "examples": examples[term],
            "basis": f"{count}/{recent_n} recent; {older[term]}/{older_n} prior",
        })
    return sorted(
        rows,
        key=lambda row: (" " in row["theme"], row["score"], row["recent_papers"]),
        reverse=True,
    )[:limit]
