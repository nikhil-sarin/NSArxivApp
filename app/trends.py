"""Conservative temporal theme detection over paper titles and abstracts."""

from __future__ import annotations

import math
import re
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

from app import research_db


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


def _normalized_vector(value) -> np.ndarray | None:
    vector = np.asarray(value, dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if vector.size and norm else None


def _cluster_label(papers: list[dict], corpus_document_frequency: Counter, corpus_size: int) -> str:
    counts: Counter = Counter()
    for paper in papers:
        _, title_terms = _terms(paper)
        counts.update(title_terms)
    if not counts:
        return "related papers"
    minimum_support = max(2, math.ceil(len(papers) * 0.15))
    candidates = [term for term in counts if counts[term] >= minimum_support]
    if not candidates:
        candidates = list(counts)
    ranked = sorted(
        candidates,
        key=lambda term: (
            counts[term] * math.log((corpus_size + 1) / (corpus_document_frequency[term] + 1))
            * (1.35 if " " in term else 1.0),
            counts[term],
        ),
        reverse=True,
    )
    selected: list[str] = []
    for term in ranked:
        if any(term in existing or existing in term for existing in selected):
            continue
        selected.append(term)
        if len(selected) == 2:
            break
    return " / ".join(selected)


def semantic_theme_clusters(
    papers: list[dict],
    *,
    interest_text: str,
    encode,
    recent_days: int = 90,
    similarity_threshold: float = 0.78,
    relevance_threshold: float = 0.30,
    limit: int = 12,
    now=None,
) -> list[dict]:
    """Cluster recent papers semantically and retain profile-relevant groups."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=recent_days)
    recent_by_id: dict[str, dict] = {}
    for paper in papers:
        published = _date(paper.get("published"))
        if not published or not cutoff <= published <= now:
            continue
        raw_id = str(paper.get("arxiv_id", ""))
        identity = re.sub(r"v\d+$", "", raw_id.lower()) or re.sub(
            r"\W+", " ", str(paper.get("title", "")).lower()
        ).strip()
        if identity and identity not in recent_by_id:
            recent_by_id[identity] = paper
    recent = list(recent_by_id.values())
    if not recent or not interest_text.strip():
        return []
    profile_vector = _normalized_vector(encode(interest_text))
    if profile_vector is None:
        return []

    clusters: list[dict] = []
    for paper in recent:
        text = f"{paper.get('title', '')}. {paper.get('abstract', '') or paper.get('summary', '')}"
        vector = _normalized_vector(encode(text))
        if vector is None or vector.shape != profile_vector.shape:
            continue
        best_index = -1
        best_similarity = -1.0
        for index, cluster in enumerate(clusters):
            similarity = float(np.dot(vector, cluster["centroid"]))
            if similarity > best_similarity:
                best_index, best_similarity = index, similarity
        if best_index >= 0 and best_similarity >= similarity_threshold:
            cluster = clusters[best_index]
            cluster["papers"].append(paper)
            cluster["vectors"].append(vector)
            cluster["centroid"] = _normalized_vector(np.mean(cluster["vectors"], axis=0))
        else:
            clusters.append({"papers": [paper], "vectors": [vector], "centroid": vector})

    corpus_document_frequency: Counter = Counter()
    for paper in recent:
        _, title_terms = _terms(paper)
        corpus_document_frequency.update(title_terms)
    rows = []
    for cluster in clusters:
        if len(cluster["papers"]) < 2:
            continue
        relevance = float(np.dot(cluster["centroid"], profile_vector))
        if relevance < relevance_threshold:
            continue
        label = _cluster_label(cluster["papers"], corpus_document_frequency, len(recent))
        rows.append({
            "theme": label,
            "paper_count": len(cluster["papers"]),
            "relevance": round(relevance, 3),
            "examples": [
                {
                    "arxiv_id": str(paper.get("arxiv_id", "")),
                    "title": str(paper.get("title", paper.get("arxiv_id", "Untitled"))),
                }
                for paper in cluster["papers"][:4]
            ],
        })
    return sorted(rows, key=lambda row: (row["relevance"], row["paper_count"]), reverse=True)[:limit]


def load_feedback() -> dict[str, str]:
    db = research_db.connect()
    try:
        row = db.execute("SELECT value_json FROM settings WHERE key='trend_feedback'").fetchone()
        return json.loads(row[0]) if row else {}
    finally:
        db.close()


def save_feedback(theme: str, state: str) -> None:
    if state not in {"follow", "mute", ""}:
        raise ValueError("trend feedback must be follow, mute, or empty")
    feedback = load_feedback()
    if state:
        feedback[theme] = state
    else:
        feedback.pop(theme, None)
    with research_db.transaction() as db:
        db.execute(
            "INSERT INTO settings(key, value_json, updated_at) VALUES('trend_feedback', ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
            (json.dumps(feedback), datetime.now(timezone.utc).isoformat()),
        )
