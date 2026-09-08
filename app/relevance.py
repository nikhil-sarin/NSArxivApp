"""Transparent relevance scoring for the daily paper inbox."""

from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
_STOPWORDS = {
    "about", "after", "also", "among", "based", "been", "being", "between",
    "from", "have", "into", "more", "paper", "results", "show", "study",
    "that", "their", "these", "this", "through", "using", "were", "which",
    "with", "within",
}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "") if token.lower() not in _STOPWORDS]


def _paper_text(paper: dict) -> str:
    categories = paper.get("categories", [])
    if isinstance(categories, list):
        categories = " ".join(categories)
    return " ".join(
        [
            str(paper.get("title", "")),
            str(paper.get("summary", "")),
            str(paper.get("abstract", "")),
            str(categories),
        ]
    )


def _cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def build_interest_text(profile: dict, ideas: list[dict]) -> str:
    parts = [
        str(profile.get("research_areas", "")),
        str(profile.get("methods_and_tools", "")),
        str(profile.get("bio", "")),
    ]
    for idea in ideas:
        if idea.get("status", "draft") != "archived":
            parts.extend([str(idea.get("title", "")), str(idea.get("description", "")), str(idea.get("notes", ""))])
    return " ".join(parts)


def score_papers(
    papers: list[dict],
    *,
    interest_text: str,
    positive_papers: list[dict] | None = None,
    negative_papers: list[dict] | None = None,
) -> list[dict]:
    """Return papers with explainable 0-100 relevance scores."""
    interest = Counter(_tokens(interest_text))
    positive = Counter(_tokens(" ".join(_paper_text(paper) for paper in (positive_papers or []))))
    negative = Counter(_tokens(" ".join(_paper_text(paper) for paper in (negative_papers or []))))
    scored = []
    for paper in papers:
        terms = Counter(_tokens(_paper_text(paper)))
        profile_score = _cosine(terms, interest)
        positive_score = _cosine(terms, positive)
        negative_score = _cosine(terms, negative)
        raw = max(0.0, min(1.0, 0.72 * profile_score + 0.33 * positive_score - 0.35 * negative_score))
        overlap = [token for token, _ in terms.most_common() if token in interest or token in positive]
        explanation = "Matched: " + ", ".join(overlap[:6]) if overlap else "No strong profile/project term match yet"
        item = dict(paper)
        item["relevance_score"] = round(raw * 100, 1)
        item["relevance_reason"] = explanation
        scored.append(item)
    return sorted(scored, key=lambda item: item["relevance_score"], reverse=True)
