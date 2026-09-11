"""Transparent relevance scoring for the daily paper inbox."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable


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


def _cosine_vectors(left, right) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    numerator = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _preference_documents(
    profile: dict,
    ideas: list[dict],
    contributions: list[dict],
    positive_papers: list[dict],
) -> list[tuple[str, str]]:
    documents: list[tuple[str, str]] = []
    tracking = str(profile.get("tracking_preferences", "")).strip()
    if tracking:
        for line in re.split(r"[\n;]+", tracking):
            if line.strip():
                documents.append(("tracking priorities", line.strip()))
    profile_text = " ".join(
        str(profile.get(key, ""))
        for key in ("research_areas", "methods_and_tools", "bio")
    ).strip()
    if profile_text:
        documents.append(("research profile", profile_text))
    for idea in ideas:
        if idea.get("status", "draft") != "archived":
            text = " ".join(
                str(idea.get(key, "")) for key in ("title", "description", "notes")
            ).strip()
            if text:
                documents.append((f"active idea: {idea.get('title', 'untitled')}", text))
    for contribution in contributions:
        text = " ".join(
            [
                str(contribution.get("name", "")),
                *[str(value) for value in contribution.get("key_claims", [])],
                *[str(value) for value in contribution.get("strong_signals", [])],
            ]
        ).strip()
        if text:
            documents.append((f"your work: {contribution.get('name', contribution.get('id', ''))}", text))
    for paper in positive_papers:
        documents.append((f"positive feedback: {paper.get('title', paper.get('arxiv_id', 'paper'))}", _paper_text(paper)))
    return documents


def score_tracking_papers(
    papers: list[dict],
    *,
    profile: dict,
    ideas: list[dict],
    contributions: list[dict],
    positive_papers: list[dict] | None = None,
    negative_papers: list[dict] | None = None,
    encode: Callable[[str], object],
) -> list[dict]:
    """Score reading interest semantically while retaining explainable provenance."""
    preferences = _preference_documents(
        profile,
        ideas,
        contributions,
        positive_papers or [],
    )
    preference_vectors = [(label, encode(text)) for label, text in preferences]
    negative_vectors = [encode(_paper_text(paper)) for paper in (negative_papers or [])]
    exclusions = [
        normalize
        for line in str(profile.get("tracking_exclusions", "")).splitlines()
        if (normalize := " ".join(_tokens(line)))
    ]

    scored = []
    for paper in papers:
        paper_text = _paper_text(paper)
        normalized_paper = " ".join(_tokens(paper_text))
        if not preference_vectors:
            item = dict(paper)
            item["relevance_score"] = 100.0
            item["relevance_reason"] = "No tracking preferences configured; admitted by default"
            scored.append(item)
            continue
        vector = encode(paper_text)
        similarities = [
            (_cosine_vectors(vector, preference_vector), label)
            for label, preference_vector in preference_vectors
        ]
        best_score, best_label = max(similarities, default=(0.0, "no configured preferences"))
        negative_score = max(
            (_cosine_vectors(vector, negative_vector) for negative_vector in negative_vectors),
            default=0.0,
        )
        adjusted = max(0.0, min(1.0, best_score - 0.25 * negative_score))
        excluded = next(
            (phrase for phrase in exclusions if phrase and phrase in normalized_paper),
            "",
        )
        if excluded:
            adjusted = 0.0
            reason = f"Excluded by tracking preference: {excluded}"
        elif best_score:
            reason = f"Closest match: {best_label} ({best_score:.0%} semantic similarity)"
            if negative_score:
                reason += f"; negative-feedback similarity {negative_score:.0%}"
        else:
            reason = "No tracking preferences are configured"
        item = dict(paper)
        item["relevance_score"] = round(adjusted * 100, 1)
        item["relevance_reason"] = reason
        scored.append(item)
    return sorted(scored, key=lambda item: item["relevance_score"], reverse=True)
