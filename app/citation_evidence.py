"""Deterministic citation/reference checks and bounded evidence extraction."""

from __future__ import annotations

import re
from typing import Iterable

from app.contribution_catalogue import normalize


MAX_QUOTE_CHARS = 700
SIGNAL_WORDS_TO_IGNORE = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "using", "with"}


def _references_start(text: str) -> int:
    matches = list(re.finditer(r"(?im)^\s*(references|bibliography)\s*$", text))
    return matches[-1].start() if matches else len(text)


def _signal_match(signal: str, text: str) -> bool:
    required = [word for word in normalize(signal).split() if word not in SIGNAL_WORDS_TO_IGNORE]
    words = set(normalize(text).split())
    if not required:
        return False
    return sum(word in words for word in required) / len(required) >= 0.75


def _matching_signals(contribution: dict, body: str) -> tuple[list[str], list[str], list[str]]:
    paragraphs = [part for part in re.split(r"\n\s*\n", body) if part.strip()]
    matched = lambda signals: [signal for signal in signals if any(_signal_match(signal, p) for p in paragraphs)]
    return (
        matched(contribution.get("strong_signals", [])),
        matched(contribution.get("weak_signals", [])),
        matched(contribution.get("exclusions", [])),
    )


def _reference_check(text: str, contribution: dict, owner_variants: list[str]) -> dict:
    ref_start = _references_start(text)
    references = text[ref_start:]
    normalized_refs = normalize(references)
    normalized_all = normalize(text)
    matched_identifiers = []
    for citation in contribution.get("canonical_citations", []):
        for kind in ("doi", "arxiv_id", "bibcode"):
            value = str(citation.get(kind, "")).strip()
            if value and normalize(value) in normalized_refs:
                matched_identifiers.append({"kind": kind, "value": value, "location": "references"})
        title = str(citation.get("title", "")).strip()
        if title and normalize(title) in normalized_refs:
            matched_identifiers.append({"kind": "title", "value": title, "location": "references"})
    aliases = contribution.get("aliases", [])
    return {
        "canonical_citation_found": bool(matched_identifiers),
        "owner_name_found": any(normalize(name) in normalized_all for name in owner_variants if normalize(name)),
        "contribution_alias_found": any(alias in normalized_all for alias in aliases),
        "matched_identifiers": matched_identifiers,
        "references_locator": f"characters {ref_start}-{len(text)}" if ref_start < len(text) else "references section not detected",
    }


def _section_spans(text: str) -> list[tuple[int, str]]:
    headings = [(match.start(), match.group(1).strip()) for match in re.finditer(
        r"(?im)^\s*(?:\d+(?:\.\d+)*\s+)?(abstract|introduction|methods?|methodology|analysis|results?|discussion|conclusions?|references|bibliography)\s*$",
        text,
    )]
    return headings or [(0, "full text")]


def _locator_for_offset(spans: list[tuple[int, str]], offset: int) -> str:
    return next((name for start, name in reversed(spans) if start <= offset), "full text")


def _passages_from_text(text: str, signals: list[str]) -> list[dict]:
    spans = _section_spans(text)
    passages = []
    for match in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text[:_references_start(text)]):
        raw_quote = match.group(0).strip()
        if not raw_quote or not any(_signal_match(signal, raw_quote) for signal in signals):
            continue
        quote, relative_start = _bounded_matching_quote(raw_quote, signals)
        absolute_start = match.start() + relative_start
        passages.append({
            "locator": f"{_locator_for_offset(spans, absolute_start)} / characters {absolute_start}-{absolute_start + len(quote)}",
            "quote": quote,
            "start_offset": absolute_start,
            "end_offset": absolute_start + len(quote),
        })
        if len(passages) >= 5:
            break
    return passages


def _passages_from_chunks(chunks: Iterable[dict], signals: list[str]) -> list[dict]:
    passages = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        if not any(_signal_match(signal, text) for signal in signals):
            continue
        locator = chunk.get("locator", {})
        label = (
            f"page {locator['page']} / chunk {locator.get('chunk', 1)}"
            if locator.get("page") else
            f"{locator.get('section') or locator.get('source') or 'indexed text'} / chunk {locator.get('chunk', 1)}"
        )
        quote, _ = _bounded_matching_quote(text, signals)
        passages.append({"locator": label, "quote": quote, "start_offset": None, "end_offset": None})
        if len(passages) >= 5:
            break
    return passages


def _bounded_matching_quote(text: str, signals: list[str]) -> tuple[str, int]:
    """Return an exact bounded substring centered near a matched signal term."""
    if len(text) <= MAX_QUOTE_CHARS:
        return text, 0
    normalized_terms = [
        word for signal in signals for word in normalize(signal).split()
        if len(word) >= 5 and word not in SIGNAL_WORDS_TO_IGNORE
    ]
    lower = text.lower()
    positions = [lower.find(term) for term in normalized_terms if lower.find(term) >= 0]
    anchor = min(positions) if positions else 0
    start = max(0, min(anchor - 180, len(text) - MAX_QUOTE_CHARS))
    return text[start:start + MAX_QUOTE_CHARS], start


def build_evidence_packet(
    paper_id: str,
    paper_text: str,
    contribution: dict,
    *,
    owner_name_variants: list[str],
    corpus_chunks: Iterable[dict] | None = None,
) -> dict:
    """Create an inspectable packet before any model judgement is allowed."""
    body = paper_text[:_references_start(paper_text)]
    strong, weak, exclusions = _matching_signals(contribution, body)
    all_signals = strong + weak
    passages = _passages_from_chunks(corpus_chunks or [], all_signals)
    if not passages:
        passages = _passages_from_text(paper_text, all_signals)
    reference_check = _reference_check(paper_text, contribution, owner_name_variants)
    candidate = bool(
        passages and not exclusions and not reference_check["canonical_citation_found"]
        and (strong or len(weak) >= 2)
    )
    return {
        "paper_id": paper_id,
        "contribution_id": contribution["id"],
        "matched_signals": strong + weak,
        "matched_strong_signals": strong,
        "matched_weak_signals": weak,
        "matched_exclusions": exclusions,
        "candidate": candidate,
        "passages": passages,
        "reference_check": reference_check,
    }
