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


def _contains_term(normalized_text: str, term: str) -> bool:
    normalized_term = normalize(term)
    return bool(normalized_term) and f" {normalized_term} " in f" {normalized_text} "


def _rule_match(rule: dict, text: str) -> bool:
    normalized_text = normalize(text)
    return all(
        any(_contains_term(normalized_text, term) for term in group)
        for group in rule.get("all", [])
    )


def _matching_signals(contribution: dict, body: str) -> tuple[list[str], list[str], list[str]]:
    paragraphs = [part for part in re.split(r"\n\s*\n", body) if part.strip()]
    matched = lambda signals: [signal for signal in signals if any(_signal_match(signal, p) for p in paragraphs)]
    return (
        matched(contribution.get("strong_signals", [])),
        matched(contribution.get("weak_signals", [])),
        matched(contribution.get("exclusions", [])),
    )


def _matching_rules(contribution: dict, body: str) -> tuple[list[dict], list[dict]]:
    paragraphs = [part for part in re.split(r"\n\s*\n", body) if part.strip()]
    context_windows = [
        "\n\n".join(paragraphs[max(0, index - 1):index + 2])
        for index in range(len(paragraphs))
    ]
    matched = [
        rule for rule in contribution.get("discovery_rules", [])
        if any(_rule_match(rule, window) for window in context_windows)
    ]
    return (
        [rule for rule in matched if rule.get("strength") == "strong"],
        [rule for rule in matched if rule.get("strength") == "weak"],
    )


def _rule_terms(rules: list[dict]) -> list[str]:
    return [term for rule in rules for group in rule.get("all", []) for term in group]


def _evidence_match(text: str, signals: list[str], rules: list[dict]) -> bool:
    return any(_signal_match(signal, text) for signal in signals) or any(
        _rule_match(rule, text) for rule in rules
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
            normalized_value = normalize(value)
            if value and normalized_value in normalized_all:
                location = "references" if normalized_value in normalized_refs else "document"
                matched_identifiers.append({"kind": kind, "value": value, "location": location})
        title = str(citation.get("title", "")).strip()
        normalized_title = normalize(title)
        if title and normalized_title in normalized_all:
            location = "references" if normalized_title in normalized_refs else "document"
            matched_identifiers.append({"kind": "title", "value": title, "location": location})
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


def _passages_from_text(text: str, signals: list[str], rules: list[dict]) -> list[dict]:
    spans = _section_spans(text)
    passages = []
    matches = list(re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text[:_references_start(text)]))
    seen_offsets: set[tuple[int, int]] = set()

    def add_passage(
        raw_quote: str,
        raw_start: int,
        evidence_terms: list[str],
        *,
        preferred_anchor_terms: list[str] | None = None,
    ) -> None:
        quote, relative_start = _bounded_matching_quote(
            raw_quote,
            evidence_terms,
            preferred_anchor_terms=preferred_anchor_terms,
        )
        absolute_start = raw_start + relative_start
        offset_key = (absolute_start, absolute_start + len(quote))
        if offset_key in seen_offsets:
            return
        seen_offsets.add(offset_key)
        passages.append({
            "locator": f"{_locator_for_offset(spans, absolute_start)} / characters {absolute_start}-{absolute_start + len(quote)}",
            "quote": quote,
            "start_offset": absolute_start,
            "end_offset": absolute_start + len(quote),
        })

    for rule in rules:
        candidates = []
        for index, match in enumerate(matches):
            context = matches[max(0, index - 1):index + 2]
            raw_start = context[0].start()
            raw_quote = text[raw_start:context[-1].end()].strip()
            if not raw_quote or not _rule_match(rule, raw_quote):
                continue
            preferred_score = sum(
                _contains_term(normalize(raw_quote), term)
                for term in rule.get("preferred_evidence", [])
            )
            candidates.append((preferred_score, -raw_start, raw_quote, raw_start))
        if candidates:
            _, _, raw_quote, raw_start = max(candidates, key=lambda item: (item[0], item[1]))
            normalized_quote = normalize(raw_quote)
            primary_terms = [
                term for term in rule.get("all", [[]])[0]
                if _contains_term(normalized_quote, term)
            ]
            add_passage(
                raw_quote,
                raw_start,
                _rule_terms([rule]),
                preferred_anchor_terms=primary_terms,
            )
        if len(passages) >= 5:
            return passages

    for match in matches:
        raw_quote = match.group(0).strip()
        if not raw_quote or not any(_signal_match(signal, raw_quote) for signal in signals):
            continue
        add_passage(raw_quote, match.start(), signals)
        if len(passages) >= 5:
            break
    return passages


def _passages_from_chunks(chunks: Iterable[dict], signals: list[str], rules: list[dict]) -> list[dict]:
    passages = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        if not _evidence_match(text, signals, rules):
            continue
        locator = chunk.get("locator", {})
        label = (
            f"page {locator['page']} / chunk {locator.get('chunk', 1)}"
            if locator.get("page") else
            f"{locator.get('section') or locator.get('source') or 'indexed text'} / chunk {locator.get('chunk', 1)}"
        )
        quote, _ = _bounded_matching_quote(text, signals + _rule_terms(rules))
        passages.append({"locator": label, "quote": quote, "start_offset": None, "end_offset": None})
        if len(passages) >= 5:
            break
    return passages


def _bounded_matching_quote(
    text: str,
    signals: list[str],
    *,
    preferred_anchor_terms: list[str] | None = None,
) -> tuple[str, int]:
    """Return an exact bounded substring centered near a matched signal term."""
    if len(text) <= MAX_QUOTE_CHARS:
        return text, 0
    anchor_source = preferred_anchor_terms or signals
    minimum_word_length = 2 if preferred_anchor_terms else 5
    normalized_terms = [
        word for signal in anchor_source for word in normalize(signal).split()
        if len(word) >= minimum_word_length and word not in SIGNAL_WORDS_TO_IGNORE
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
    strong_rules, weak_rules = _matching_rules(contribution, body)
    all_signals = strong + weak
    all_rules = strong_rules + weak_rules
    passages = _passages_from_chunks(corpus_chunks or [], all_signals, all_rules)
    if not passages:
        passages = _passages_from_text(paper_text, all_signals, all_rules)
    reference_check = _reference_check(paper_text, contribution, owner_name_variants)
    candidate = bool(
        passages and not exclusions and not reference_check["canonical_citation_found"]
        and (strong or strong_rules or len(weak) + len(weak_rules) >= 2)
    )
    return {
        "paper_id": paper_id,
        "contribution_id": contribution["id"],
        "matched_signals": strong + weak,
        "matched_strong_signals": strong,
        "matched_weak_signals": weak,
        "matched_rules": [rule["name"] for rule in all_rules],
        "matched_strong_rules": [rule["name"] for rule in strong_rules],
        "matched_exclusions": exclusions,
        "candidate": candidate,
        "passages": passages,
        "reference_check": reference_check,
    }
