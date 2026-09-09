"""Helpers for robust paper summary generation."""

import re

from app.summarizer import PaperSummarizer


def completeness_issues(summary: str) -> list[str]:
    """Detect summaries that are empty, implausibly short, or cut off mid-output."""
    text = (summary or "").strip()
    if not text:
        return ["missing"]
    issues = []
    if len(re.findall(r"\b\w+\b", text)) < 60:
        issues.append("too short")
    if text.endswith(("...", "…", ":", ";", ",")) or not re.search(r"[.!?\])}]$", text):
        issues.append("appears truncated")
    return issues


def summarize_with_fallback(
    summarizer: PaperSummarizer,
    text: str,
    abstract: str = "",
    *,
    max_length: int = 300,
    detailed: bool = False,
) -> str:
    """Summarize full text when available, otherwise fall back to the abstract."""
    full_text = (text or "").strip()
    abstract_text = (abstract or "").strip()

    if full_text:
        summary = summarizer.summarize(full_text, max_length=max_length, detailed=detailed).strip()
        if summary:
            return summary

    if abstract_text:
        summary = summarizer.summarize(abstract_text, max_length=max_length, detailed=detailed).strip()
        if summary:
            return summary
        return abstract_text

    return ""
