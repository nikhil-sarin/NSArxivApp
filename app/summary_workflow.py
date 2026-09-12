"""Helpers for robust paper summary generation."""

import re
from datetime import datetime, timezone

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


def summarize_with_provenance(
    summarizer: PaperSummarizer,
    text: str,
    abstract: str = "",
    *,
    max_length: int = 300,
    detailed: bool = False,
) -> tuple[str, dict]:
    """Generate a summary together with durable source and model provenance."""
    full_text = (text or "").strip()
    abstract_text = (abstract or "").strip()
    source = "full_text" if full_text else "abstract" if abstract_text else "none"
    summary = summarize_with_fallback(
        summarizer,
        full_text,
        abstract_text,
        max_length=max_length,
        detailed=detailed,
    )
    issues = completeness_issues(summary)
    fallback_reason = str(getattr(summarizer, "last_fallback_reason", "") or "")
    status = "failed" if not summary else "fallback" if fallback_reason else "incomplete" if issues else "complete"
    provenance = {
        "status": status,
        "provider": summarizer._active_provider(),
        "model": summarizer._active_model(),
        "input_source": source,
        "detailed": bool(detailed),
        "max_words": int(max_length),
        "quality_issues": issues,
        "fallback_reason": fallback_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_version": "summary-v2",
    }
    return summary, provenance
