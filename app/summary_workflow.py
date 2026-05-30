"""Helpers for robust paper summary generation."""

from app.summarizer import PaperSummarizer


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
