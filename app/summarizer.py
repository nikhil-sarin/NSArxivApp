"""LLM-based paper summarization module."""

import os
from typing import Optional
from pathlib import Path
import requests


class PaperSummarizer:
    """Summarize papers using LLM."""

    def __init__(self, use_api: bool = False, api_key: Optional[str] = None):
        """
        Initialize summarizer.

        Args:
            use_api: If True, use OpenAI API. If False, use a local model or fallback.
            api_key: OpenAI API key if using API.
        """
        self.use_api = use_api
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def summarize(self, text: str, max_length: int = 300) -> str:
        """
        Generate a summary of the given text.

        Args:
            text: The full text to summarize.
            max_length: Maximum length of the summary.

        Returns:
            Summary string.
        """
        if not text or len(text) < 100:
            return text[:max_length] if text else ""

        # Use API if configured
        if self.use_api and self.api_key:
            return self._summarize_with_api(text, max_length)

        # Fallback: Use the abstract/summary if available
        if "\n\n" in text:
            first_section = text.split("\n\n")[0][:max_length]
            return first_section.strip()

        # Last resort: truncate
        return text[:max_length] + "..." if len(text) > max_length else text

    def _summarize_with_api(self, text: str, max_length: int = 300) -> str:
        """Summarize using OpenAI API."""
        # Truncate text to fit in token limit
        max_tokens = 4000
        if len(text) > max_tokens:
            text = text[:max_tokens] + "..."

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a research assistant. Provide concise, accurate summaries of academic papers. Focus on the main contribution, methodology, and key findings.",
                        },
                        {
                            "role": "user",
                            "content": f"Summarize this paper in {max_length} words or less:\n\n{text}",
                        },
                    ],
                    "max_tokens": max_length,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"API summarization failed: {e}. Using fallback.")
            return self._fallback_summarize(text, max_length)

    def _fallback_summarize(self, text: str, max_length: int = 300) -> str:
        """Fallback summarization without API."""
        lines = text.split("\n")
        summary_parts = []

        # Look for abstract-like sections
        for line in lines:
            line_lower = line.lower()
            if any(
                keyword in line_lower
                for keyword in ["abstract", "summary", "introduction", "conclusion"]
            ):
                summary_parts.append(line)

        if summary_parts:
            summary = " ".join(summary_parts)[:max_length]
            return summary.strip()

        # Fall back to first paragraph
        for line in lines:
            if line.strip() and len(line.strip()) > 50:
                return line.strip()[:max_length]

        return text[:max_length] if text else ""
