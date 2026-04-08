"""LLM-based paper summarization module using Ollama."""

import os
import json
import requests
from typing import Optional
from pathlib import Path


class PaperSummarizer:
    """Summarize papers using Ollama with local models."""

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_OLLAMA_HOST = "http://localhost:11434"

    def __init__(self, model: Optional[str] = None, ollama_host: Optional[str] = None):
        """
        Initialize summarizer with Ollama.

        Args:
            model: Ollama model to use (default: llama3.2).
            ollama_host: Ollama server URL (default: http://localhost:11434).
        """
        self.model = model or os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.ollama_host = ollama_host or os.getenv(
            "OLLAMA_HOST", self.DEFAULT_OLLAMA_HOST
        )

    def summarize(self, text: str, max_length: int = 300) -> str:
        """
        Generate a summary of the given text using Ollama.

        Args:
            text: The full text to summarize.
            max_length: Maximum length of the summary.

        Returns:
            Summary string.
        """
        if not text or len(text) < 100:
            return text[:max_length] if text else ""

        return self._summarize_with_ollama(text, max_length)

    def _summarize_with_ollama(self, text: str, max_length: int = 300) -> str:
        """Summarize using Ollama local model."""
        # Truncate text to fit in context window
        max_tokens = 8000
        if len(text) > max_tokens:
            text = text[:max_tokens] + "...[truncated]"

        url = f"{self.ollama_host}/api/generate"

        prompt = f"""You are a research assistant. Summarize this academic paper concisely, focusing on:
1. Main contribution/innovation
2. Key methodology
3. Important findings

Keep it under {max_length} words. Be precise and technical where appropriate.

Paper text:
{text}

Summary:"""

        try:
            response = requests.post(
                url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_length},
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            summary = result.get("response", "").strip()

            # Clean up the summary
            summary = summary.replace("Summary:", "").strip()
            return summary

        except requests.exceptions.ConnectionError:
            print(f"Could not connect to Ollama at {self.ollama_host}")
            print("Make sure Ollama is running: ollama serve")
            return self._fallback_summarize(text, max_length)
        except Exception as e:
            print(f"Ollama summarization failed: {e}. Using fallback.")
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
