"""LLM-based paper summarization with multi-provider support.

Supported providers (set via SUMMARIZER_PROVIDER env var):
  ollama    - Local or remote Ollama server (default)
  gemini    - Google Gemini API
  anthropic - Anthropic Claude API
  openai    - OpenAI API
"""

import os
import re

import requests
from typing import Optional


def _gemini_post(url: str, api_key: str, payload: dict, timeout: int = 120) -> dict:
    """POST to Gemini API."""
    response = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _build_prompt(text: str, max_length: int, detailed: bool) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given mode."""
    if detailed:
        system = "You are a research assistant that writes precise, technical summaries of academic papers."
        user = (
            f"Write a detailed technical summary of this paper covering:\n"
            f"1. Main contribution and novelty\n"
            f"2. Methodology and approach\n"
            f"3. Key results and findings\n"
            f"4. Limitations or open questions\n"
            f"5. Relevance and potential impact\n\n"
            f"Aim for {max_length} words.\n\nPaper text:\n{text}"
        )
    else:
        system = "You are a research assistant that writes concise, technical summaries of academic papers."
        user = (
            f"Summarize this paper concisely (under {max_length} words), covering:\n"
            f"1. Main contribution/innovation\n"
            f"2. Key methodology\n"
            f"3. Important findings\n\nPaper text:\n{text}"
        )
    return system, user


class PaperSummarizer:
    """Summarize papers using a configurable LLM provider."""

    CONTEXT_LIMIT = 30000  # chars — safe for all providers
    CHUNK_SUMMARY_WORDS = 220
    QUICK_SUMMARY_HEAD_CHARS = 12000
    QUICK_SUMMARY_TAIL_CHARS = 6000
    OLLAMA_CONCISE_TIMEOUT_SECONDS = 30
    OLLAMA_DETAILED_TIMEOUT_SECONDS = 300

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or os.getenv("SUMMARIZER_PROVIDER", "ollama")).lower()
        self.model = model or os.getenv("LLM_MODEL", "")
        # Provider-specific defaults
        self._defaults = {
            "ollama":    {"model": os.getenv("OLLAMA_MODEL", "llama3.1:latest"),    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434")},
            "gemini":    {"model": "gemini-2.0-flash"},
            "anthropic": {"model": "claude-3-5-haiku-20241022"},
            "openai":    {"model": "gpt-4o-mini"},
        }
        if not self.model:
            self.model = self._defaults.get(self.provider, {}).get("model", "")

    def _active_provider(self) -> str:
        """Read provider from env each call so .env changes take effect without restart."""
        return os.getenv("SUMMARIZER_PROVIDER", self.provider).lower()

    def _active_model(self) -> str:
        override = os.getenv("LLM_MODEL", "")
        if override:
            return override
        provider = self._active_provider()
        return self._defaults.get(provider, {}).get("model", self.model)

    def summarize(self, text: str, max_length: int = 300, detailed: bool = False) -> str:
        if not text or len(text) < 100:
            return text[:max_length] if text else ""
        if len(text) > self.CONTEXT_LIMIT:
            if not detailed:
                focused_text = self._build_quick_summary_text(text)
                return self._summarize_once(focused_text, max_length=max_length, detailed=False)
            return self._summarize_long_text(text, max_length=max_length, detailed=detailed)

        return self._summarize_once(text, max_length=max_length, detailed=detailed)

    def complete(
        self,
        system: str,
        user: str,
        max_length: int = 1024,
        *,
        detailed: bool = True,
    ) -> str:
        """Run a single provider completion without fallback handling."""
        provider = self._active_provider()
        self.model = self._active_model()
        dispatch = {
            "ollama": self._call_ollama,
            "gemini": self._call_gemini,
            "anthropic": self._call_anthropic,
            "openai": self._call_openai,
        }
        fn = dispatch.get(provider)
        if fn is None:
            raise ValueError(f"Unknown provider: {provider!r}. Choose from: {list(dispatch)}")
        return fn(system, user, max_length, detailed)

    def _summarize_once(self, text: str, max_length: int, detailed: bool) -> str:
        system_prompt, user_prompt = _build_prompt(text, max_length, detailed)
        try:
            return self.complete(system_prompt, user_prompt, max_length=max_length, detailed=detailed)
        except Exception as e:
            provider = self._active_provider()
            print(f"[summarizer] {provider} failed: {e}. Using fallback.")
            return self._fallback_summarize(text, max_length)

    def _split_text(self, text: str) -> list[str]:
        """Split long paper text into prompt-sized chunks, preferring paragraph boundaries."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return [text[: self.CONTEXT_LIMIT]]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for paragraph in paragraphs:
            if len(paragraph) > self.CONTEXT_LIMIT:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                for start in range(0, len(paragraph), self.CONTEXT_LIMIT):
                    chunks.append(paragraph[start:start + self.CONTEXT_LIMIT])
                continue

            addition = len(paragraph) + (2 if current else 0)
            if current and current_len + addition > self.CONTEXT_LIMIT:
                chunks.append("\n\n".join(current))
                current = [paragraph]
                current_len = len(paragraph)
            else:
                current.append(paragraph)
                current_len += addition

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _build_quick_summary_text(self, text: str) -> str:
        """Build a focused excerpt for concise summaries of long papers."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return text[: self.CONTEXT_LIMIT]

        selected: list[str] = []
        selected_set: set[str] = set()

        def add_paragraphs(candidates: list[str], budget: int) -> None:
            used = 0
            for paragraph in candidates:
                if paragraph in selected_set:
                    continue
                addition = len(paragraph) + (2 if selected else 0)
                if used and used + addition > budget:
                    break
                selected.append(paragraph)
                selected_set.add(paragraph)
                used += addition

        add_paragraphs(paragraphs, self.QUICK_SUMMARY_HEAD_CHARS)

        tail_keywords = (
            "conclusion",
            "conclusions",
            "discussion",
            "results",
            "summary",
            "we find",
            "we show",
            "we present",
            "in this paper",
        )
        tail_candidates = [
            paragraph
            for paragraph in paragraphs[-20:]
            if any(keyword in paragraph.lower() for keyword in tail_keywords)
        ]
        if not tail_candidates:
            tail_candidates = paragraphs[-6:]

        add_paragraphs(tail_candidates, self.QUICK_SUMMARY_TAIL_CHARS)

        focused_text = "\n\n".join(selected)
        if len(focused_text) > self.CONTEXT_LIMIT:
            return focused_text[: self.CONTEXT_LIMIT]
        return focused_text

    def _summarize_section(self, chunk: str, chunk_index: int, total_chunks: int, max_length: int, detailed: bool) -> str:
        """Summarize one chunk of a long paper."""
        system = "You are a research assistant summarizing one contiguous section of an academic paper."
        if detailed:
            user = (
                f"This is section {chunk_index} of {total_chunks} from a longer paper.\n"
                f"Summarize the important content from this section only, focusing on methods, results, assumptions, and limitations.\n"
                f"Keep it under {max_length} words.\n\nSection text:\n{chunk}"
            )
        else:
            user = (
                f"This is section {chunk_index} of {total_chunks} from a longer paper.\n"
                f"Summarize the main contribution, method, and findings that appear in this section only.\n"
                f"Keep it under {max_length} words.\n\nSection text:\n{chunk}"
            )

        try:
            return self.complete(system, user, max_length=max_length, detailed=detailed)
        except Exception as e:
            provider = self._active_provider()
            print(f"[summarizer] {provider} failed on section {chunk_index}/{total_chunks}: {e}. Using fallback.")
            return self._fallback_summarize(chunk, max_length)

    def _summarize_long_text(self, text: str, max_length: int, detailed: bool) -> str:
        """Summarize an entire paper by map-reducing chunk summaries."""
        chunks = self._split_text(text)
        if len(chunks) == 1:
            return self._summarize_once(chunks[0], max_length=max_length, detailed=detailed)

        chunk_word_budget = max(max_length, self.CHUNK_SUMMARY_WORDS if detailed else max_length)
        chunk_summaries = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk_summary = self._summarize_section(
                chunk,
                chunk_index=idx,
                total_chunks=len(chunks),
                max_length=chunk_word_budget,
                detailed=detailed,
            ).strip()
            if chunk_summary:
                chunk_summaries.append(f"Section {idx}/{len(chunks)} summary:\n{chunk_summary}")

        if not chunk_summaries:
            return self._fallback_summarize(text, max_length)

        synthesis_text = "\n\n".join(chunk_summaries)
        if len(synthesis_text) <= self.CONTEXT_LIMIT:
            system = "You are a research assistant combining section summaries into one paper summary."
            if detailed:
                user = (
                    f"These are summaries of sequential sections of the same academic paper.\n"
                    f"Write one coherent technical summary under {max_length} words covering the full paper's contribution, methodology, results, limitations, and relevance.\n"
                    f"Do not describe the input as excerpts unless the summaries themselves indicate missing content.\n\nSection summaries:\n{synthesis_text}"
                )
            else:
                user = (
                    f"These are summaries of sequential sections of the same academic paper.\n"
                    f"Write one concise summary under {max_length} words covering the full paper's main contribution, method, and findings.\n"
                    f"Do not describe the input as excerpts unless the summaries themselves indicate missing content.\n\nSection summaries:\n{synthesis_text}"
                )
            return self._summarize_from_prompt(system, user, max_length, detailed, fallback_text=synthesis_text)

        return self._fallback_summarize(synthesis_text, max_length)

    def _summarize_from_prompt(self, system: str, user: str, max_length: int, detailed: bool, fallback_text: str) -> str:
        """Run a custom summarization prompt through the configured provider."""
        try:
            return self.complete(system, user, max_length=max_length, detailed=detailed)
        except Exception as e:
            provider = self._active_provider()
            print(f"[summarizer] {provider} failed during summary synthesis: {e}. Using fallback.")
            return self._fallback_summarize(fallback_text, max_length)

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_ollama(self, system: str, user: str, max_length: int, detailed: bool) -> str:
        host = os.getenv("OLLAMA_HOST", self._defaults["ollama"]["host"])
        num_predict = max_length * 2 if detailed else max_length
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream": False,
                "think": False,
                "options": {"num_predict": num_predict},
            },
            timeout=self.OLLAMA_DETAILED_TIMEOUT_SECONDS if detailed else self.OLLAMA_CONCISE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def _call_gemini(self, system: str, user: str, max_length: int, detailed: bool) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_length * 2},
        }
        result = _gemini_post(url, api_key, payload)
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_anthropic(self, system: str, user: str, max_length: int, detailed: bool) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_length * 2,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"].strip()

    def _call_openai(self, system: str, user: str, max_length: int, detailed: bool) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "max_tokens": max_length * 2,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def dispatch_chat_gemini(self, system: str, messages: list) -> str:
        """Always use Gemini for chat regardless of SUMMARIZER_PROVIDER.
        Falls back to the configured provider if GEMINI_API_KEY is not set."""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return self._dispatch_chat(system, messages)
        model = os.getenv("CHAT_LLM_MODEL", "gemini-2.0-flash")
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        result = _gemini_post(url, api_key, {"system_instruction": {"parts": [{"text": system}]}, "contents": contents})
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _dispatch_chat(self, system: str, messages: list) -> str:
        """Send a multi-turn chat request to the configured provider. messages = [{role, content}]"""
        provider = self._active_provider()
        self.model = self._active_model()
        if provider == "ollama":
            host = self._defaults["ollama"]["host"]
            num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
            response = requests.post(
                f"{host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}] + messages,
                    "stream": False,
                    "think": False,
                    "options": {"num_ctx": num_ctx},
                },
                timeout=300,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()

        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            contents = []
            for m in messages:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            result = _gemini_post(url, api_key, {"system_instruction": {"parts": [{"text": system}]}, "contents": contents})
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": self.model, "max_tokens": 1024, "system": system, "messages": messages},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"].strip()

        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "max_tokens": 1024, "messages": [{"role": "system", "content": system}] + messages},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        else:
            raise ValueError(f"Unknown provider: {provider!r}")

    # ------------------------------------------------------------------

    def _fallback_summarize(self, text: str, max_length: int = 300) -> str:
        lines = text.split("\n")
        for line in lines:
            if any(k in line.lower() for k in ["abstract", "summary", "introduction", "conclusion"]):
                if len(line.strip()) > 50:
                    return line.strip()[:max_length]
        for line in lines:
            if line.strip() and len(line.strip()) > 50:
                return line.strip()[:max_length]
        return text[:max_length] if text else ""
