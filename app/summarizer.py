"""LLM-based paper summarization with multi-provider support.

Supported providers (set via SUMMARIZER_PROVIDER env var):
  ollama    - Local or remote Ollama server (default)
  gemini    - Google Gemini API
  anthropic - Anthropic Claude API
  openai    - OpenAI API
"""

import os
import requests
from typing import Optional


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
            text = text[:self.CONTEXT_LIMIT] + "\n...[truncated]"

        provider = self._active_provider()
        self.model = self._active_model()

        system_prompt, user_prompt = _build_prompt(text, max_length, detailed)
        try:
            dispatch = {
                "ollama":    self._call_ollama,
                "gemini":    self._call_gemini,
                "anthropic": self._call_anthropic,
                "openai":    self._call_openai,
            }
            fn = dispatch.get(provider)
            if fn is None:
                raise ValueError(f"Unknown provider: {provider!r}. Choose from: {list(dispatch)}")
            return fn(system_prompt, user_prompt, max_length, detailed)
        except Exception as e:
            print(f"[summarizer] {provider} failed: {e}. Using fallback.")
            return self._fallback_summarize(text, max_length)

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_ollama(self, system: str, user: str, max_length: int, detailed: bool) -> str:
        host = self._defaults["ollama"]["host"]
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
                "options": {"num_predict": num_predict},
            },
            timeout=300 if detailed else 120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def _call_gemini(self, system: str, user: str, max_length: int, detailed: bool) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        # Gemini REST API — no extra SDK needed
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_length * 2},
        }
        response = requests.post(
            url, params={"key": api_key}, json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

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

    def _dispatch_chat(self, system: str, messages: list) -> str:
        """Send a multi-turn chat request to the configured provider. messages = [{role, content}]"""
        provider = self._active_provider()
        self.model = self._active_model()
        if provider == "ollama":
            host = self._defaults["ollama"]["host"]
            response = requests.post(
                f"{host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}] + messages,
                    "stream": False,
                },
                timeout=120,
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
            response = requests.post(
                url, params={"key": api_key},
                json={"system_instruction": {"parts": [{"text": system}]}, "contents": contents},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

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
