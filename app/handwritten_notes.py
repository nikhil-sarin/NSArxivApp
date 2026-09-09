"""Local multimodal transcription for handwritten research notes."""

from __future__ import annotations

import base64
import os

import requests


def transcribe_image(image_bytes: bytes, *, context: str = "") -> str:
    """Transcribe a note image with a local Ollama vision model."""
    if not image_bytes:
        return ""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OCR_MODEL", "gemma3:latest")
    prompt = (
        "Transcribe this handwritten scientific note faithfully. Preserve equations, symbols, bullet structure, "
        "uncertain readings as [unclear], and do not add information. Return plain Markdown only."
    )
    if context.strip():
        prompt += f"\nPaper context for disambiguating names and symbols: {context[:2000]}"
    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "stream": False,
            "think": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()
