"""Fetch full paper text from ArXiv HTML format.

ArXiv provides HTML versions of most papers at https://arxiv.org/html/{arxiv_id}
This is lighter than the TeX tarball and gives complete, clean prose text.
"""

import re
import requests
from pathlib import Path
from typing import Optional


def _clean_html(html: str) -> str:
    """Strip HTML tags and clean up whitespace, keeping prose text."""
    try:
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.chunks = []
                self._skip = False
                self._skip_tags = {"script", "style", "nav", "footer", "head",
                                   "figure", "table", "math", "svg"}
                self._block_tags = {"p", "h1", "h2", "h3", "h4", "section",
                                    "div", "li", "br", "tr"}
                self._depth = {t: 0 for t in self._skip_tags}

            def handle_starttag(self, tag, attrs):
                if tag in self._skip_tags:
                    self._depth[tag] = self._depth.get(tag, 0) + 1
                if tag in self._block_tags:
                    self.chunks.append("\n")

            def handle_endtag(self, tag):
                if tag in self._skip_tags:
                    self._depth[tag] = max(0, self._depth.get(tag, 0) - 1)
                if tag in self._block_tags:
                    self.chunks.append("\n")

            def handle_data(self, data):
                if any(self._depth.get(t, 0) > 0 for t in self._skip_tags):
                    return
                self.chunks.append(data)

        extractor = TextExtractor()
        extractor.feed(html)
        text = "".join(extractor.chunks)

    except Exception:
        # Fallback: naive tag stripping
        text = re.sub(r"<[^>]+>", " ", html)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_html_text(arxiv_id: str, cache_dir: Path) -> Optional[str]:
    """
    Fetch full paper text via ArXiv HTML format.
    Returns plain text, or None if unavailable.
    Caches result as {safe_id}_html.txt in cache_dir.
    """
    clean_id = arxiv_id.split("v")[0]
    safe_id = clean_id.replace(".", "_")
    cache_path = cache_dir / f"{safe_id}_html.txt"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")

    url = f"https://arxiv.org/html/{clean_id}"
    try:
        resp = requests.get(
            url, timeout=30,
            headers={"User-Agent": "NSArxivApp/1.0"},
            allow_redirects=True,
        )
        if resp.status_code != 200:
            print(f"[tex_extractor] HTML not available for {clean_id} (status {resp.status_code})")
            return None

        text = _clean_html(resp.text)
        if len(text) < 500:
            print(f"[tex_extractor] HTML text too short for {clean_id}, likely unavailable")
            return None

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return text

    except Exception as e:
        print(f"[tex_extractor] Failed to fetch HTML for {clean_id}: {e}")
        return None
