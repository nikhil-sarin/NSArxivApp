"""Extract public corresponding-author contact details from paper text or arXiv source."""

from __future__ import annotations

import gzip
import io
import re
import tarfile

import requests


MAX_SOURCE_BYTES = 50 * 1024 * 1024
MAX_TEX_BYTES = 8 * 1024 * 1024
PLACEHOLDER_CONTACT_NAMES = {"paper contact", "corresponding author"}
EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    flags=re.I,
)


def _clean_tex(value: str) -> str:
    value = re.sub(r"\\(?:textsuperscript|thanks|footnote)\s*\{[^{}]*\}", "", value)
    value = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", value)
    value = value.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", value).strip(" ,;~")


def _valid_email(value: str) -> str | None:
    cleaned = value.strip().strip(".,;:()[]{}<>")
    if not EMAIL_RE.fullmatch(cleaned):
        return None
    if cleaned.casefold().endswith("@example.com"):
        return None
    return cleaned


def contact_from_text(text: str) -> dict | None:
    """Return a conservative public contact found near the start of rendered text."""
    front_matter = text[:30_000]
    match = EMAIL_RE.search(front_matter)
    if not match:
        return None
    email = _valid_email(match.group(1))
    if not email:
        return None
    return {"name": "Paper contact", "email": email, "source": "paper full text"}


def contact_from_tex(tex: str) -> dict | None:
    """Prefer an explicit AASTeX corresponding-author/email declaration."""
    email_matches = list(
        re.finditer(r"\\email\s*(?:\[[^]]*\])?\s*\{([^{}]+)\}", tex, flags=re.I)
    )
    for email_match in email_matches:
        email = _valid_email(email_match.group(1))
        if not email:
            continue
        prefix = tex[max(0, email_match.start() - 2_000):email_match.start()]
        names = list(
            re.finditer(
                r"\\correspondingauthor\s*(?:\[[^]]*\])?\s*\{([^{}]+)\}",
                prefix,
                flags=re.I,
            )
        )
        name = _clean_tex(names[-1].group(1)) if names else "Corresponding author"
        return {"name": name or "Corresponding author", "email": email, "source": "arXiv source"}
    return None


def _tex_documents(payload: bytes) -> list[str]:
    documents: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            total = 0
            for member in archive.getmembers():
                if not member.isfile() or not member.name.casefold().endswith(".tex"):
                    continue
                if member.size < 0 or total + member.size > MAX_TEX_BYTES:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                total += member.size
                documents.append(handle.read().decode("utf-8", errors="replace"))
    except tarfile.ReadError:
        try:
            documents.append(gzip.decompress(payload).decode("utf-8", errors="replace"))
        except (OSError, UnicodeDecodeError):
            documents.append(payload.decode("utf-8", errors="replace"))
    return documents


def fetch_arxiv_contact(arxiv_id: str, *, timeout: int = 60) -> dict | None:
    """Fetch bounded public arXiv source and extract an explicitly declared contact."""
    clean_id = re.sub(r"v\d+$", "", arxiv_id.strip())
    response = requests.get(
        f"https://export.arxiv.org/e-print/{clean_id}",
        headers={"User-Agent": "NSArxivApp/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.content
    if len(payload) > MAX_SOURCE_BYTES:
        raise ValueError("arXiv source exceeds the contact-extraction size limit")
    for document in _tex_documents(payload):
        if contact := contact_from_tex(document):
            return contact
    return None


def find_public_contact(paper_text: str, arxiv_id: str) -> dict | None:
    """Prefer a named arXiv declaration, retaining a rendered-text email fallback."""
    text_contact = contact_from_text(paper_text)
    try:
        source_contact = fetch_arxiv_contact(arxiv_id)
    except (requests.RequestException, OSError, ValueError):
        source_contact = None
    return source_contact or text_contact


def display_name(contact: dict) -> str:
    """Hide internal fallback labels from user-facing contact descriptions."""
    name = str(contact.get("name", "")).strip()
    return (
        "Public contact"
        if not name or name.casefold() in PLACEHOLDER_CONTACT_NAMES
        else name
    )
