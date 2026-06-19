"""Fetch announcement-day ArXiv papers using the RSS/catchup flow."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime

import requests

RSS_URL = "https://rss.arxiv.org/rss/{list_name}"
CATCHUP_URL = "https://arxiv.org/catchup/{list_name}/{date}"
OAI_URL = "https://oaipmh.arxiv.org/oai"
RATE_LIMIT_SECONDS = 3.0
MAX_RETRIES = 4
USER_AGENT = "NSArxivApp/1.0 (+https://github.com/nikhil-sarin/NSArxivApp)"

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
}

_NEW_SECTION_RE = re.compile(
    r"<h3>\s*New submissions[^<]*</h3>(.*?)(?:<h3>|</dl>)",
    flags=re.DOTALL | re.IGNORECASE,
)
_ARXIV_ID_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})")
_VERSION_RE = re.compile(r"v\d+$")


@dataclass(frozen=True)
class AnnouncementPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    primary_category: str
    categories: list[str]
    announced_date: str
    v1_date: str
    abs_url: str
    pdf_url: str


def parse_catchup_new_ids(html: str) -> list[str]:
    """Extract IDs from the catchup page's New submissions section only."""
    section = _NEW_SECTION_RE.search(html)
    if not section:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for arxiv_id in _ARXIV_ID_RE.findall(section.group(1)):
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            ordered.append(arxiv_id)
    return ordered


def _split_authors(text: str) -> list[str]:
    if not text:
        return []
    text = text.replace(" and ", ", ")
    return [author.strip() for author in text.split(",") if author.strip()]


def _get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    timeout: float = 60.0,
) -> requests.Response:
    """GET with simple retry/backoff for 429s and transient transport errors."""
    headers = {"User-Agent": USER_AGENT}
    response: requests.Response | None = None
    for attempt in range(MAX_RETRIES):
        last = attempt == MAX_RETRIES - 1
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers=headers,
                allow_redirects=True,
            )
        except requests.RequestException:
            if last:
                raise
            time.sleep(RATE_LIMIT_SECONDS * 2**attempt)
            continue
        if response.status_code == 429 and not last:
            retry_after = response.headers.get("Retry-After", "").strip()
            delay = float(retry_after) if retry_after.isdigit() else RATE_LIMIT_SECONDS * 2**attempt
            time.sleep(delay)
            continue
        return response
    assert response is not None
    return response


def fetch_catchup_ids(list_name: str, target_date: date) -> list[str]:
    """Fetch canonical new-submission IDs for one category/day."""
    response = _get_with_retry(
        CATCHUP_URL.format(list_name=list_name, date=target_date.isoformat())
    )
    response.raise_for_status()
    return parse_catchup_new_ids(response.text)


def parse_rss_papers(xml_bytes: bytes) -> tuple[str, list[AnnouncementPaper]]:
    """Parse one arXiv RSS feed, keeping only announce_type == new items."""
    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        return "", []
    pub = (channel.findtext("pubDate") or "").strip()
    announced = parsedate_to_datetime(pub).date().isoformat() if pub else ""

    papers: list[AnnouncementPaper] = []
    for item in channel.findall("item"):
        fields: dict[str, list[str]] = {}
        for child in item:
            local = child.tag.rsplit("}", 1)[-1]
            fields.setdefault(local, []).append((child.text or "").strip())
        if (fields.get("announce_type") or [""])[0] != "new":
            continue
        link = (fields.get("link") or [""])[0]
        arxiv_id = _VERSION_RE.sub("", link.rsplit("/", 1)[-1])
        if not arxiv_id:
            continue
        description = (fields.get("description") or [""])[0]
        _, _, abstract = description.partition("Abstract:")
        categories = [category for category in fields.get("category", []) if category]
        papers.append(
            AnnouncementPaper(
                arxiv_id=arxiv_id,
                title=" ".join((fields.get("title") or [""])[0].split()),
                authors=_split_authors((fields.get("creator") or [""])[0]),
                abstract=" ".join(abstract.split()),
                primary_category=categories[0] if categories else "",
                categories=categories,
                announced_date=announced,
                v1_date=announced,
                abs_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )
        )
    return announced, papers


def parse_get_record(xml_bytes: bytes) -> AnnouncementPaper | None:
    """Parse an OAI-PMH GetRecord response."""
    root = ET.fromstring(xml_bytes)
    err = root.find("oai:error", NS)
    if err is not None:
        if err.get("code") == "idDoesNotExist":
            return None
        raise RuntimeError(f"OAI-PMH error [{err.get('code')}]: {err.text}")
    record = root.find(".//oai:record", NS)
    if record is None:
        return None
    arxiv_id = (record.findtext(".//raw:id", namespaces=NS) or "").strip()
    if not arxiv_id:
        return None
    announced = (record.findtext("oai:header/oai:datestamp", namespaces=NS) or "").strip()
    categories = (record.findtext(".//raw:categories", namespaces=NS) or "").split()
    primary_category = categories[0] if categories else ""
    title = " ".join((record.findtext(".//raw:title", namespaces=NS) or "").split())
    abstract = " ".join((record.findtext(".//raw:abstract", namespaces=NS) or "").split())
    authors = _split_authors((record.findtext(".//raw:authors", namespaces=NS) or "").strip())
    versions = record.findall(".//raw:version", NS)
    v1_date = ""
    if versions:
        v1_text = (versions[0].findtext("raw:date", namespaces=NS) or "").strip()
        if v1_text:
            v1_date = parsedate_to_datetime(v1_text).date().isoformat()
    return AnnouncementPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        primary_category=primary_category,
        categories=categories,
        announced_date=announced,
        v1_date=v1_date,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def fetch_record(arxiv_id: str) -> AnnouncementPaper | None:
    """Fetch one paper via OAI-PMH."""
    response = _get_with_retry(
        OAI_URL,
        params={
            "verb": "GetRecord",
            "identifier": f"oai:arXiv.org:{arxiv_id}",
            "metadataPrefix": "arXivRaw",
        },
    )
    response.raise_for_status()
    return parse_get_record(response.content)


def fetch_rss_papers(list_name: str, target_date: date) -> list[AnnouncementPaper] | None:
    """Fetch the latest RSS papers if the feed date matches target_date."""
    response = _get_with_retry(RSS_URL.format(list_name=list_name))
    response.raise_for_status()
    announced, papers = parse_rss_papers(response.content)
    if announced != target_date.isoformat():
        return None
    return papers


def fetch_via_catchup(list_name: str, target_date: date) -> list[AnnouncementPaper]:
    """Backfill older dates via catchup + OAI."""
    ids = fetch_catchup_ids(list_name, target_date)
    papers: list[AnnouncementPaper] = []
    for index, arxiv_id in enumerate(ids):
        if index > 0:
            time.sleep(RATE_LIMIT_SECONDS)
        paper = fetch_record(arxiv_id)
        if paper is not None:
            papers.append(paper)
    return papers


def fetch_papers(list_name: str, target_date: date) -> list[AnnouncementPaper]:
    """Fetch new submissions for one category/day."""
    papers = fetch_rss_papers(list_name, target_date)
    if papers is not None:
        return papers
    return fetch_via_catchup(list_name, target_date)


def paper_to_metadata(paper: AnnouncementPaper) -> dict:
    """Convert an announcement paper to the metadata shape used in this app."""
    return {
        "title": paper.title,
        "authors": paper.authors,
        "summary": paper.abstract,
        "abstract": paper.abstract,
        "published": paper.announced_date,
        "pdf_url": paper.pdf_url,
        "arxiv_id": paper.arxiv_id,
        "categories": paper.categories,
        "comment": "",
        "primary_category": paper.primary_category,
        "announced_date": paper.announced_date,
        "v1_date": paper.v1_date,
        "abs_url": paper.abs_url,
    }
