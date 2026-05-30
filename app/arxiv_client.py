"""ArXiv API client for fetching paper metadata and PDFs."""

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arxiv
import requests
from pypdf import PdfReader


@dataclass
class ArxivAuthor:
    name: str


@dataclass
class ArxivSearchResult:
    entry_id: str
    title: str
    authors: List[ArxivAuthor]
    summary: str
    published: datetime
    pdf_url: str
    categories: List[str]
    comment: Optional[str] = None


class ArxivClient:
    """Client for interacting with ArXiv API."""

    SEARCH_RESULT_LIMIT = 20
    SEARCH_REQUEST_DELAY_SECONDS = 6.0
    SEARCH_CACHE_TTL_SECONDS = 900.0
    SEARCH_RETRY_ATTEMPTS = 1
    SEARCH_FALLBACK_OVERSCAN = 3
    SEARCH_API_COOLDOWN_SECONDS = 600.0
    PDF_CHUNK_SIZE = 1024 * 1024
    PDF_DOWNLOAD_ATTEMPTS = 3
    REQUEST_TIMEOUT = (10, 120)
    USER_AGENT = "NSArxivApp/1.0"
    PDF_REQUEST_DELAY_SECONDS = 1.0

    def __init__(self, download_dir: str = "data/papers"):
        # delay_seconds: wait between requests to avoid 429 rate limiting
        # num_retries: retry on transient failures (arxiv.Client default is 3)
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=5.0,
            num_retries=0,
        )
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._download_lock = threading.Lock()
        self._last_pdf_request = 0.0
        self._search_lock = threading.Lock()
        self._last_search_request = 0.0
        self._api_backoff_until = 0.0
        self._search_cache: Dict[Tuple[str, int, Tuple[str, ...], Optional[str], str], Tuple[float, List[object]]] = {}

    def _build_pdf_filename(self, arxiv_id: str, title: Optional[str] = None) -> str:
        """Build a stable PDF filename from an arXiv id and optional title."""
        clean_id = arxiv_id.split("/")[-1].split("v")[0]
        safe_id = clean_id.replace(".", "_")
        safe_title = re.sub(r"[^\w\-]", "_", title or "").strip("_")[:40]
        if safe_title:
            return f"{safe_id}_{safe_title}.pdf"
        return f"{safe_id}.pdf"

    def _rate_limit_pdf_request(self) -> None:
        """Serialize direct PDF fetches so bulk ingest stays under ArXiv's rate limits."""
        with self._download_lock:
            now = time.monotonic()
            wait_seconds = self.PDF_REQUEST_DELAY_SECONDS - (now - self._last_pdf_request)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_pdf_request = time.monotonic()

    def _rate_limit_search_request(self) -> None:
        """Serialize ArXiv API searches to avoid hammering the export endpoint."""
        with self._search_lock:
            now = time.monotonic()
            wait_seconds = self.SEARCH_REQUEST_DELAY_SECONDS - (now - self._last_search_request)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_search_request = time.monotonic()

    def _strip_html(self, text: str) -> str:
        """Convert ArXiv HTML fragments into plain text."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    def _extract_meta_content(self, html: str, name: str) -> Optional[str]:
        """Extract the first matching meta tag content from an arXiv HTML page."""
        match = re.search(
            rf'<meta\s+name="{re.escape(name)}"\s+content="(.*?)"\s*/?>',
            html,
            flags=re.I | re.S,
        )
        if not match:
            return None
        return unescape(match.group(1)).strip()

    def _extract_meta_contents(self, html: str, name: str) -> List[str]:
        """Extract all matching meta tag contents from an arXiv HTML page."""
        matches = re.findall(
            rf'<meta\s+name="{re.escape(name)}"\s+content="(.*?)"\s*/?>',
            html,
            flags=re.I | re.S,
        )
        return [unescape(value).strip() for value in matches if value.strip()]

    def get_result_by_id(self, arxiv_id: str) -> ArxivSearchResult:
        """Fetch a single paper directly from its abstract page, avoiding the export API."""
        response = requests.get(
            f"https://arxiv.org/abs/{arxiv_id}",
            headers={"User-Agent": self.USER_AGENT},
            timeout=60,
        )
        if response.status_code == 404:
            raise RuntimeError(f"Could not find paper {arxiv_id} on ArXiv.")
        response.raise_for_status()
        html = response.text

        title = self._extract_meta_content(html, "citation_title") or arxiv_id
        authors = [
            ArxivAuthor(name=name)
            for name in self._extract_meta_contents(html, "citation_author")
        ]
        abstract = self._extract_meta_content(html, "citation_abstract") or ""
        pdf_url = self._extract_meta_content(html, "citation_pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}"
        canonical_id = self._extract_meta_content(html, "citation_arxiv_id") or arxiv_id

        date_value = self._extract_meta_content(html, "citation_date")
        published = datetime.now(timezone.utc)
        if date_value:
            for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
                try:
                    published = datetime.strptime(date_value, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue

        categories: List[str] = []
        subjects_match = re.search(
            r'<span class="descriptor">Subjects:</span>\s*(.*?)\s*</td>',
            html,
            flags=re.I | re.S,
        )
        if subjects_match:
            subjects_text = self._strip_html(subjects_match.group(1))
            categories = [part.strip() for part in subjects_text.split(";") if part.strip()]

        comments_match = re.search(
            r'<span class="descriptor">Comments:</span>\s*(.*?)\s*</td>',
            html,
            flags=re.I | re.S,
        )
        comment = self._strip_html(comments_match.group(1)) if comments_match else None

        return ArxivSearchResult(
            entry_id=f"https://arxiv.org/abs/{canonical_id}",
            title=title,
            authors=authors,
            summary=abstract,
            published=published,
            pdf_url=pdf_url,
            categories=categories,
            comment=comment,
        )

    def _search_html(
        self,
        query: str,
        max_results: int,
        categories: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        author: str = "",
    ) -> List[ArxivSearchResult]:
        """Fallback search using the public ArXiv website when the export API is rate-limited."""
        search_terms = " ".join(part for part in [query.strip(), author.strip()] if part).strip()
        if not search_terms:
            raise RuntimeError("ArXiv web fallback requires a query or author term.")

        self._rate_limit_search_request()
        response = requests.get(
            "https://arxiv.org/search/",
            params={
                "query": search_terms,
                "searchtype": "all",
                "source": "header",
                "order": "-announced_date_first",
            },
            headers={"User-Agent": self.USER_AGENT},
            timeout=60,
        )
        response.raise_for_status()
        html = response.text

        results: List[ArxivSearchResult] = []
        for block in re.findall(r'<li class="arxiv-result">(.*?)</li>', html, flags=re.S):
            abs_match = re.search(r'href="https://arxiv\.org/abs/([^"#?]+)"', block)
            if not abs_match:
                continue
            arxiv_id = abs_match.group(1)

            title_match = re.search(r'<p class="title is-5 mathjax">\s*(.*?)\s*</p>', block, flags=re.S)
            title = self._strip_html(title_match.group(1)) if title_match else arxiv_id

            authors_match = re.search(r'<p class="authors">(.*?)</p>', block, flags=re.S)
            author_names = []
            if authors_match:
                author_names = [
                    self._strip_html(name)
                    for name in re.findall(r"<a [^>]*>(.*?)</a>", authors_match.group(1), flags=re.S)
                ]

            abstract_match = re.search(
                r'<span class="abstract-full has-text-grey-dark mathjax"[^>]*>(.*?)</span>',
                block,
                flags=re.S,
            )
            if abstract_match:
                summary = self._strip_html(abstract_match.group(1))
            else:
                short_match = re.search(
                    r'<span class="abstract-short has-text-grey-dark mathjax"[^>]*>(.*?)</span>',
                    block,
                    flags=re.S,
                )
                summary = self._strip_html(short_match.group(1)) if short_match else ""
            summary = re.sub(r"(Less|More)\s*$", "", summary).strip()

            categories_found = [
                self._strip_html(cat)
                for cat in re.findall(r'<span class="tag [^"]*"[^>]*>(.*?)</span>', block, flags=re.S)
            ]

            comments_match = re.search(r'<p class="comments is-size-7">\s*(.*?)\s*</p>', block, flags=re.S)
            comment = self._strip_html(comments_match.group(1)) if comments_match else None
            if comment and comment.lower().startswith("comments:"):
                comment = comment.split(":", 1)[1].strip()

            submitted_match = re.search(r"<span[^>]*>Submitted</span>\s*([^;]+);", block, flags=re.S)
            published = datetime.now(timezone.utc)
            if submitted_match:
                try:
                    published = datetime.strptime(submitted_match.group(1).strip(), "%d %B, %Y").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass

            result = ArxivSearchResult(
                entry_id=f"https://arxiv.org/abs/{arxiv_id}",
                title=title,
                authors=[ArxivAuthor(name=name) for name in author_names],
                summary=summary,
                published=published,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                categories=categories_found,
                comment=comment,
            )
            results.append(result)

        filtered_results = []
        author_lc = author.strip().lower()
        requested_categories = set(categories or [])
        for result in results:
            if author_lc and not any(author_lc in author_obj.name.lower() for author_obj in result.authors):
                continue
            if requested_categories and not requested_categories.intersection(result.categories):
                continue
            if date_from and result.published < date_from:
                continue
            filtered_results.append(result)
            if len(filtered_results) >= max_results:
                break

        return filtered_results

    def _download_pdf(self, pdf_url: str, pdf_path: Path) -> None:
        """Download a PDF atomically with retries on short or interrupted transfers."""
        tmp_path = pdf_path.with_suffix(f"{pdf_path.suffix}.part")
        last_error: Optional[Exception] = None

        for attempt in range(1, self.PDF_DOWNLOAD_ATTEMPTS + 1):
            try:
                if tmp_path.exists():
                    tmp_path.unlink()

                self._rate_limit_pdf_request()
                with requests.get(
                    pdf_url,
                    stream=True,
                    timeout=self.REQUEST_TIMEOUT,
                    headers={"User-Agent": self.USER_AGENT},
                    allow_redirects=True,
                ) as response:
                    response.raise_for_status()

                    content_length = response.headers.get("Content-Length")
                    expected_bytes = int(content_length) if content_length and content_length.isdigit() else None
                    written_bytes = 0

                    with tmp_path.open("wb") as fh:
                        for chunk in response.iter_content(chunk_size=self.PDF_CHUNK_SIZE):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            written_bytes += len(chunk)

                if written_bytes == 0:
                    raise IOError("download returned no data")
                if expected_bytes is not None and written_bytes != expected_bytes:
                    raise IOError(f"retrieval incomplete: got {written_bytes} out of {expected_bytes} bytes")

                with tmp_path.open("rb") as fh:
                    if fh.read(5) != b"%PDF-":
                        raise IOError("downloaded content was not a PDF")

                tmp_path.replace(pdf_path)
                return
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt < self.PDF_DOWNLOAD_ATTEMPTS:
                    time.sleep(attempt)

        raise RuntimeError(f"Failed to download PDF from {pdf_url}: {last_error}") from last_error

    def _is_valid_pdf(self, pdf_path: Path) -> bool:
        """Return whether a cached PDF exists and can be parsed."""
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            return False

        try:
            with pdf_path.open("rb") as fh:
                if fh.read(5) != b"%PDF-":
                    return False
            reader = PdfReader(str(pdf_path))
            return len(reader.pages) > 0
        except Exception as exc:
            print(f"Invalid cached PDF {pdf_path}: {exc}")
            return False

    def search(
        self,
        query: str = "",
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        author: str = "",
    ) -> List[arxiv.Result]:
        """Search for papers on ArXiv, optionally filtered by category, date, and author."""
        max_results = max(1, min(max_results, self.SEARCH_RESULT_LIMIT))
        parts = []

        if query:
            parts.append(f"(ti:{query} OR abs:{query})")

        if author:
            # ArXiv author search: quote multi-word names so they match as a phrase,
            # not as separate tokens (which would match "Nikhil X" and "Y Sarin" separately).
            # Also support lastname_firstinitial format e.g. sarin_n.
            clean_author = author.strip().strip('"').strip("'")
            if " " in clean_author and "_" not in clean_author:
                # Convert "Nikhil Sarin" -> "sarin_n" for most precise ArXiv matching,
                # but also keep the quoted form as a fallback OR clause.
                parts_name = clean_author.split()
                lastname = parts_name[-1].lower()
                firstinit = parts_name[0][0].lower()
                parts.append(f'(au:"{clean_author}" OR au:{lastname}_{firstinit})')
            else:
                parts.append(f'au:"{clean_author}"')

        if categories:
            parts.append("(" + " OR ".join(f"cat:{c}" for c in categories) + ")")

        if date_from:
            date_str = date_from.strftime("%Y%m%d")
            parts.append(f"submittedDate:[{date_str}000000 TO 99991231235959]")

        if not parts:
            # Nothing specified — refuse to return everything
            return []

        combined_query = " AND ".join(parts)
        cache_key = (
            combined_query,
            max_results,
            tuple(sorted(categories or [])),
            date_from.isoformat() if date_from else None,
            author.strip(),
        )
        cached = self._search_cache.get(cache_key)
        now = time.monotonic()
        if cached and now - cached[0] < self.SEARCH_CACHE_TTL_SECONDS:
            return list(cached[1])
        if now < self._api_backoff_until:
            if cached:
                return list(cached[1])
            html_results = self._search_html(
                query=query,
                max_results=max_results,
                categories=categories,
                date_from=date_from,
                author=author,
            )
            self._search_cache[cache_key] = (time.monotonic(), list(html_results))
            print("ArXiv export API is cooling down after rate limiting; using web search fallback.")
            return html_results

        search = arxiv.Search(
            query=combined_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        last_error: Optional[Exception] = None
        for attempt in range(1, self.SEARCH_RETRY_ATTEMPTS + 1):
            try:
                self._rate_limit_search_request()
                results = list(self.client.results(search))
                self._search_cache[cache_key] = (time.monotonic(), list(results))
                return results
            except arxiv.HTTPError as exc:
                last_error = exc
                if "429" not in str(exc):
                    raise
                break
            except Exception as exc:
                last_error = exc
                break

        if last_error is not None and "429" in str(last_error):
            self._api_backoff_until = time.monotonic() + self.SEARCH_API_COOLDOWN_SECONDS
            if cached:
                print("ArXiv API rate-limited search; using cached results.")
                return list(cached[1])
            html_results = self._search_html(
                query=query,
                max_results=max_results,
                categories=categories,
                date_from=date_from,
                author=author,
            )
            self._search_cache[cache_key] = (time.monotonic(), list(html_results))
            print("ArXiv API rate-limited search; using web search fallback.")
            return html_results

        if last_error is not None:
            raise last_error
        return []

    def get_pdf_path(self, result: arxiv.Result) -> Path:
        """Get or download the PDF for a paper result."""
        arxiv_id = result.entry_id.split("/")[-1]
        return self.get_pdf_path_for_id(arxiv_id, title=result.title, pdf_url=result.pdf_url)

    def get_pdf_path_for_id(self, arxiv_id: str, title: Optional[str] = None, pdf_url: Optional[str] = None) -> Path:
        """Get or download a PDF directly from an arXiv id without another API metadata lookup."""
        filename = self._build_pdf_filename(arxiv_id, title=title)
        pdf_path = self.download_dir / filename

        if pdf_path.exists():
            if self._is_valid_pdf(pdf_path):
                return pdf_path
            print(f"Removing invalid cached PDF: {pdf_path}")
            pdf_path.unlink()

        old = self.get_pdf_path_by_id(arxiv_id)
        if old is not None and old.exists():
            return old

        resolved_pdf_url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"Downloading PDF for {arxiv_id}")
        self._download_pdf(resolved_pdf_url, pdf_path)
        if not self._is_valid_pdf(pdf_path):
            if pdf_path.exists():
                pdf_path.unlink()
            raise RuntimeError(f"Downloaded PDF is invalid: {pdf_path}")
        return pdf_path

    def get_pdf_path_by_id(self, arxiv_id: str) -> Optional[Path]:
        """Find an already-downloaded PDF by arxiv ID embedded in the filename."""
        clean_id = arxiv_id.split("v")[0]  # strip version suffix
        safe_id = clean_id.replace(".", "_")
        # New naming: {safe_id}_*.pdf
        matches = list(self.download_dir.glob(f"{safe_id}_*.pdf"))
        for match in matches:
            if self._is_valid_pdf(match):
                return match
            print(f"Removing invalid cached PDF: {match}")
            match.unlink()
        # Legacy naming: title contained the id tokens
        for pdf in self.download_dir.glob("*.pdf"):
            if safe_id in pdf.stem or clean_id in pdf.stem:
                if self._is_valid_pdf(pdf):
                    return pdf
                print(f"Removing invalid cached PDF: {pdf}")
                pdf.unlink()
        return None

    def get_paper_metadata(self, result: object) -> dict:
        """Extract useful metadata from a paper result."""
        return {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "abstract": result.summary,
            "published": result.published.isoformat(),
            "pdf_url": result.pdf_url,
            "arxiv_id": result.entry_id.split("/")[-1],
            "categories": result.categories,
            "comment": result.comment,
        }
