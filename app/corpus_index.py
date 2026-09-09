"""Create attributable page/section records for retrieval and citations."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

from app import research_db
from app.paper_text import get_paper_text


CHUNK_CHARS = 2800
CHUNK_OVERLAP_CHARS = 300


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _clean(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip())


def split_text(text: str, *, source: str, page: int | None = None) -> list[dict]:
    """Split text at paragraph boundaries while preserving useful locators."""
    text = _clean(text)
    if not text:
        return []
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[dict] = []
    current = ""
    heading = ""
    for paragraph in paragraphs:
        if len(paragraph) < 140 and not paragraph.endswith("."):
            heading = paragraph[:160]
        if current and len(current) + len(paragraph) + 2 > CHUNK_CHARS:
            chunks.append({"text": current, "heading": heading})
            current = current[-CHUNK_OVERLAP_CHARS:] + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append({"text": current, "heading": heading})
    for index, chunk in enumerate(chunks, start=1):
        chunk["locator"] = {
            "source": source,
            "page": page,
            "chunk": index,
            "section": chunk.pop("heading", ""),
        }
    return chunks


def _pdf_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    return [(index, page.extract_text() or "") for index, page in enumerate(reader.pages, start=1)]


def _write_records(paper: dict, records: list[dict], *, owner_type: str, kind: str, vector_db) -> int:
    paper_id = str(paper.get("arxiv_id", ""))
    title = str(paper.get("title", paper_id))
    research_db.delete_documents(owner_type, paper_id)
    vector_db.delete_documents(owner_type, paper_id)
    for index, record in enumerate(records, start=1):
        document_id = f"{owner_type}:{paper_id}:{index}"
        locator = {**record.get("locator", {}), "arxiv_id": paper_id}
        research_db.upsert_document(
            document_id,
            kind=kind,
            owner_type=owner_type,
            owner_id=paper_id,
            paper_id=paper_id,
            title=title,
            text=record["text"],
            locator=locator,
        )
        vector_db.upsert_document(
            document_id,
            record["text"],
            {
                "kind": kind,
                "owner_key": f"{owner_type}:{paper_id}",
                "paper_id": paper_id,
                "title": title,
                "locator_json": json.dumps(locator),
            },
        )
    return len(records)


def index_paper(paper: dict, *, arxiv_client, pdf_extractor, vector_db, force: bool = False) -> int:
    """Index cached PDF pages when available, otherwise section-like text chunks."""
    paper_id = str(paper.get("arxiv_id", ""))
    if not paper_id:
        return 0
    if not force and research_db.has_documents("paper_content", paper_id):
        return 0

    pdf_path = arxiv_client.get_pdf_path_by_id(paper_id)
    records: list[dict] = []
    if pdf_path is not None and pdf_path.exists():
        for page_number, page_text in _pdf_pages(pdf_path):
            records.extend(split_text(page_text, source="pdf", page=page_number))
    else:
        text = get_paper_text(
            paper_id,
            arxiv_client,
            pdf_extractor,
            title=paper.get("title"),
            pdf_url=paper.get("pdf_url"),
        )
        records = split_text(text, source="html_or_pdf")
    return _write_records(paper, records, owner_type="paper_content", kind="paper_chunk", vector_db=vector_db)


def index_report(paper: dict, report_path: Path, *, vector_db) -> int:
    """Index generated report prose separately from primary paper evidence."""
    parser = _TextExtractor()
    parser.feed(report_path.read_text(encoding="utf-8", errors="replace"))
    records = split_text("\n\n".join(parser.parts), source="generated_report")
    return _write_records(paper, records, owner_type="report", kind="report", vector_db=vector_db)


def index_text_record(
    *,
    owner_type: str,
    owner_id: str,
    kind: str,
    title: str,
    text: str,
    vector_db,
    paper_id: str | None = None,
    locator: dict | None = None,
) -> int:
    """Index arbitrary attributable research text such as meetings or OCR notes."""
    records = split_text(text, source=kind)
    research_db.delete_documents(owner_type, owner_id)
    vector_db.delete_documents(owner_type, owner_id)
    for index, record in enumerate(records, start=1):
        document_id = f"{owner_type}:{owner_id}:{index}"
        record_locator = {**(locator or {}), **record.get("locator", {})}
        research_db.upsert_document(
            document_id,
            kind=kind,
            owner_type=owner_type,
            owner_id=owner_id,
            paper_id=paper_id,
            title=title,
            text=record["text"],
            locator=record_locator,
        )
        vector_db.upsert_document(
            document_id,
            record["text"],
            {
                "kind": kind,
                "owner_key": f"{owner_type}:{owner_id}",
                "paper_id": paper_id or "",
                "title": title,
                "locator_json": json.dumps(record_locator),
            },
        )
    return len(records)


def index_figures(paper: dict, figures: list[dict], *, vector_db) -> int:
    """Index extracted figure captions with stable figure locators."""
    paper_id = str(paper.get("arxiv_id", ""))
    research_db.delete_documents("figure", paper_id)
    vector_db.delete_documents("figure", paper_id)
    count = 0
    for index, figure in enumerate(figures, start=1):
        caption = str(figure.get("caption", "")).strip()
        if not caption:
            continue
        locator = {
            "arxiv_id": paper_id,
            "figure": figure.get("number", index),
            "path": str(figure.get("path", "")),
            "source": "paper_figure",
        }
        document_id = f"figure:{paper_id}:{index}"
        research_db.upsert_document(
            document_id,
            kind="figure",
            owner_type="figure",
            owner_id=paper_id,
            paper_id=paper_id,
            title=str(paper.get("title", paper_id)),
            text=caption,
            locator=locator,
        )
        vector_db.upsert_document(
            document_id,
            caption,
            {
                "kind": "figure",
                "owner_key": f"figure:{paper_id}",
                "paper_id": paper_id,
                "title": str(paper.get("title", paper_id)),
                "locator_json": json.dumps(locator),
            },
        )
        count += 1
    return count
