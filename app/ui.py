"""Streamlit frontend for the paper wiki application."""

import os
import re
import json
import shlex
import sys
import tempfile
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from html import escape as html_escape
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import subprocess
import textwrap

from app.arxiv_client import ArxivClient
from app.pdf_extractor import PDFExtractor
from app.summarizer import PaperSummarizer
from app.vector_db import PaperVectorDB
from app import paper_store
from app.paper_text import get_paper_text
from app.report_generator import ReportUnavailable, generate_report
from app.summary_workflow import completeness_issues, summarize_with_fallback
from app.tex_extractor import fetch_html_text
from app import researcher_profile
from app import idea_store
from app import privacy
from app import relevance
from app import research_db
from app import corpus_index
from app import retrieval
from app import project_store
from app import action_contract
from app import evaluation
from app import handwritten_notes
from app import index_jobs
from app import synthesis
from app import trends
from app import citation_opportunities
from app import citation_contacts
from app import citation_opportunity_store
from app import contribution_catalogue
from app import citation_discovery
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def get_vector_db():
    return PaperVectorDB()


@st.cache_resource
def get_summarizer():
    return PaperSummarizer()


@st.cache_resource
def get_arxiv_client():
    return ArxivClient()


@st.cache_resource
def get_pdf_extractor():
    return PDFExtractor()


def _session_vector_db() -> PaperVectorDB:
    """Initialize the embedding index only when a workflow actually needs it."""
    if "vdb" not in st.session_state:
        st.session_state.vdb = get_vector_db()
    return st.session_state.vdb


def _reload_papers_from_store() -> None:
    """Reload persisted papers into session state."""
    stored = paper_store.load_all_papers()
    st.session_state.papers = stored
    st.session_state.papers_store_mtime_ns = (
        paper_store.STORE_PATH.stat().st_mtime_ns if paper_store.STORE_PATH.exists() else None
    )


def _sync_papers_from_store() -> None:
    """Refresh session state when papers.json changes outside the current UI action."""
    current_mtime_ns = paper_store.STORE_PATH.stat().st_mtime_ns if paper_store.STORE_PATH.exists() else None
    if "papers" not in st.session_state or st.session_state.get("papers_store_mtime_ns") != current_mtime_ns:
        _reload_papers_from_store()


@st.fragment(run_every=300)
def _auto_sync_papers_from_store() -> None:
    """Periodically notice cron updates while a browser session is left open."""
    current_mtime_ns = paper_store.STORE_PATH.stat().st_mtime_ns if paper_store.STORE_PATH.exists() else None
    if st.session_state.get("papers_store_mtime_ns") != current_mtime_ns:
        _reload_papers_from_store()
        st.toast("Library updated from scheduled fetch.")
        st.rerun()


def _format_file_mtime(path: Path) -> str:
    if not path.exists():
        return "Never"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _latest_fetch_block(log_path: Path) -> str:
    """Return the most recent fetch-job block from the append-only log."""
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\[\d{4}-\d{2}-\d{2} .*?\] Fetching ", line):
            start = i
    return "\n".join(lines[start:]).strip()


def _recent_stored_papers(limit: int = 5) -> list[Dict]:
    papers = paper_store.load_all_papers()
    return sorted(papers, key=_published_sort_key, reverse=True)[:limit]


def init_session_state():
    """Initialize session state variables, loading persisted papers on first run."""
    st.session_state.summarizer = get_summarizer()
    st.session_state.arxiv = get_arxiv_client()
    st.session_state.pdf_extractor = get_pdf_extractor()
    _sync_papers_from_store()


def render_header():
    st.title("ArXiv Paper Wiki")
    st.markdown("Discover, summarize, and explore connections between research papers.")


def render_sidebar():
    """Render sidebar search controls. Returns (query, author, categories, max_results, date_from) or Nones."""
    st.sidebar.header("Search & Filters")
    search_limit = ArxivClient.SEARCH_RESULT_LIMIT

    DEFAULT_QUERY = "neutron star mergers OR kilonovae OR GRBs OR TDEs OR neutron stars OR gravitational waves OR supernovae OR FXTs"
    DEFAULT_CATEGORIES = ["astro-ph.HE"]

    if not st.session_state.get("sidebar_query_initialised"):
        st.session_state["sidebar_query_initialised"] = True
        st.session_state["sidebar_default_query"] = DEFAULT_QUERY
        st.session_state["sidebar_default_cats"] = DEFAULT_CATEGORIES

    query = st.sidebar.text_input("Search query", value=st.session_state.get("sidebar_default_query", DEFAULT_QUERY), placeholder="e.g., neutron star merger")
    author = st.sidebar.text_input("Author", placeholder="e.g., Sarin or Nikhil Sarin")

    st.sidebar.markdown("**Categories**")
    categories = st.sidebar.multiselect(
        "Select categories:",
        options=[
            "cs.LG", "cs.CL", "cs.CV", "cs.AI",
            "astro-ph.HE", "astro-ph.CO", "astro-ph.GA",
            "physics.hep-th", "gr-qc",
            "q-bio.QM", "q-fin.CP",
        ],
        default=st.session_state.get("sidebar_default_cats", DEFAULT_CATEGORIES),
    )

    max_results = st.sidebar.slider("Max results", 5, search_limit, min(20, search_limit))
    st.sidebar.caption(f"Searches are capped at {search_limit} papers to stay within ArXiv rate limits.")

    st.sidebar.markdown("**Date filter**")
    date_option = st.sidebar.selectbox(
        "Submitted since:",
        ["All time", "Today", "Last 3 days", "Last 7 days", "Last 30 days",
         "Last 3 months", "Last 6 months", "Last year", "Custom range"],
        index=0,
    )

    date_from: Optional[datetime] = None
    now = datetime.now(timezone.utc)
    if date_option == "Today":
        date_from = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif date_option == "Last 3 days":
        date_from = now - timedelta(days=3)
    elif date_option == "Last 7 days":
        date_from = now - timedelta(days=7)
    elif date_option == "Last 30 days":
        date_from = now - timedelta(days=30)
    elif date_option == "Last 3 months":
        date_from = now - timedelta(days=90)
    elif date_option == "Last 6 months":
        date_from = now - timedelta(days=180)
    elif date_option == "Last year":
        date_from = now - timedelta(days=365)
    elif date_option == "Custom range":
        custom_date = st.sidebar.date_input("From date:", value=now.date() - timedelta(days=30))
        date_from = datetime(custom_date.year, custom_date.month, custom_date.day, tzinfo=timezone.utc)

    search_clicked = st.sidebar.button("Search", type="primary")

    # Provider status
    st.sidebar.markdown("---")
    summ_provider = os.getenv("SUMMARIZER_PROVIDER", "ollama")
    summ_model = os.getenv("LLM_MODEL", "") or {
        "ollama": os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
        "gemini": "gemini-2.0-flash",
        "anthropic": "claude-3-5-haiku-20241022",
        "openai": "gpt-4o-mini",
    }.get(summ_provider, summ_provider)
    # Add paper by URL or ID
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Add paper by ArXiv URL or ID**")
    arxiv_input = st.sidebar.text_input(
        "ArXiv URL or ID",
        placeholder="e.g. 2301.12345 or arxiv.org/abs/2301.12345",
        label_visibility="collapsed",
        key="sidebar_arxiv_input",
    )
    if st.sidebar.button("Add paper", key="sidebar_add_paper"):
        if arxiv_input.strip():
            _ingest_by_arxiv_id(arxiv_input.strip(), in_sidebar=True)
        else:
            st.sidebar.warning("Please enter an ArXiv URL or ID.")

    st.sidebar.markdown("---")
    preferred_chat = os.getenv("PUBLIC_LLM_PROVIDER", summ_provider).strip().lower()
    public_provider = privacy.choose_provider(
        summ_provider,
        preferred_cloud_provider=preferred_chat,
        contains_private_data=False,
    )
    private_provider = privacy.choose_provider(
        summ_provider,
        preferred_cloud_provider=preferred_chat,
        contains_private_data=True,
    )
    st.sidebar.caption(f"**Summarization:** {summ_provider} / {summ_model}")
    st.sidebar.caption(f"**Paper chat:** {privacy.routing_label(public_provider, contains_private_data=False)}")
    st.sidebar.caption(f"**Notes/projects:** {privacy.routing_label(private_provider, contains_private_data=True)}")
    with st.sidebar.expander("System health", expanded=False):
        health = research_db.health_snapshot()
        st.caption(f"Database schema: v{health['schema_version']}")
        st.caption(f"Papers: {health['papers']}")
        st.caption(f"Search documents: {health['documents']} | inbox: {health['inbox']}")
        selected_policy = st.selectbox(
            "Data routing",
            ["local_only", "paper_cloud", "allow_cloud"],
            index=["local_only", "paper_cloud", "allow_cloud"].index(privacy.active_policy()),
            help="Local only keeps all model calls local; paper cloud permits public paper text in cloud models; allow cloud includes private notes and projects.",
        )
        if selected_policy != privacy.active_policy() and st.button("Apply routing policy", use_container_width=True):
            privacy.set_policy(selected_policy)
            st.rerun()
        if health["documents"] != health["fts_documents"]:
            st.error("Full-text index requires repair.")
        if st.button("Rebuild text index", key="rebuild_text_index", use_container_width=True):
            indexed = paper_store.rebuild_search_index()
            st.success(f"Rebuilt searchable records for {indexed} papers.")

    if search_clicked:
        return query, author, categories, max_results, date_from
    return None, author, categories, max_results, date_from


def _authors_str(paper: Dict, max_shown: int = 3) -> str:
    """Return a display string for authors, handling both list and string formats."""
    authors = paper.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    shown = ", ".join(authors[:max_shown])
    if len(authors) > max_shown:
        shown += f" +{len(authors) - max_shown}"
    return shown


def _paper_label(paper: Dict) -> str:
    """Compact title/ID label for selectors."""
    pid = paper.get("arxiv_id", paper.get("id", ""))
    title = paper.get("title", pid)
    return f"{title[:95]} [{pid}]"


def _notes_lines(paper: Dict) -> List[str]:
    """Return populated structured-note lines for assistant contexts."""
    notes = paper.get("research_notes", {})
    if not isinstance(notes, dict):
        return []
    labels = {
        "key_result": "Key result",
        "why_i_care": "Why I care",
        "cite_for": "Cite for",
        "caveats": "Caveats",
        "follow_up": "Follow-up questions",
    }
    lines = []
    for key, label in labels.items():
        value = str(notes.get(key, "")).strip()
        if value:
            lines.append(f"{label}: {value}")
    return lines


def _published_timestamp(value) -> pd.Timestamp:
    """Parse mixed published-date formats into a sortable UTC timestamp."""
    if value is None or value == "":
        return pd.NaT
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _published_sort_key(paper: Dict) -> float:
    """Numeric sort key for paper publication date."""
    ts = _published_timestamp(paper.get("published", ""))
    return ts.timestamp() if not pd.isna(ts) else float("-inf")


def _format_notes_for_display(paper: Dict) -> str:
    lines = _notes_lines(paper)
    return "\n".join(f"- {line}" for line in lines)


def _sanitize_paper_dict(paper: Dict) -> Dict:
    """Normalize pandas NaN values to None for downstream helpers."""
    clean: Dict = {}
    for key, value in paper.items():
        if isinstance(value, float) and math.isnan(value):
            clean[key] = None
        else:
            clean[key] = value
    return clean


def render_paper_notes(pid: str):
    """Render editable citation-aware notes for a stored paper."""
    notes = paper_store.get_notes(pid)
    with st.form(f"notes_form_{pid}", clear_on_submit=False):
        st.markdown("**Citation-aware notes**")
        key_result = st.text_area(
            "Key result",
            value=notes["key_result"],
            placeholder="The result I would quote when citing this paper...",
            height=70,
            key=f"notes_key_result_{pid}",
        )
        why_i_care = st.text_area(
            "Why I care",
            value=notes["why_i_care"],
            placeholder="How this connects to my work or project...",
            height=70,
            key=f"notes_why_{pid}",
        )
        cite_for = st.text_area(
            "Cite for",
            value=notes["cite_for"],
            placeholder="Methods, data, claim, comparison, background...",
            height=70,
            key=f"notes_cite_for_{pid}",
        )
        caveats = st.text_area(
            "Caveats",
            value=notes["caveats"],
            placeholder="Limitations, assumptions, possible failure modes...",
            height=70,
            key=f"notes_caveats_{pid}",
        )
        follow_up = st.text_area(
            "Follow-up questions",
            value=notes["follow_up"],
            placeholder="Questions to answer before relying on this paper...",
            height=70,
            key=f"notes_follow_up_{pid}",
        )
        if st.form_submit_button("Save notes"):
            paper_store.save_notes(
                pid,
                {
                    "key_result": key_result,
                    "why_i_care": why_i_care,
                    "cite_for": cite_for,
                    "caveats": caveats,
                    "follow_up": follow_up,
                },
            )
            st.session_state.papers = paper_store.load_all_papers()
            st.success("Notes saved.")

    with st.expander("Import handwritten note", expanded=False):
        upload = st.file_uploader(
            "Note image",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"handwritten_upload_{pid}",
        )
        destination = st.selectbox(
            "Save transcription to",
            list(paper_store.DEFAULT_NOTES),
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"handwritten_destination_{pid}",
        )
        if upload and st.button("Transcribe locally", key=f"handwritten_transcribe_{pid}"):
            paper = paper_store.get_paper(pid) or {}
            with st.spinner("Transcribing with the local vision model..."):
                try:
                    transcription = handwritten_notes.transcribe_image(
                        upload.getvalue(), context=f"{paper.get('title', '')}\n{paper.get('abstract', '')}"
                    )
                except Exception as exc:
                    st.error(f"Transcription failed: {exc}")
                else:
                    current = paper_store.get_notes(pid)
                    current[destination] = "\n\n".join(filter(None, [current[destination], transcription]))
                    paper_store.save_notes(pid, current)
                    corpus_index.index_text_record(
                        owner_type="handwritten_note",
                        owner_id=pid,
                        kind="handwritten_note",
                        title=paper.get("title", pid),
                        text=transcription,
                        paper_id=pid,
                        locator={"arxiv_id": pid, "filename": upload.name},
                        vector_db=_session_vector_db(),
                    )
                    st.success("Transcription saved and indexed.")
                    st.rerun()


def render_paper_card(paper: Dict, show_actions: bool = True):
    with st.expander(f"**{paper['title']}**", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {paper['title']}")
            authors = _authors_str(paper)
            st.caption(f"Authors: {authors}")
            st.caption(f"Published: {paper.get('published', 'N/A')}")
            st.caption(f"Categories: {' | '.join(paper.get('categories', []))}")
            st.markdown("**Summary**")
            st.write(paper.get("summary", "No summary available."))
            note_summary = _format_notes_for_display(paper)
            if note_summary:
                st.markdown("**Research notes**")
                st.markdown(note_summary)
        with col2:
            pid = paper.get("arxiv_id", paper.get("id", ""))
            if pid:
                st.markdown(f"[View on ArXiv](https://arxiv.org/abs/{pid})")
            if show_actions:
                pdf_url = paper.get("pdf_url", "")
                if pdf_url and st.button("Download PDF", key=f"dl_{pid}"):
                    pdf_path = st.session_state.arxiv.get_pdf_path_for_id(
                        pid,
                        title=paper.get("title"),
                        pdf_url=pdf_url,
                    )
                    st.success(f"Saved to {pdf_path}")
                if pid:
                    _render_report_controls(paper, pid, key_prefix="search")


_REPORTS_DIR = Path(__file__).parent.parent / "static" / "reports"


def _report_url(report_path: Path) -> str:
    return f"/app/static/reports/{report_path.name}"


def _report_paths(pid: str) -> tuple[Path, Path]:
    clean_id = pid.split("v")[0]
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return (
        _REPORTS_DIR / f"{clean_id.replace('/', '_')}.html",
        Path("data/sources"),
    )


def _load_report_html(report_path: Path) -> str:
    return report_path.read_text(encoding="utf-8")


def _render_report_controls(paper: Dict, pid: str, *, key_prefix: str):
    report_path, sources_dir = _report_paths(pid)
    exists = report_path.exists() and report_path.stat().st_size > 0
    st.markdown("**Detailed report**")

    if exists:
        open_col, regen_col = st.columns(2)
        open_col.link_button("Open report", url=_report_url(report_path))
        if regen_col.button("Regenerate report", key=f"{key_prefix}_report_regen_{pid}"):
            with st.spinner("Generating report..."):
                try:
                    report_path = generate_report(
                        paper,
                        summarizer=st.session_state.summarizer,
                        arxiv_client=st.session_state.arxiv,
                        pdf_extractor=st.session_state.pdf_extractor,
                        reports_dir=_REPORTS_DIR,
                        sources_dir=sources_dir,
                        force=True,
                        vector_db=_session_vector_db(),
                    )
                except ReportUnavailable as exc:
                    st.error(f"Could not generate a report for {pid}: {exc}")
                except Exception as exc:
                    st.error(f"Report generation failed for {pid}: {exc}")
                else:
                    st.success(f"Detailed report updated: {report_path}")
                    exists = True
    else:
        if st.button("Generate detailed report", key=f"{key_prefix}_report_generate_{pid}"):
            with st.spinner("Generating report..."):
                try:
                    report_path = generate_report(
                        paper,
                        summarizer=st.session_state.summarizer,
                        arxiv_client=st.session_state.arxiv,
                        pdf_extractor=st.session_state.pdf_extractor,
                        reports_dir=_REPORTS_DIR,
                        sources_dir=sources_dir,
                        vector_db=_session_vector_db(),
                    )
                except ReportUnavailable as exc:
                    st.error(f"Could not generate a report for {pid}: {exc}")
                except Exception as exc:
                    st.error(f"Report generation failed for {pid}: {exc}")
                else:
                    st.success(f"Detailed report saved to {report_path}")
                    exists = True

    if not exists:
        return

    html = _load_report_html(report_path)
    st.download_button(
        "Download report HTML",
        data=html,
        file_name=report_path.name,
        mime="text/html",
        key=f"{key_prefix}_report_download_{pid}",
    )


def _store_paper(pid: str, metadata: Dict, summary: str, paper_text: str = ""):
    """Persist a paper, update its embedding, and queue citation discovery."""
    paper_store.save_paper(pid, metadata, summary)

    def _chroma_safe(v):
        """Convert a value to a ChromaDB-safe scalar."""
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, list):
            return ", ".join(str(i) for i in v)
        # Timestamps, dates, or anything else — stringify
        return str(v)

    chroma_meta = {
        k: _chroma_safe(v) for k, v in {
            **metadata,
            "summary": summary,
            "authors": ", ".join(str(a) for a in metadata.get("authors", [])) if isinstance(metadata.get("authors"), list) else str(metadata.get("authors", "")),
            "categories": ", ".join(str(c) for c in metadata.get("categories", [])) if isinstance(metadata.get("categories"), list) else str(metadata.get("categories", "")),
        }.items()
        if _chroma_safe(v) is not None
    }
    # Use upsert pattern: delete first if exists (for regeneration), then add
    try:
        _session_vector_db().collection.delete(ids=[pid])
    except Exception:
        pass
    _session_vector_db().add_paper(
        paper_id=pid,
        title=metadata["title"],
        summary=summary,
        metadata=chroma_meta,
    )

    citation_discovery.enqueue_paper({**metadata, "summary": summary}, paper_text)


def _parse_arxiv_id(raw: str) -> str:
    """Extract a clean arxiv ID from a URL or bare ID string."""
    raw = raw.strip().rstrip("/")
    # Handle URLs like https://arxiv.org/abs/2301.12345 or arxiv.org/pdf/2301.12345v2
    for prefix in ("abs/", "pdf/", "html/", "src/"):
        if prefix in raw:
            raw = raw.split(prefix)[-1]
    # Strip version suffix for lookup but keep original
    return raw.split("v")[0] if re.match(r"^\d{4}\.\d{4,5}", raw.split("v")[0]) else raw


def _ingest_by_arxiv_id(raw_input: str, in_sidebar: bool = False):
    """Fetch, summarize, and store a paper given an ArXiv URL or ID."""
    arxiv_id = _parse_arxiv_id(raw_input)

    if paper_store.paper_exists(arxiv_id) or paper_store.paper_exists(arxiv_id + "v1"):
        if in_sidebar:
            st.sidebar.info("Paper already in library.")
        else:
            st.info("Paper already in library.")
        return

    context = st.sidebar if in_sidebar else nullcontext()
    with context:
        with st.spinner("Fetching paper..."):
            try:
                result = st.session_state.arxiv.get_result_by_id(arxiv_id)
            except Exception as exc:
                st.error(str(exc))
                return
        if result is None:
            st.error(f"Could not find paper {arxiv_id} on ArXiv.")
            return

        metadata = st.session_state.arxiv.get_paper_metadata(result)
        pid = metadata["arxiv_id"]

        with st.spinner("Loading paper text..."):
            text = _fetch_text(result, pid, st.session_state.arxiv, st.session_state.pdf_extractor)

        with st.spinner("Summarizing..."):
            summary = summarize_with_fallback(
                st.session_state.summarizer,
                text,
                metadata.get("abstract", ""),
                max_length=300,
                detailed=False,
            )
            if not summary.strip():
                st.error(f"Could not generate a summary for {pid}.")
                return

        metadata["summary"] = summary
        _store_paper(pid, metadata, summary, text)
        st.session_state.papers.append(metadata)
        st.success(f"Added: {metadata['title'][:60]}...")


def _fetch_text(result, arxiv_id: str, arxiv_client: ArxivClient, pdf_extractor: PDFExtractor) -> str:
    """Get the fullest available paper text for summarization."""
    return get_paper_text(
        arxiv_id,
        arxiv_client,
        pdf_extractor,
        result=result,
        title=result.title,
        pdf_url=result.pdf_url,
        cache_dir=Path("data/papers"),
    )


def render_search_results(query: str, author: str, categories: List[str], max_results: int, date_from: Optional[datetime]):
    with st.spinner("Searching ArXiv..."):
        try:
            papers = st.session_state.arxiv.search(
                query=query, author=author, max_results=max_results, categories=categories, date_from=date_from
            )
        except Exception as e:
            if "429" in str(e):
                st.error("ArXiv rate limit hit (HTTP 429). Wait 30-60 seconds and try again. "
                         "If this keeps happening, reduce max results or search less frequently.")
            else:
                st.error(f"ArXiv search failed: {e}")
            return

    if not papers:
        st.info("No papers found. Try different keywords, categories, or date range.")
        return

    # Split into new vs already stored
    stored_data = paper_store._load()
    all_metadata = []
    new_results = []
    for result in papers:
        metadata = st.session_state.arxiv.get_paper_metadata(result)
        pid = metadata["arxiv_id"]
        if pid in stored_data:
            metadata["summary"] = stored_data[pid].get("summary", "")
        else:
            new_results.append((result, metadata))
            metadata["summary"] = metadata.get("abstract", "")
        all_metadata.append(metadata)

    already_stored = sum(1 for metadata in all_metadata if paper_store.paper_exists(metadata["arxiv_id"]))
    st.markdown(f"### Found {len(papers)} papers ({already_stored} already in library)")

    if new_results:
        progress = st.progress(0, text="Fetching paper text...")
        texts = {}
        arxiv_client = st.session_state.arxiv
        pdf_extractor = st.session_state.pdf_extractor
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = {
                pool.submit(_fetch_text, result, meta["arxiv_id"], arxiv_client, pdf_extractor): (result, meta)
                for result, meta in new_results
            }
            for i, future in enumerate(as_completed(futures), start=1):
                _, meta = futures[future]
                try:
                    texts[meta["arxiv_id"]] = future.result()
                except Exception as exc:
                    texts[meta["arxiv_id"]] = ""
                    if "429" in str(exc):
                        st.warning(f"ArXiv rate limited full-text fetches for {meta['arxiv_id']}; using abstract fallback.")
                    else:
                        st.warning(f"Could not fetch full text for {meta['arxiv_id']}: {exc}")
                progress.progress(i / len(new_results), text=f"Fetched {i}/{len(new_results)} papers")

        progress.progress(0, text="Summarizing...")
        existing_ids = {p.get("arxiv_id", p.get("id", "")) for p in st.session_state.papers}
        added_count = 0
        for i, (_, metadata) in enumerate(new_results, start=1):
            pid = metadata["arxiv_id"]
            progress.progress(i / len(new_results), text=f"Summarizing {i}/{len(new_results)}: {metadata['title'][:50]}...")
            summary = summarize_with_fallback(
                st.session_state.summarizer,
                texts.get(pid, ""),
                metadata.get("abstract", ""),
                max_length=300,
                detailed=False,
            )
            if not summary.strip():
                st.warning(f"Skipping {pid}: no readable full text or abstract was available.")
                continue
            metadata["summary"] = summary
            _store_paper(pid, metadata, summary, texts.get(pid, ""))
            if pid not in existing_ids:
                st.session_state.papers.append(metadata)
                existing_ids.add(pid)
            added_count += 1

        progress.empty()
        if added_count:
            st.success(f"Added {added_count} new papers to your library.")

    for metadata in all_metadata:
        render_paper_card(metadata)


def render_multi_paper_chat():
    """Chat across multiple selected papers simultaneously."""
    st.markdown("### Multi-paper chat")
    all_papers = paper_store.load_all_papers()
    if not all_papers:
        st.info("No papers in your library yet.")
        return

    options = {p.get("title", p.get("arxiv_id", "")): p for p in all_papers}
    selected_titles = st.multiselect(
        "Select papers to chat with:",
        options=list(options.keys()),
        key="multi_chat_selection",
    )

    if not selected_titles:
        return

    selected_papers = [options[t] for t in selected_titles]

    if st.button("Clear multi-paper chat", key="clear_multi_chat"):
        st.session_state.pop("multi_chat_history", None)
        st.session_state.pop("multi_chat_context", None)

    # Build combined context once (cached in session state)
    context_key = "multi_chat_context"
    context_ids = tuple(p.get("arxiv_id", "") for p in selected_papers)
    if st.session_state.get(f"{context_key}_ids") != context_ids:
        with st.spinner("Loading paper texts..."):
            parts = []
            public_provider = privacy.choose_provider(
                st.session_state.summarizer._active_provider(),
                preferred_cloud_provider=os.getenv(
                    "PUBLIC_LLM_PROVIDER", st.session_state.summarizer._active_provider()
                ),
                contains_private_data=False,
            )
            for p in selected_papers:
                pid = p.get("arxiv_id", "")
                text = _get_paper_text(pid, p)
                parts.append(_paper_source_context(pid, p, text, max_chunks=None if public_provider == "gemini" else 4))
            st.session_state[context_key] = "\n\n".join(parts)
            st.session_state[f"{context_key}_ids"] = context_ids

    system_prompt = (
        "You are a research assistant. You have read the following academic paper sources. "
        "Answer accurately using the supplied source chunks. Cite evidence for substantive claims using labels "
        "like [arXiv:2301.12345 chunk 2]. If something is not covered in the sources, say so.\n\n"
        f"{st.session_state[context_key]}\n\n"
        "When comparing papers, be specific about which paper you are referring to."
    )

    history = st.session_state.get("multi_chat_history", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a question across these papers...", key="multi_chat_input")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                messages = history + [{"role": "user", "content": user_input}]
                try:
                    reply = st.session_state.summarizer.dispatch_chat_gemini(
                        system_prompt,
                        messages,
                        contains_private_data=any(_notes_lines(paper) for paper in selected_papers),
                    )
                except Exception as e:
                    reply = f"Error: {e}"
            st.write(reply)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        st.session_state["multi_chat_history"] = history


def render_vector_search():
    st.header("Unified Research Search")
    semantic_query = st.text_input(
        "Describe what you're looking for...",
        placeholder="e.g., papers about gravitational wave detection methods",
    )
    if semantic_query:
        with st.spinner("Searching..."):
            results = retrieval.hybrid_search(semantic_query, vector_db=_session_vector_db(), limit=15)
        if results:
            st.markdown(f"### Found {len(results)} attributable sources")
            for result in results:
                label = retrieval.citation_label(result)
                title = result.get("title") or result.get("kind", "Source").replace("_", " ").title()
                with st.expander(f"{title} - {label}"):
                    st.caption(result.get("kind", "source").replace("_", " ").title())
                    st.write(result.get("text", ""))
                    url = retrieval.source_url(result)
                    if url:
                        st.link_button("Open source", url)
        else:
            st.info("No indexed sources matched. Check the search index in Library Health.")

    st.markdown("---")
    render_multi_paper_chat()


def _all_saved_ideas() -> list[Dict]:
    return idea_store.load_ideas("paper") + idea_store.load_ideas("grant")


def _rescore_inbox() -> int:
    inbox = paper_store.load_triage(status="inbox", limit=500)
    all_triage = paper_store.load_triage(status=None, limit=5000)
    positive = [paper for paper in all_triage if paper.get("triage", {}).get("feedback") == 1]
    negative = [paper for paper in all_triage if paper.get("triage", {}).get("feedback") == -1]
    interest_text = relevance.build_interest_text(researcher_profile.load(), _all_saved_ideas())
    scored = relevance.score_papers(
        inbox,
        interest_text=interest_text,
        positive_papers=positive,
        negative_papers=negative,
    )
    for paper in scored:
        paper_store.update_triage(
            paper.get("arxiv_id", ""),
            relevance_score=paper["relevance_score"],
            relevance_reason=paper["relevance_reason"],
        )
    return len(scored)


def render_inbox():
    """Render the daily paper triage and relevance-feedback workflow."""
    st.header("Research Inbox")
    status_labels = {
        "inbox": "Inbox",
        "read_later": "Read later",
        "skimmed": "Skimmed",
        "read": "Read",
        "saved": "Library",
        "irrelevant": "Irrelevant",
    }
    ctrl1, ctrl2 = st.columns([2, 1])
    selected_label = ctrl1.selectbox("View", list(status_labels.values()), key="inbox_status")
    selected_status = next(key for key, label in status_labels.items() if label == selected_label)
    if ctrl2.button("Recalculate relevance", use_container_width=True):
        count = _rescore_inbox()
        st.success(f"Scored {count} inbox papers against your profile, projects, and feedback.")

    papers = paper_store.load_triage(status=selected_status, limit=100)
    if not papers:
        st.info("No papers in this view.")
        return

    st.caption(f"{len(papers)} papers shown. New scheduled papers arrive in Inbox.")
    for paper in papers:
        pid = paper.get("arxiv_id", "")
        triage = paper.get("triage", {})
        score = triage.get("relevance_score")
        score_label = f"{score:.0f}%" if isinstance(score, (int, float)) else "unscored"
        with st.expander(f"{score_label} - {paper.get('title', pid)}", expanded=False):
            st.caption(f"{_authors_str(paper)} | {str(paper.get('published', ''))[:10]} | {pid}")
            if triage.get("relevance_reason"):
                st.caption(triage["relevance_reason"])
            st.write(paper.get("summary") or paper.get("abstract") or "No summary available.")
            action_cols = st.columns(6)
            actions = [
                ("Read later", "read_later", None),
                ("Skimmed", "skimmed", None),
                ("Read", "read", None),
                ("Save", "saved", 1),
                ("Relevant", None, 1),
                ("Irrelevant", "irrelevant", -1),
            ]
            for column, (label, new_status, feedback) in zip(action_cols, actions):
                if column.button(label, key=f"triage_{label}_{pid}", use_container_width=True):
                    paper_store.update_triage(pid, status=new_status, feedback=feedback)
                    st.rerun()


def render_knowledge_graph():
    st.header("Knowledge Graph")
    papers = paper_store.load_all_papers()
    if len(papers) < 2:
        st.info("Add more papers to see connections.")
        return

    df = pd.DataFrame(papers)
    all_cats = sorted(df["categories"].explode().dropna().unique().tolist()) if "categories" in df.columns else []

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    max_nodes = ctrl1.slider("Papers shown", 2, min(100, len(papers)), min(40, len(papers)))
    selected_cats = ctrl2.multiselect("Filter categories", options=all_cats, default=[])
    edge_types = ctrl3.multiselect(
        "Connection types",
        ["shared category", "shared author", "semantic similarity"],
        default=["shared category", "shared author", "semantic similarity"],
    )
    sim_threshold = st.slider("Minimum semantic similarity", 0.10, 0.95, 0.55, 0.05)

    filtered = papers
    if selected_cats:
        filtered = [
            p for p in papers
            if any(cat in (p.get("categories", []) if isinstance(p.get("categories"), list) else []) for cat in selected_cats)
        ]
    if not filtered:
        st.info("No papers match the selected graph filters.")
        return

    filtered = sorted(filtered, key=_published_sort_key, reverse=True)[:max_nodes]
    paper_by_id = {p.get("arxiv_id", p.get("id", "")): p for p in filtered if p.get("arxiv_id", p.get("id", ""))}

    G = nx.Graph()
    for pid, paper in paper_by_id.items():
        G.add_node(pid, title=paper.get("title", pid), paper=paper)

    def add_edge(a: str, b: str, reason: str, weight: float = 1.0):
        if a == b or a not in G or b not in G:
            return
        if G.has_edge(a, b):
            G[a][b]["reasons"].append(reason)
            G[a][b]["weight"] = max(G[a][b]["weight"], weight)
        else:
            G.add_edge(a, b, reasons=[reason], weight=weight)

    ids = list(paper_by_id.keys())
    if "shared category" in edge_types:
        for i, pid in enumerate(ids):
            cats = set(paper_by_id[pid].get("categories", []) or [])
            for other in ids[i + 1:]:
                shared = cats.intersection(set(paper_by_id[other].get("categories", []) or []))
                if shared:
                    add_edge(pid, other, "category: " + ", ".join(sorted(shared)[:3]), weight=0.7)

    if "shared author" in edge_types:
        for i, pid in enumerate(ids):
            authors = set(paper_by_id[pid].get("authors", []) or [])
            for other in ids[i + 1:]:
                shared = authors.intersection(set(paper_by_id[other].get("authors", []) or []))
                if shared:
                    add_edge(pid, other, "author: " + ", ".join(sorted(shared)[:3]), weight=1.0)

    if "semantic similarity" in edge_types:
        with st.spinner("Adding semantic similarity edges..."):
            for pid in ids:
                vec = _session_vector_db().get_embedding(pid)
                if vec is None:
                    continue
                for result in _session_vector_db().search_by_vector(vec, top_k=5, exclude_id=pid):
                    other = result["id"]
                    if other not in paper_by_id:
                        continue
                    score = 1 - result.get("distance", 1)
                    if score >= sim_threshold:
                        add_edge(pid, other, f"semantic similarity: {score:.2f}", weight=score)

    if G.number_of_edges() == 0:
        st.info("No connections found with the selected filters.")
        return

    pos = nx.spring_layout(G, seed=7, weight="weight", k=0.75)
    edge_x, edge_y = [], []
    for source, target in G.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.8, color="#9aa0a6"),
        hoverinfo="none",
        mode="lines",
    )

    degrees = dict(G.degree())
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        paper = paper_by_id[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(14 + degrees[node] * 3)
        node_color.append(degrees[node])
        node_text.append(
            f"<b>{paper.get('title', node)}</b><br>"
            f"{_authors_str(paper, max_shown=4)}<br>"
            f"{node}<br>"
            f"Connections: {degrees[node]}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_color,
            size=node_size,
            colorbar=dict(title="Degree"),
            line_width=1,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            height=650,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Papers shown", G.number_of_nodes())
    col2.metric("Connections", G.number_of_edges())
    col3.metric("Avg connections", f"{2 * G.number_of_edges() / G.number_of_nodes():.1f}")

    edge_rows = []
    for source, target, data in G.edges(data=True):
        edge_rows.append({
            "source": paper_by_id[source].get("title", source),
            "target": paper_by_id[target].get("title", target),
            "reason": "; ".join(data.get("reasons", [])),
            "source_arxiv": f"https://arxiv.org/abs/{source}",
            "target_arxiv": f"https://arxiv.org/abs/{target}",
        })
    with st.expander("Connection table", expanded=False):
        st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)


def _get_paper_text(pid: str, paper: Dict) -> str:
    """Get full paper text for chat without doing another metadata lookup."""
    return get_paper_text(
        pid,
        st.session_state.arxiv,
        st.session_state.pdf_extractor,
        title=paper.get("title"),
        pdf_url=paper.get("pdf_url"),
        cache_dir=Path("data/papers"),
    )


def _source_chunks(text: str, source_label: str, chunk_chars: int = 3500, max_chunks: Optional[int] = None) -> str:
    """Add stable source labels to paper text so chat answers can cite evidence."""
    cleaned = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not cleaned:
        return f"[{source_label} chunk 1]\nNo paper text was available."

    chunks = []
    for idx, start in enumerate(range(0, len(cleaned), chunk_chars), start=1):
        if max_chunks is not None and idx > max_chunks:
            chunks.append(f"[{source_label} omitted]\nAdditional text omitted because of context limits.")
            break
        chunk = cleaned[start:start + chunk_chars]
        chunks.append(f"[{source_label} chunk {idx}]\n{chunk}")
    return "\n\n".join(chunks)


def _paper_source_context(pid: str, paper: Dict, text: str, max_chunks: Optional[int] = None) -> str:
    """Build citation-ready context for one paper."""
    notes = "\n".join(_notes_lines(paper))
    source_label = f"arXiv:{pid}"
    return (
        f"=== Paper source: {source_label} ===\n"
        f"Title: {paper.get('title', pid)}\n"
        f"Authors: {_authors_str(paper)}\n"
        f"Published: {paper.get('published', '')}\n"
        + (f"Research notes:\n{notes}\n" if notes else "")
        + "\n"
        + _source_chunks(text, source_label, max_chunks=max_chunks)
    )


def _chat_with_paper(pid: str, paper: Dict, user_message: str) -> str:
    """Send a message to the configured LLM provider with the paper in context."""
    source_key = f"chat_text_source_{pid}"
    try:
        corpus_index.index_paper(
            paper,
            arxiv_client=st.session_state.arxiv,
            pdf_extractor=st.session_state.pdf_extractor,
            vector_db=_session_vector_db(),
        )
        evidence = retrieval.hybrid_search(
            user_message,
            vector_db=_session_vector_db(),
            paper_ids=[pid],
            limit=8,
        )
    except Exception as exc:
        print(f"[retrieval] could not build/query evidence index for {pid}: {exc}")
        evidence = []

    if evidence:
        source_context = retrieval.context_block(evidence)
        st.session_state[source_key] = "hybrid page/section retrieval"
    else:
        paper_text = _get_paper_text(pid, paper)
        source_context = _paper_source_context(pid, paper, paper_text, max_chunks=10)
        st.session_state[source_key] = "full-text fallback"

    system_prompt = (
        "You are a research assistant. Answer using only the retrieved evidence unless the user explicitly asks "
        "for outside knowledge. Every substantive factual claim must cite one or more supplied labels exactly. "
        "If the evidence does not support an answer, say that clearly. End with a short Evidence section containing "
        "the most important quoted passages.\n\n"
        + source_context
    )

    history_key = f"chat_history_{pid}"
    history = st.session_state.get(history_key, [])
    messages = history + [{"role": "user", "content": user_message}]

    try:
        reply = st.session_state.summarizer.dispatch_chat_gemini(
            system_prompt,
            messages,
            contains_private_data=any(item.get("kind") == "notes" for item in evidence),
        )
    except Exception as e:
        reply = f"Error: {e}"
    if evidence:
        reply = retrieval.linkify_citations(reply, evidence)

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    st.session_state[history_key] = history

    return reply


def render_paper_chat(pid: str, paper: Dict):
    """Render an inline chat panel for a paper."""
    history_key = f"chat_history_{pid}"
    history = st.session_state.get(history_key, [])

    source = st.session_state.get(f"chat_text_source_{pid}", "")
    provider = privacy.choose_provider(
        st.session_state.summarizer._active_provider(),
        preferred_cloud_provider=os.getenv(
            "PUBLIC_LLM_PROVIDER", st.session_state.summarizer._active_provider()
        ),
        contains_private_data=bool(_notes_lines(paper)),
    )
    st.markdown("**Chat with this paper**")
    st.caption(f"Evidence: {source or 'indexed on first question'} | route: {provider}")

    # Show conversation history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    user_input = st.chat_input("Ask a question about this paper...", key=f"chat_input_{pid}")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = _chat_with_paper(pid, paper, user_input)
            st.write(reply)

    if history and st.button("Clear chat", key=f"clear_chat_{pid}"):
        st.session_state[history_key] = []
        st.session_state.pop(f"chat_text_{pid}", None)


def _regenerate_summary(paper: Dict, detailed: bool = False):
    """Re-summarize a single paper and update all stores."""
    pid = paper.get("arxiv_id", paper.get("id", ""))
    try:
        text = get_paper_text(
            pid,
            st.session_state.arxiv,
            st.session_state.pdf_extractor,
            title=paper.get("title"),
            pdf_url=paper.get("pdf_url"),
            cache_dir=Path("data/papers"),
        )
    except Exception as exc:
        text = ""
        if "429" in str(exc):
            st.warning(f"ArXiv rate limited paper fetches for {pid}; using the abstract as a fallback where possible.")
        else:
            st.warning(f"Could not fetch full paper text for {pid}: {exc}")

    summary = summarize_with_fallback(
        st.session_state.summarizer,
        text,
        paper.get("abstract", ""),
        max_length=500 if detailed else 300,
        detailed=detailed,
    )
    if not summary.strip():
        st.error(f"Could not generate a summary for {pid}. No readable full text or abstract was available.")
        return None
    _store_paper(pid, paper, summary)

    # Update session state papers list
    for p in st.session_state.papers:
        if p.get("arxiv_id") == pid:
            p["summary"] = summary
            break
    return summary


def render_papers_list():
    _sync_papers_from_store()
    st.header("All Papers")
    papers = st.session_state.papers
    if not papers:
        st.info("No papers stored yet. Use the search to add papers.")
        return

    df = pd.DataFrame(papers)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Papers", len(df))
    col2.metric("Categories", df["categories"].explode().nunique() if "categories" in df.columns else 0)
    col3.metric("Last saved", _format_file_mtime(paper_store.STORE_PATH))

    latest = _recent_stored_papers(limit=1)
    if latest:
        newest = latest[0]
        col4.metric("Newest paper", newest.get("published", "N/A"))
        st.caption(f"Latest stored: {newest.get('arxiv_id', '')} - {newest.get('title', '')}")
    else:
        col4.metric("Newest paper", "N/A")

    # Bulk summary buttons
    btn_col1, btn_col2, btn_col3, btn_col4, mode_col = st.columns([1, 1, 1, 1, 2])
    detailed_all = mode_col.checkbox("Detailed mode", key="regen_detailed_all")
    if btn_col1.button("Regenerate all"):
        progress = st.progress(0, text="Regenerating summaries...")
        all_papers = paper_store.load_all_papers()
        for i, p in enumerate(all_papers):
            progress.progress((i + 1) / len(all_papers), text=f"Summarizing {i+1}/{len(all_papers)}: {p.get('title','')[:50]}...")
            _regenerate_summary(p, detailed=detailed_all)
        progress.empty()
        st.session_state.papers = paper_store.load_all_papers()
        st.success("All summaries regenerated.")
        st.rerun()
    if btn_col2.button("Fill missing"):
        missing = [p for p in paper_store.load_all_papers() if not p.get("summary", "").strip()]
        if not missing:
            st.info("No missing summaries.")
        else:
            progress = st.progress(0, text="Filling missing summaries...")
            for i, p in enumerate(missing):
                progress.progress((i + 1) / len(missing), text=f"Summarizing {i+1}/{len(missing)}: {p.get('title','')[:50]}...")
                _regenerate_summary(p, detailed=detailed_all)
            progress.empty()
            st.session_state.papers = paper_store.load_all_papers()
            st.success(f"Filled {len(missing)} missing summaries.")
            st.rerun()
    if btn_col3.button("Refresh"):
        st.session_state.papers = paper_store.load_all_papers()
        st.rerun()
    if btn_col4.button("Repair incomplete"):
        incomplete = [p for p in paper_store.load_all_papers() if completeness_issues(p.get("summary", ""))]
        progress = st.progress(0, text="Repairing incomplete summaries...")
        for i, paper in enumerate(incomplete, start=1):
            progress.progress(i / max(len(incomplete), 1), text=f"Repairing {i}/{len(incomplete)}")
            _regenerate_summary(paper, detailed=True)
        progress.empty()
        st.session_state.papers = paper_store.load_all_papers()
        st.success(f"Rebuilt {len(incomplete)} summaries from full text.")
        st.rerun()

    st.markdown("---")

    # Search and filter controls
    search_col1, search_col2 = st.columns(2)
    text_query = search_col1.text_input("Search titles & summaries", placeholder="e.g., gravitational waves", key="lib_text_search")
    author_query = search_col2.text_input("Search by author", placeholder="e.g., Smith", key="lib_author_search")

    filter_col1, filter_col2 = st.columns(2)
    if "categories" in df.columns:
        all_cats = sorted(df["categories"].explode().dropna().unique().tolist())
        selected_cats = filter_col1.multiselect("Filter by category:", options=all_cats, default=[])
    else:
        selected_cats = []

    sort_by = filter_col2.selectbox("Sort by:", ["Date (newest)", "Date (oldest)", "Title (A–Z)", "Title (Z–A)"])

    # Apply text search (overrides df if query given)
    if text_query.strip():
        matches = paper_store.search_by_text(text_query.strip())
        df = pd.DataFrame(matches) if matches else pd.DataFrame()
    if author_query.strip():
        matches = paper_store.search_by_author(author_query.strip())
        df = pd.DataFrame(matches) if matches else pd.DataFrame()

    # Apply category filter
    if selected_cats and not df.empty and "categories" in df.columns:
        mask = df["categories"].apply(lambda x: any(c in (x if isinstance(x, list) else []) for c in selected_cats))
        df = df[mask]

    # Apply sort
    if not df.empty and "published" in df.columns:
        df = df.copy()
        df["_pub_sort"] = df["published"].apply(_published_timestamp)
        if sort_by == "Date (newest)":
            df = df.sort_values("_pub_sort", ascending=False)
        elif sort_by == "Date (oldest)":
            df = df.sort_values("_pub_sort", ascending=True)
        elif sort_by == "Title (A–Z)":
            df = df.sort_values("title", ascending=True)
        elif sort_by == "Title (Z–A)":
            df = df.sort_values("title", ascending=False)

    if df.empty:
        st.info("No papers match the current filters.")
        return

    st.caption(f"Showing {min(len(df), 50)} of {len(df)} papers")

    # Per-paper cards with regenerate button
    # Keep a live summary cache in session state so regeneration shows immediately
    if "live_summaries" not in st.session_state:
        st.session_state.live_summaries = {}

    for _, row in df.head(50).iterrows():
        paper = _sanitize_paper_dict(row.to_dict())
        pid = paper.get("arxiv_id", "")
        # Use live summary if we just regenerated it this session
        displayed_summary = st.session_state.live_summaries.get(pid, paper.get("summary", "No summary available."))
        with st.expander(f"**{paper.get('title', pid)}**", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"Authors: {_authors_str(paper)}")
                st.caption(f"Published: {paper.get('published', 'N/A')}")
                st.markdown("**Summary**")
                st.write(displayed_summary)
                issues = completeness_issues(displayed_summary)
                if issues:
                    st.warning("Summary quality check: " + ", ".join(issues) + ". Regenerate in Detailed mode.")
                notes_text = _format_notes_for_display(paper)
                if notes_text:
                    st.markdown("**Saved research notes**")
                    st.markdown(notes_text)
            with col2:
                if pid:
                    st.markdown(f"[View on ArXiv](https://arxiv.org/abs/{pid})")
                detailed = st.checkbox("Detailed", key=f"det_{pid}")
                if st.button("Regenerate summary", key=f"regen_{pid}"):
                    with st.spinner("Summarizing..."):
                        new_summary = _regenerate_summary(paper, detailed=detailed)
                    if new_summary:
                        st.session_state.live_summaries[pid] = new_summary
                        st.session_state.papers = paper_store.load_all_papers()
                        st.success("Done. Summary updated.")
                        st.rerun()
                show_chat_key = f"show_chat_{pid}"
                if st.button("Chat with paper", key=f"chat_btn_{pid}"):
                    st.session_state[show_chat_key] = not st.session_state.get(show_chat_key, False)
                if st.button("More like this", key=f"mlt_{pid}"):
                    st.session_state[f"show_mlt_{pid}"] = not st.session_state.get(f"show_mlt_{pid}", False)
                # Delete with confirmation
                confirm_key = f"confirm_delete_{pid}"
                if not st.session_state.get(confirm_key, False):
                    if st.button("Delete", key=f"del_{pid}", type="secondary"):
                        st.session_state[confirm_key] = True
                else:
                    st.warning("Delete this paper?")
                    dcol1, dcol2 = st.columns(2)
                    if dcol1.button("Yes, delete", key=f"del_confirm_{pid}", type="primary"):
                        paper_store.delete_paper(pid)
                        _session_vector_db().delete_paper(pid)
                        st.session_state.papers = [
                            p for p in st.session_state.papers
                            if p.get("arxiv_id") != pid
                        ]
                        st.session_state.pop(confirm_key, None)
                        st.success("Deleted.")
                    if dcol2.button("Cancel", key=f"del_cancel_{pid}"):
                        st.session_state[confirm_key] = False
                if pid:
                    _render_report_controls(paper, pid, key_prefix="library")
            if st.session_state.get(f"show_mlt_{pid}", False):
                vec = _session_vector_db().get_embedding(pid)
                if vec is not None:
                    similar = _session_vector_db().search_by_vector(vec, top_k=5, exclude_id=pid)
                    if similar:
                        st.markdown("**Similar papers in your library:**")
                        for s in similar:
                            m = s["metadata"]
                            title = m.get("title", s["id"])
                            sid = s["id"]
                            score = 1 - s["distance"]
                            st.markdown(f"- **{title}** — similarity {score:.2f}  \n"
                                        f"  [{sid}](https://arxiv.org/abs/{sid})")
                    else:
                        st.info("No similar papers found in your library yet.")
                else:
                    st.info("Embedding not available for this paper.")
            if st.session_state.get(f"show_chat_{pid}", False):
                st.markdown("---")
                render_paper_chat(pid, paper)
            st.markdown("---")
            render_paper_notes(pid)

    st.markdown("---")
    data_dir = Path("data").resolve()
    st.caption(f"Data: {data_dir}  |  PDFs: papers/  |  Vector DB: vector_db/  |  Metadata: papers.json")


_CRON_BEGIN = "# BEGIN NSArxivApp managed fetch"
_CRON_END = "# END NSArxivApp managed fetch"


def _build_fetch_command(
    *,
    app_dir: Path,
    python_path: str,
    mode: str,
    query: str,
    categories: List[str],
    max_results: int,
    days_back: int,
) -> str:
    args = [
        python_path,
        "-m",
        "app.fetch_job",
        "--mode",
        mode,
        "--max-results",
        str(max_results),
        "--days-back",
        str(days_back),
    ]
    if query.strip():
        args.extend(["--query", query.strip()])
    if categories:
        args.append("--categories")
        args.extend(categories)
    return f"cd {shlex.quote(str(app_dir))} && {shlex.join(args)}"


def _install_cron_job(cron_line: str) -> tuple[bool, str]:
    try:
        current = subprocess.run(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, "crontab is not installed on this system."

    stderr = (current.stderr or "").strip().lower()
    if current.returncode not in (0, 1):
        return False, current.stderr.strip() or current.stdout.strip() or "Could not read current crontab."
    if current.returncode == 1 and stderr and "no crontab" not in stderr:
        return False, current.stderr.strip()

    lines = current.stdout.splitlines() if current.returncode == 0 else []
    filtered: list[str] = []
    in_block = False
    for line in lines:
        if line.strip() == _CRON_BEGIN:
            in_block = True
            continue
        if line.strip() == _CRON_END:
            in_block = False
            continue
        if not in_block:
            filtered.append(line)
    while filtered and not filtered[-1].strip():
        filtered.pop()
    filtered.extend(["", _CRON_BEGIN, cron_line, _CRON_END, ""])

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            tmp.write("\n".join(filtered))
            tmp_path = tmp.name
        installed = subprocess.run(
            ["crontab", tmp_path],
            capture_output=True,
            text=True,
        )
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

    if installed.returncode != 0:
        return False, installed.stderr.strip() or installed.stdout.strip() or "Could not install cron job."
    return True, "Installed managed NSArxivApp cron job."


def _run_fetch_command(fetch_cmd: str, log_path: Path) -> tuple[bool, str]:
    """Run the fetch job immediately and append output to the fetch log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["/bin/sh", "-lc", fetch_cmd],
        capture_output=True,
        text=True,
    )
    combined = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part).strip()
    if combined:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(combined)
            if not combined.endswith("\n"):
                handle.write("\n")
    if result.returncode != 0:
        return False, combined or "Fetch command failed."
    return True, combined or "Fetch completed."


def render_schedule():
    """Show cron/launchd setup for daily automated fetch."""
    st.header("Scheduled Daily Fetch")
    st.markdown(
        "Set up an automated daily fetch that runs even when the app is closed. "
        "Results will be added to your local library and available next time you open the app."
    )

    col1, col2 = st.columns(2)
    with col1:
        schedule_mode = st.radio(
            "Scheduled fetch mode",
            ["New submissions (ArXivSelaa-style)", "Keyword search"],
            index=0,
        )
        mode = "new-submissions" if schedule_mode.startswith("New submissions") else "query-search"
        sched_query = ""
        if mode == "query-search":
            sched_query = st.text_input("Search query for scheduled job", placeholder="neutron star kilonova")
        sched_cats = st.multiselect(
            "Categories",
            ["cs.LG", "cs.CL", "cs.CV", "cs.AI", "astro-ph.HE", "astro-ph.SR", "astro-ph.CO", "astro-ph.GA", "physics.hep-th", "gr-qc"],
            default=st.session_state.get("sidebar_default_cats", ["astro-ph.HE", "astro-ph.SR", "gr-qc"]),
        )
        sched_max = st.number_input(
            "Max results per run",
            min_value=0,
            max_value=500,
            value=0 if mode == "new-submissions" else 20,
            step=10,
            help="Use 0 to ingest every matching announcement. A positive value keeps only the first N papers.",
        )
        sched_days_back = st.slider(
            "Days back",
            0,
            7,
            1,
            help="In new-submissions mode, 1 starts from the previous UTC announcement day and backs up to the latest non-empty date. In keyword mode, it searches papers submitted within the last N days.",
        )
        sched_hour = st.slider("Run at hour (24h, local time)", 0, 23, 7)

    with col2:
        app_dir = Path(".").resolve()
        python_path = sys.executable or "python"
        fetch_cmd = _build_fetch_command(
            app_dir=app_dir,
            python_path=python_path,
            mode=mode,
            query=sched_query,
            categories=sched_cats,
            max_results=sched_max,
            days_back=sched_days_back,
        )
        log_path = app_dir / "data" / "fetch.log"
        plist_command = html_escape(f"{fetch_cmd} >> {log_path} 2>&1")

        st.markdown("**Current scheduled-fetch status**")
        status_col1, status_col2 = st.columns(2)
        status_col1.metric("Fetch log updated", _format_file_mtime(log_path))
        status_col2.metric("Library updated", _format_file_mtime(paper_store.STORE_PATH))
        recent = _recent_stored_papers(limit=5)
        if recent:
            st.caption(
                "Recent stored papers: "
                + "; ".join(
                    f"{p.get('arxiv_id', '')} ({p.get('published', 'N/A')})"
                    for p in recent
                )
            )
        latest_block = _latest_fetch_block(log_path)
        with st.expander("Latest fetch log", expanded=False):
            if latest_block:
                st.code(latest_block, language="text")
            else:
                st.info("No fetch log found yet.")

        st.markdown("**cron entry** (paste into `crontab -e`)")
        cron_line = f"0 {sched_hour} * * * {fetch_cmd} >> {shlex.quote(str(log_path))} 2>&1"
        st.code(cron_line, language="bash")
        if mode == "new-submissions" and not sched_cats:
            st.warning("Choose at least one category for ArXivSelaa-style new-submissions mode.")
        else:
            action_col1, action_col2 = st.columns(2)
            if action_col1.button("Run fetch now"):
                with st.spinner("Running fetch job..."):
                    ok, output = _run_fetch_command(fetch_cmd, log_path)
                if ok:
                    st.session_state.papers = paper_store.load_all_papers()
                    st.success("Fetch completed and library reloaded.")
                else:
                    st.error("Fetch failed.")
                if output:
                    st.code(output, language="text")
            if action_col2.button("Install cron job (Linux)"):
                ok, message = _install_cron_job(cron_line)
                if ok:
                    st.success(f"Installed! Managed cron job will run daily at {sched_hour:02d}:00.")
                    st.caption(message)
                else:
                    st.error(message)

        plist_label = "com.nsarxivapp.dailyfetch"
        plist = textwrap.dedent(f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
              "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0">
            <dict>
              <key>Label</key>
              <string>{plist_label}</string>
              <key>ProgramArguments</key>
              <array>
                <string>/bin/sh</string>
                <string>-c</string>
                <string>{plist_command}</string>
              </array>
              <key>StartCalendarInterval</key>
              <dict>
                <key>Hour</key>
                <integer>{sched_hour}</integer>
                <key>Minute</key>
                <integer>0</integer>
              </dict>
              <key>WorkingDirectory</key>
              <string>{app_dir}</string>
            </dict>
            </plist>
        """)
        plist_path = Path.home() / "Library/LaunchAgents" / f"{plist_label}.plist"
        st.markdown("**macOS launchd plist**")
        st.code(plist, language="xml")

        if st.button("Install launchd agent (macOS)"):
            if mode == "new-submissions" and not sched_cats:
                st.error("Choose at least one category before installing the launchd agent.")
                return
            plist_path.parent.mkdir(parents=True, exist_ok=True)
            plist_path.write_text(plist)
            result = subprocess.run(
                ["launchctl", "load", str(plist_path)], capture_output=True, text=True
            )
            if result.returncode == 0:
                st.success(f"Installed! Agent will run daily at {sched_hour:02d}:00.")
                st.caption(f"Plist written to {plist_path}")
            else:
                st.error(f"launchctl error: {result.stderr}")
                st.caption("You can install manually: launchctl load " + str(plist_path))


def _build_selected_papers_context(papers: List[Dict], include_notes: bool = True) -> str:
    """Build context from an explicit set of papers for lit reviews and project workspaces."""
    lines = [f"Selected paper set ({len(papers)} papers):\n"]
    for p in papers:
        pid = p.get("arxiv_id", p.get("id", ""))
        notes = "\n".join(_notes_lines(p)) if include_notes else ""
        lines.append(
            f"---\n"
            f"ID: {pid}\n"
            f"Title: {p.get('title', pid)}\n"
            f"Authors: {_authors_str(p, max_shown=8)}\n"
            f"Published: {str(p.get('published', ''))[:10]}\n"
            f"Categories: {', '.join(p.get('categories', [])) if isinstance(p.get('categories'), list) else p.get('categories', '')}\n"
            f"Summary: {p.get('summary', '')}\n"
            + (f"Research notes:\n{notes}\n" if notes else "")
        )
    return "\n".join(lines)


def _assistant_call(system: str, messages: list, *, contains_private_data: bool = True) -> str:
    """Call the best provider allowed by the active data-routing policy."""
    try:
        return st.session_state.summarizer.dispatch_chat_gemini(
            system,
            messages,
            contains_private_data=contains_private_data,
        )
    except Exception as e:
        return f"Error: {e}"


def _retrieved_library_context(query: str, *, paper_ids: list[str] | None = None, limit: int = 16) -> tuple[str, list[dict]]:
    """Retrieve bounded, attributable context rather than serializing the full library."""
    evidence = retrieval.hybrid_search(
        query or "main contribution methods results limitations",
        vector_db=_session_vector_db(),
        paper_ids=paper_ids,
        limit=limit,
    )
    return retrieval.context_block(evidence), evidence


def render_lit_review_builder():
    st.header("Literature Review Builder")
    papers = paper_store.load_all_papers()
    if not papers:
        st.info("Your library is empty. Search for papers first.")
        return

    options = {_paper_label(p): p for p in sorted(papers, key=_published_sort_key, reverse=True)}
    selected_labels = st.multiselect(
        "Papers to include",
        options=list(options.keys()),
        default=list(options.keys())[: min(6, len(options))],
        key="lit_review_selected_papers",
    )
    selected = [options[label] for label in selected_labels]
    if not selected:
        st.info("Select at least one paper.")
        return

    col1, col2 = st.columns(2)
    focus = col1.text_input(
        "Review focus",
        placeholder="e.g., kilonova opacity systematics, multimessenger constraints",
        key="lit_review_focus",
    )
    review_type = col2.selectbox(
        "Output",
        [
            "Structured literature review",
            "Citation map",
            "Introduction draft",
            "Related work section",
            "Reading synthesis",
        ],
        key="lit_review_type",
    )
    include_notes = st.checkbox("Use my citation-aware notes", value=True, key="lit_review_include_notes")

    evidence_query = focus.strip() or "main claims methods results limitations"
    selected_ids = [paper.get("arxiv_id", "") for paper in selected]
    matrix = synthesis.claim_matrix(
        evidence_query,
        vector_db=_session_vector_db(),
        paper_ids=selected_ids,
        limit=24,
    )
    if matrix:
        st.markdown("**Evidence matrix**")
        st.dataframe(pd.DataFrame(matrix), use_container_width=True, hide_index=True)
        st.download_button(
            "Download claim matrix CSV",
            pd.DataFrame(matrix).to_csv(index=False),
            file_name="claim_matrix.csv",
            mime="text/csv",
        )
    st.download_button(
        "Download BibTeX",
        synthesis.bibtex(selected),
        file_name="selected_papers.bib",
        mime="application/x-bibtex",
    )

    if st.button("Generate literature review", type="primary", key="generate_lit_review"):
        profile_ctx = researcher_profile.to_context_string(researcher_profile.load())
        paper_ctx, evidence = _retrieved_library_context(evidence_query, paper_ids=selected_ids, limit=24)
        if include_notes:
            paper_ctx += "\n\n" + _build_selected_papers_context(selected, include_notes=True)
        focus_clause = f"Focus specifically on: {focus.strip()}." if focus.strip() else "Use the strongest common themes in the selected papers."
        system = (
            "You are a research assistant helping write accurate, useful literature reviews for an active researcher. "
            "Use the selected papers and the user's notes as the source of truth. Be specific about paper titles and authors. "
            "Every factual claim must cite one of the exact bracketed source labels supplied below. "
            "Do not invent claims or citations.\n\n"
            + (profile_ctx + "\n\n" if profile_ctx else "")
            + paper_ctx
        )
        prompt = (
            f"Generate a {review_type.lower()} in Markdown. {focus_clause}\n\n"
            "Required structure:\n"
            "## Scope\n"
            "Define the topic and why these papers belong together.\n\n"
            "## Main Claims and Evidence\n"
            "Synthesize the central claims. Attribute claims to specific papers by title and author.\n\n"
            "## Methods, Data, and Assumptions\n"
            "Compare the methods or datasets used across the papers.\n\n"
            "## Tensions and Caveats\n"
            "Identify disagreements, limitations, or assumptions.\n\n"
            "## How I Would Cite These Papers\n"
            "Map each paper to the reason it should be cited.\n\n"
            "## Open Questions\n"
            "List concrete gaps that could motivate future work.\n\n"
            "Keep the writing concise but publication-useful."
        )
        with st.spinner("Generating literature review..."):
            review = _assistant_call(system, [{"role": "user", "content": prompt}])
        review = retrieval.linkify_citations(review, evidence)
        st.session_state["last_lit_review"] = review
        st.session_state["last_lit_review_title"] = focus.strip() or review_type

    if st.session_state.get("last_lit_review"):
        st.markdown("---")
        st.markdown(st.session_state["last_lit_review"])
        filename_stub = re.sub(r"[^a-zA-Z0-9]+", "_", st.session_state.get("last_lit_review_title", "literature_review")).strip("_").lower()
        st.download_button(
            "Download Markdown",
            st.session_state["last_lit_review"],
            file_name=f"{filename_stub or 'literature_review'}.md",
            mime="text/markdown",
        )


def render_assistant():
    st.header("Research Assistant")
    st.markdown(
        "Chat with your entire library, generate a research briefing, or find gaps and open questions."
    )

    papers = paper_store.load_all_papers()
    if not papers:
        st.info("Your library is empty. Search for papers first.")
        return

    n = len(papers)
    model_name = st.session_state.summarizer._active_model()
    st.caption(f"{n} papers in library — model: {model_name}")

    mode = st.radio(
        "Mode:",
        ["Cross-library chat", "Research briefing", "Gap finder", "Project ideas"],
        horizontal=True,
        key="assistant_mode",
    )

    profile = researcher_profile.load()
    profile_context = researcher_profile.to_context_string(profile)
    library_context = ""

    # ------------------------------------------------------------------ #
    if mode == "Cross-library chat":
        st.markdown("Ask anything about your library — themes, comparisons, recommendations, connections.")

        system = (
            "You are a research assistant with access to the user's personal library of academic papers. "
            "Answer questions accurately based on the papers listed. When referencing a paper, mention its title and authors. "
            "If the answer requires knowledge beyond the library, say so.\n\n"
            + library_context
        )

        history = st.session_state.get("assistant_chat_history", [])
        for msg in history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask about your library...", key="assistant_chat_input")
        if user_input:
            library_context, evidence = _retrieved_library_context(user_input, limit=18)
            system = (
                "Answer only from the retrieved research evidence. Cite factual statements using the exact "
                "bracketed labels. If evidence is insufficient, say so.\n\n"
                + (profile_context + "\n\n" if profile_context else "")
                + library_context
            )
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = _assistant_call(system, history + [{"role": "user", "content": user_input}])
                    reply = retrieval.linkify_citations(reply, evidence)
                st.write(reply)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            st.session_state["assistant_chat_history"] = history

        if history and st.button("Clear conversation", key="clear_assistant_chat"):
            st.session_state["assistant_chat_history"] = []

    # ------------------------------------------------------------------ #
    elif mode == "Research briefing":
        st.markdown(
            "Generate a structured overview of your library: key themes, recent developments, "
            "and connections between papers."
        )

        custom_focus = st.text_input(
            "Optional focus (leave blank for general overview):",
            placeholder="e.g., kilonova models, machine learning methods, gravitational waves",
            key="briefing_focus",
        )

        if st.button("Generate briefing", type="primary", key="gen_briefing"):
            focus_clause = f" Focus particularly on: {custom_focus}." if custom_focus.strip() else ""
            library_context, _ = _retrieved_library_context(
                custom_focus or "key themes recent developments methods connections", limit=24
            )
            system = (
                "You are a research assistant helping a scientist understand their paper library.\n\n"
                + library_context
            )
            prompt = (
                f"Write a structured research briefing based on this library.{focus_clause}\n\n"
                "Structure it as:\n"
                "## Key Themes\n"
                "What are the main research themes and topics across these papers?\n\n"
                "## Recent Developments\n"
                "What are the most recent findings or advances (focus on newest papers)?\n\n"
                "## Notable Connections\n"
                "Which papers are closely related? What methodological or thematic threads connect them?\n\n"
                "## Suggested Reading Order\n"
                "For someone new to this field, suggest an order to read these papers and why.\n\n"
                "Be specific — cite paper titles and authors."
            )
            with st.spinner("Generating briefing..."):
                briefing = _assistant_call(system, [{"role": "user", "content": prompt}])
            st.session_state["last_briefing"] = briefing

        if "last_briefing" in st.session_state:
            st.markdown(st.session_state["last_briefing"])
            if st.button("Copy to clipboard", key="copy_briefing"):
                st.code(st.session_state["last_briefing"], language="markdown")

    # ------------------------------------------------------------------ #
    elif mode == "Gap finder":
        st.markdown(
            "Identify understudied areas, open questions, and potential research directions "
            "based on your library."
        )

        angle = st.selectbox(
            "Perspective:",
            [
                "Open questions and unknowns",
                "Methodological gaps",
                "Contradictions or debates in the literature",
                "Potential future directions",
                "Grant / proposal angles",
            ],
            key="gap_angle",
        )

        if st.button("Analyse gaps", type="primary", key="gen_gaps"):
            angle_prompts = {
                "Open questions and unknowns": (
                    "What are the key open questions and unresolved problems across these papers? "
                    "What do the authors themselves identify as unknowns or future work?"
                ),
                "Methodological gaps": (
                    "What methodological limitations or gaps exist across these papers? "
                    "What methods are missing, underused, or identified as needing improvement?"
                ),
                "Contradictions or debates in the literature": (
                    "Are there any contradictions, disagreements, or active debates between papers in this library? "
                    "Where do authors reach different conclusions on the same question?"
                ),
                "Potential future directions": (
                    "Based on the current state of research in these papers, what are the most promising "
                    "future research directions? What natural next steps follow from these findings?"
                ),
                "Grant / proposal angles": (
                    "Identify the most compelling research gaps that could form the basis of a grant proposal. "
                    "What problems are clearly important, currently unsolved, and tractable?"
                ),
            }
            library_context, _ = _retrieved_library_context(angle, limit=24)
            system = (
                "You are a senior research advisor helping a scientist identify gaps and opportunities "
                "in their field based on their paper library.\n\n"
                + library_context
            )
            prompt = (
                f"{angle_prompts[angle]}\n\n"
                "Be specific — reference paper titles and authors where relevant. "
                "Structure your response with clear headings and bullet points."
            )
            with st.spinner("Analysing..."):
                gaps = _assistant_call(system, [{"role": "user", "content": prompt}])
            st.session_state["last_gaps"] = gaps

        if "last_gaps" in st.session_state:
            st.markdown(st.session_state["last_gaps"])

    # ------------------------------------------------------------------ #
    elif mode == "Project ideas":
        st.markdown(
            "Propose concrete new research projects based on fresh ArXiv papers and your library, "
            "tailored to your specialities."
        )

        col1, col2 = st.columns(2)
        topic = col1.text_input(
            "Research topic",
            placeholder="e.g., kilonovae, magnetar-powered transients, gravitational wave counterparts",
            key="proj_topic",
        )
        profile_tools = researcher_profile.load().get("methods_and_tools", "")
        specialities = col2.text_input(
            "Your specialities / tools",
            value=profile_tools,
            placeholder="e.g., Bayesian inference, light curve modelling, redback, MCMC",
            key="proj_specialities",
        )
        n_ideas = st.slider("Number of project ideas", 2, 6, 3, key="proj_n_ideas")
        fetch_fresh = st.checkbox(
            "Fetch fresh ArXiv papers on this topic in real-time",
            value=True,
            key="proj_fetch_fresh",
        )

        if st.button("Generate project ideas", type="primary", key="gen_projects"):
            if not topic.strip():
                st.warning("Please enter a research topic.")
            else:
                # Optionally fetch fresh ArXiv abstracts
                fresh_context = ""
                if fetch_fresh:
                    with st.spinner(f"Fetching latest ArXiv papers on '{topic}'..."):
                        try:
                            fresh_results = st.session_state.arxiv.search(
                                query=topic.strip(),
                                max_results=10,
                                categories=[],
                                date_from=None,
                                author="",
                            )
                            if fresh_results:
                                fresh_lines = [f"\nLatest ArXiv papers on '{topic}' (fetched in real-time):\n"]
                                for r in fresh_results:
                                    meta = st.session_state.arxiv.get_paper_metadata(r)
                                    authors_str = ", ".join(meta.get("authors", [])[:3])
                                    fresh_lines.append(
                                        f"- {meta['title']} — {authors_str} ({str(meta.get('published',''))[:10]})\n"
                                        f"  Abstract: {r.summary[:400].strip()}...\n"
                                    )
                                fresh_context = "\n".join(fresh_lines)
                        except Exception as e:
                            st.warning(f"Could not fetch fresh papers: {e}")

                system = (
                    "You are a senior research advisor helping an astrophysics researcher identify "
                    "novel, tractable project ideas. You have access to their paper library and "
                    "knowledge of the latest work in the field.\n\n"
                    + _retrieved_library_context(topic, limit=18)[0]
                    + fresh_context
                )

                specialities_clause = (
                    f"The researcher's specialities and tools include: {specialities.strip()}.\n"
                    if specialities.strip() else ""
                )

                prompt = (
                    f"Propose {n_ideas} concrete, novel research project ideas on the topic of: **{topic}**.\n\n"
                    f"{specialities_clause}"
                    "For each project idea, structure it as:\n\n"
                    "### Project [N]: [Catchy title]\n"
                    "**Motivation:** Why is this problem important and timely? What gap does it address?\n"
                    "**Approach:** What would you actually do? Be specific about methods, data, and tools.\n"
                    "**Novelty:** What makes this distinct from existing work in the library or the fresh papers?\n"
                    "**Relevant papers:** Which papers from the library or the fresh ArXiv list are most relevant?\n"
                    "**Difficulty / timeline:** Is this a 3-month, 1-year, or multi-year project? What are the main risks?\n\n"
                    "Prioritise ideas that are:\n"
                    "- Feasible given the researcher's specialities\n"
                    "- Motivated by genuine gaps in the current literature\n"
                    "- Timely given the most recent papers\n"
                    "Be specific and concrete — avoid vague suggestions."
                )

                with st.spinner("Generating project ideas..."):
                    ideas = _assistant_call(system, [{"role": "user", "content": prompt}])
                st.session_state["last_project_ideas"] = ideas
                st.session_state["last_project_topic"] = topic

        if "last_project_ideas" in st.session_state:
            st.markdown(f"*Project ideas for: **{st.session_state.get('last_project_topic', '')}***")
            st.markdown(st.session_state["last_project_ideas"])

            # Follow-up chat to drill into a specific idea
            st.markdown("---")
            st.markdown("**Drill down on an idea**")
            followup = st.chat_input("Ask a follow-up question about any of these ideas...", key="proj_followup_input")
            if followup:
                proj_history = st.session_state.get("proj_followup_history", [])
                # System includes the generated ideas as context
                proj_system = (
                    "You are a senior research advisor. You previously proposed the following project ideas:\n\n"
                    + st.session_state["last_project_ideas"]
                    + "\n\nAnswer follow-up questions about these ideas in detail. "
                    "Be specific and practical."
                )
                with st.chat_message("user"):
                    st.write(followup)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        reply = _assistant_call(proj_system, proj_history + [{"role": "user", "content": followup}])
                    st.write(reply)
                proj_history.append({"role": "user", "content": followup})
                proj_history.append({"role": "assistant", "content": reply})
                st.session_state["proj_followup_history"] = proj_history


def render_profile():
    st.header("Researcher Profile")
    st.markdown(
        "Your profile is used by the Assistant to tailor project ideas, briefings, and gap analysis to your background. "
        "Generate it automatically from your library or fill it in manually."
    )

    profile = researcher_profile.load()

    # Auto-generate from library
    papers = paper_store.load_all_papers()
    gen_col, _ = st.columns([1, 3])
    if gen_col.button("Auto-generate from library", type="primary", disabled=len(papers) == 0):
        if not papers:
            st.warning("Add some papers to your library first.")
        else:
            library_ctx, _ = _retrieved_library_context(
                "research areas methods tools recurring topics professional profile", limit=40
            )
            system = "You are helping a researcher build their professional profile based on their paper library."
            prompt = (
                "Based on this researcher's paper library, infer a professional profile. "
                "Return ONLY a JSON object with these exact keys (no markdown, no explanation):\n"
                '{"position": "...", "research_areas": "...", "methods_and_tools": "...", "bio": "..."}\n\n'
                "Guidelines:\n"
                "- position: likely career stage (e.g. 'Postdoctoral researcher')\n"
                "- research_areas: comma-separated list of specific research topics (2-5 items)\n"
                "- methods_and_tools: comma-separated list of methods, codes, frameworks evident from the papers (3-8 items)\n"
                "- bio: 2-3 sentence summary of their research focus and approach, written in third person\n\n"
                + library_ctx
            )
            with st.spinner("Inferring profile from your library..."):
                raw = _assistant_call(system, [{"role": "user", "content": prompt}])
            # Parse JSON from response
            try:
                # Strip markdown code fences if present
                clean = raw.strip().strip("```json").strip("```").strip()
                inferred = json.loads(clean)
                # Merge — keep name/institution if user already set them
                for k in ("position", "research_areas", "methods_and_tools", "bio"):
                    if inferred.get(k):
                        profile[k] = inferred[k]
                researcher_profile.save(profile)
                st.success("Profile generated — review and edit below.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not parse response: {e}")
                st.code(raw)

    st.markdown("---")

    # Editable fields
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        profile["name"] = col1.text_input("Name", value=profile.get("name", ""))
        profile["position"] = col2.text_input("Position", value=profile.get("position", ""), placeholder="e.g. Postdoctoral researcher")
        profile["institution"] = col1.text_input("Institution", value=profile.get("institution", ""))
        profile["research_areas"] = st.text_area(
            "Research areas",
            value=profile.get("research_areas", ""),
            placeholder="e.g. kilonovae, neutron star mergers, gravitational wave counterparts",
            height=80,
        )
        profile["methods_and_tools"] = st.text_area(
            "Methods & tools",
            value=profile.get("methods_and_tools", ""),
            placeholder="e.g. Bayesian inference, MCMC, redback, light curve modelling, Python",
            height=80,
        )
        profile["bio"] = st.text_area(
            "Bio",
            value=profile.get("bio", ""),
            placeholder="A short description of your research focus...",
            height=100,
        )
        if st.form_submit_button("Save profile", type="primary"):
            researcher_profile.save(profile)
            st.success("Profile saved.")

    if not researcher_profile.is_empty(profile):
        st.markdown("---")
        st.markdown("**Current profile (as seen by the Assistant):**")
        st.code(researcher_profile.to_context_string(profile), language="markdown")


# ---------------------------------------------------------------------------
# Shared idea workspace
# ---------------------------------------------------------------------------

def _idea_workspace(idea_type: str, idea: Dict):
    """Render the workspace for a single saved idea."""
    iid = idea["id"]
    profile = researcher_profile.load()
    profile_ctx = researcher_profile.to_context_string(profile)

    idea_context = (
        f"The researcher is working on the following {'paper' if idea_type == 'paper' else 'grant'} idea:\n\n"
        f"Title: {idea['title']}\n"
        f"Description: {idea['description']}\n"
    )

    all_papers = paper_store.load_all_papers()
    paper_by_id = {p.get("arxiv_id", p.get("id", "")): p for p in all_papers if p.get("arxiv_id", p.get("id", ""))}
    label_by_id = {pid: _paper_label(paper) for pid, paper in paper_by_id.items()}
    paper_by_label = {label: paper_by_id[pid] for pid, label in label_by_id.items()}
    current_ids = [pid for pid in idea.get("linked_papers", []) if pid in paper_by_id]
    current_labels = [label_by_id[pid] for pid in current_ids]

    selected_labels = st.multiselect(
        "Linked papers",
        options=list(paper_by_label.keys()),
        default=current_labels,
        key=f"{idea_type}_linked_papers_{iid}",
    )
    selected_ids = [paper_by_label[label].get("arxiv_id", paper_by_label[label].get("id", "")) for label in selected_labels]
    if selected_ids != current_ids:
        idea_store.set_linked_papers(idea_type, iid, selected_ids)
        idea["linked_papers"] = selected_ids
        current_ids = selected_ids

    linked_papers = [paper_by_id[pid] for pid in current_ids if pid in paper_by_id]
    if linked_papers:
        linked_rows = []
        for paper in linked_papers:
            notes = paper.get("research_notes", {}) if isinstance(paper.get("research_notes", {}), dict) else {}
            linked_rows.append({
                "title": paper.get("title", paper.get("arxiv_id", "")),
                "arxiv_id": paper.get("arxiv_id", ""),
                "cite_for": notes.get("cite_for", ""),
                "key_result": notes.get("key_result", ""),
            })
        with st.expander("Linked paper context", expanded=False):
            st.dataframe(pd.DataFrame(linked_rows), use_container_width=True, hide_index=True)

    library_ctx, _ = _retrieved_library_context(
        f"{idea.get('title', '')} {idea.get('description', '')}", limit=24
    )
    linked_ctx = _build_selected_papers_context(linked_papers, include_notes=True) if linked_papers else ""
    base_context = (
        (profile_ctx + "\n\n" if profile_ctx else "")
        + (f"## Papers linked to this idea\n{linked_ctx}\n\n" if linked_ctx else "")
        + "## Wider paper library\n"
        + library_ctx
    )

    ws_tab1, ws_tab2, ws_tab3, ws_tab4 = st.tabs([
        "Iterate", "Literature check",
        "Skills & tools" if idea_type == "paper" else "Team & resources",
        "Project plan" if idea_type == "paper" else "Impact & funding",
    ])

    # ---- Iterate ----
    with ws_tab1:
        st.markdown("Chat with Gemini to refine, reshape, or explore variations of this idea.")
        history = idea.get("chat_history", [])
        for msg in history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        user_input = st.chat_input("Refine or question this idea...", key=f"ws_chat_{iid}")
        if user_input:
            system = (
                f"You are a research advisor helping develop and refine a {'paper' if idea_type == 'paper' else 'grant'} idea.\n\n"
                + base_context + "\n\n" + idea_context
            )
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = _assistant_call(system, history + [{"role": "user", "content": user_input}])
                st.write(reply)
            idea_store.append_chat(idea_type, iid, "user", user_input)
            idea_store.append_chat(idea_type, iid, "assistant", reply)
            # Refresh local copy
            idea["chat_history"] = idea_store.get_idea(idea_type, iid).get("chat_history", [])
        if history and st.button("Clear chat", key=f"ws_clearchat_{iid}"):
            idea_store.update_idea(idea_type, iid, {"chat_history": []})
            st.rerun()

    # ---- Literature check ----
    with ws_tab2:
        st.markdown("Find what's already been done and clarify the gap this idea fills.")
        fetch_fresh_lit = st.checkbox("Also fetch fresh ArXiv papers", value=True, key=f"ws_lit_fresh_{iid}")
        if st.button("Run literature check", key=f"ws_lit_{iid}"):
            fresh_ctx = ""
            if fetch_fresh_lit:
                with st.spinner("Fetching fresh ArXiv papers..."):
                    try:
                        results = st.session_state.arxiv.search(
                            query=idea["title"], max_results=8, categories=[], date_from=None, author=""
                        )
                        if results:
                            lines = ["\nFresh ArXiv papers related to this idea:\n"]
                            for r in results:
                                meta = st.session_state.arxiv.get_paper_metadata(r)
                                lines.append(
                                    f"- {meta['title']} — {', '.join(meta.get('authors', [])[:3])} "
                                    f"({str(meta.get('published',''))[:10]})\n"
                                    f"  {r.summary[:300].strip()}...\n"
                                )
                            fresh_ctx = "\n".join(lines)
                    except Exception as e:
                        st.warning(f"Could not fetch fresh papers: {e}")

            system = "You are a research advisor performing a literature review.\n\n" + base_context + fresh_ctx
            prompt = (
                f"For the following idea:\n{idea_context}\n\n"
                "Provide a structured literature check:\n\n"
                "## What has already been done\n"
                "Summarise the most relevant existing work from the library and fresh papers. Be specific — cite titles and authors.\n\n"
                "## Key gap this idea addresses\n"
                "What specifically has NOT been done? Why does the gap exist?\n\n"
                "## Closest competing work\n"
                "Which existing papers come closest to this idea? How would this work differentiate itself?\n\n"
                "## Suggested papers to read\n"
                "List the 5 most important papers to read before starting this project."
            )
            with st.spinner("Checking literature..."):
                lit_review = _assistant_call(system, [{"role": "user", "content": prompt}])
            idea_store.update_idea(idea_type, iid, {"lit_review": lit_review})
            st.session_state[f"lit_review_{iid}"] = lit_review

        saved_lit = idea.get("lit_review") or st.session_state.get(f"lit_review_{iid}")
        if saved_lit:
            st.markdown(saved_lit)

    # ---- Skills & tools / Team & resources ----
    with ws_tab3:
        if idea_type == "paper":
            st.markdown("Understand what skills and tools this project requires vs what you already have.")
            if st.button("Analyse skills & tools", key=f"ws_skills_{iid}"):
                system = "You are a research advisor helping a researcher assess the feasibility of a project.\n\n" + base_context
                prompt = (
                    f"For this paper idea:\n{idea_context}\n\n"
                    "Provide a structured skills and tools breakdown:\n\n"
                    "## Skills & tools the researcher already has\n"
                    "Based on their profile, what relevant expertise do they bring?\n\n"
                    "## Skills & tools they would need to develop or acquire\n"
                    "What gaps exist? How significant are they?\n\n"
                    "## Software and data requirements\n"
                    "What specific codes, pipelines, or datasets are needed? Are they publicly available?\n\n"
                    "## Potential collaborators\n"
                    "What expertise would be valuable to bring in? Any obvious groups to approach?\n\n"
                    "## Overall feasibility assessment\n"
                    "Is this realistic given the researcher's current profile? What's the biggest risk?"
                )
                with st.spinner("Analysing..."):
                    skills = _assistant_call(system, [{"role": "user", "content": prompt}])
                idea_store.update_idea(idea_type, iid, {"skills_review": skills})
                st.session_state[f"skills_{iid}"] = skills

            saved_skills = idea.get("skills_review") or st.session_state.get(f"skills_{iid}")
            if saved_skills:
                st.markdown(saved_skills)

        else:  # grant
            st.markdown("Assess the team, resources, and infrastructure needed for this grant.")
            if st.button("Analyse team & resources", key=f"ws_team_{iid}"):
                system = "You are a grant advisor helping a researcher plan a funding application.\n\n" + base_context
                prompt = (
                    f"For this grant idea:\n{idea_context}\n\n"
                    "Provide a structured team and resources assessment:\n\n"
                    "## Core team required\n"
                    "What roles are needed (PI, postdocs, students, collaborators)? What expertise?\n\n"
                    "## Infrastructure and data\n"
                    "What computing, instruments, or datasets are required? What exists vs needs funding?\n\n"
                    "## Budget considerations\n"
                    "What are the major cost drivers? Any rough estimates?\n\n"
                    "## Existing strengths\n"
                    "Based on the researcher's profile, what do they already bring to this grant?\n\n"
                    "## Key gaps to address before applying\n"
                    "What partnerships, preliminary results, or infrastructure need to be in place first?"
                )
                with st.spinner("Analysing..."):
                    team = _assistant_call(system, [{"role": "user", "content": prompt}])
                idea_store.update_idea(idea_type, iid, {"team_review": team})
                st.session_state[f"team_{iid}"] = team

            saved_team = idea.get("team_review") or st.session_state.get(f"team_{iid}")
            if saved_team:
                st.markdown(saved_team)

    # ---- Project plan / Impact & funding ----
    with ws_tab4:
        if idea_type == "paper":
            st.markdown("Break this project into milestones with a realistic timeline.")
            if st.button("Generate project plan", key=f"ws_plan_{iid}"):
                system = "You are a research advisor helping plan a research project.\n\n" + base_context
                prompt = (
                    f"For this paper idea:\n{idea_context}\n\n"
                    "Generate a concrete project plan:\n\n"
                    "## Milestones\n"
                    "Break the project into 4-6 concrete milestones with rough timeframes.\n\n"
                    "## Key decision points\n"
                    "Where might the project pivot or fail? What are the go/no-go criteria?\n\n"
                    "## Minimum viable paper\n"
                    "What is the smallest version of this project that still produces a publishable result?\n\n"
                    "## Target journals\n"
                    "What journals would be appropriate for this work? Why?\n\n"
                    "## Overall timeline estimate\n"
                    "Realistic best case, expected, and worst case timelines."
                )
                with st.spinner("Planning..."):
                    plan = _assistant_call(system, [{"role": "user", "content": prompt}])
                idea_store.update_idea(idea_type, iid, {"project_plan": plan})
                st.session_state[f"plan_{iid}"] = plan

            saved_plan = idea.get("project_plan") or st.session_state.get(f"plan_{iid}")
            if saved_plan:
                st.markdown(saved_plan)

        else:  # grant
            st.markdown("Develop the impact statement and identify suitable funding bodies.")
            if st.button("Generate impact & funding analysis", key=f"ws_impact_{iid}"):
                system = (
                    "You are a highly experienced grant advisor who has helped researchers win ERC, UKRI, ARC, "
                    "NASA, and NSF grants. You write with ambition and clarity.\n\n" + base_context
                )
                prompt = (
                    f"For this grant idea:\n{idea_context}\n\n"
                    "Provide a detailed funding and impact analysis:\n\n"
                    "## Vision statement\n"
                    "Write a compelling 200-word vision statement as it might appear in the opening of a grant proposal. "
                    "Make it bold, clear, and memorable. Avoid jargon where possible.\n\n"
                    "## Scientific significance\n"
                    "What fundamental question does this address? What changes in the field if it succeeds? "
                    "Be specific about the scientific stakes.\n\n"
                    "## Broader impact\n"
                    "Societal, technological, multimessenger, or cross-disciplinary relevance. "
                    "Include potential downstream applications or public interest angles.\n\n"
                    "## Most suitable funding schemes\n"
                    "List 5-7 specific, named funding schemes with country/agency, typical budget range, "
                    "duration, and why this idea is a strong fit for each. Include both fellowship-style "
                    "and programme/project grant options.\n\n"
                    "## Competitive landscape\n"
                    "Who are the 3-5 strongest competing groups globally? How does this proposal differentiate? "
                    "What is the 'only you' argument?\n\n"
                    "## Key preliminary results needed\n"
                    "What proof-of-concept results should the researcher generate before submitting? "
                    "What would make reviewers confident this is achievable?"
                )
                with st.spinner("Analysing..."):
                    impact = _assistant_call(system, [{"role": "user", "content": prompt}])
                idea_store.update_idea(idea_type, iid, {"impact_review": impact})
                st.session_state[f"impact_{iid}"] = impact

            saved_impact = idea.get("impact_review") or st.session_state.get(f"impact_{iid}")
            if saved_impact:
                st.markdown(saved_impact)

    # Notes
    st.markdown("---")
    st.markdown("**Personal notes**")
    notes = st.text_area("Notes", value=idea.get("notes", ""), key=f"ws_notes_{iid}", height=100, label_visibility="collapsed")
    if st.button("Save notes", key=f"ws_savenotes_{iid}"):
        idea_store.update_idea(idea_type, iid, {"notes": notes})
        st.success("Notes saved.")


def _render_ideas_tab(idea_type: str):
    """Render the full ideas tab for paper or grant ideas."""
    label = "Paper" if idea_type == "paper" else "Grant"
    st.header(f"{label} Ideas")

    profile = researcher_profile.load()
    profile_ctx = researcher_profile.to_context_string(profile)
    base_context = profile_ctx

    # ---- Generate new idea ----
    with st.expander("Generate new idea", expanded=not idea_store.load_ideas(idea_type)):
        col1, col2 = st.columns(2)
        topic = col1.text_input(
            "Topic / focus area",
            placeholder="e.g., kilonovae light curve modelling" if idea_type == "paper" else "e.g., multimessenger transient astronomy",
            key=f"{idea_type}_gen_topic",
        )
        profile_tools = profile.get("methods_and_tools", "")
        specialities = col2.text_input(
            "Your specialities / tools",
            value=profile_tools,
            placeholder="e.g., Bayesian inference, redback, MCMC",
            key=f"{idea_type}_gen_spec",
        )
        n_ideas = st.slider("Number of ideas to generate", 1, 5, 3, key=f"{idea_type}_n_ideas")
        fetch_fresh = st.checkbox("Fetch fresh ArXiv papers on this topic", value=True, key=f"{idea_type}_fetch_fresh")

        if st.button(f"Generate {label.lower()} ideas", type="primary", key=f"{idea_type}_gen_btn"):
            if not topic.strip():
                st.warning("Please enter a topic.")
            else:
                library_ctx, _ = _retrieved_library_context(topic, limit=24)
                base_context = (profile_ctx + "\n\n" if profile_ctx else "") + library_ctx
                fresh_ctx = ""
                if fetch_fresh:
                    with st.spinner("Fetching latest ArXiv papers..."):
                        try:
                            results = st.session_state.arxiv.search(
                                query=topic.strip(), max_results=10, categories=[], date_from=None, author=""
                            )
                            if results:
                                lines = [f"\nLatest ArXiv papers on '{topic}':\n"]
                                for r in results:
                                    meta = st.session_state.arxiv.get_paper_metadata(r)
                                    lines.append(
                                        f"- {meta['title']} — {', '.join(meta.get('authors', [])[:3])} "
                                        f"({str(meta.get('published',''))[:10]})\n"
                                        f"  {r.summary[:350].strip()}...\n"
                                    )
                                fresh_ctx = "\n".join(lines)
                        except Exception as e:
                            st.warning(f"Could not fetch ArXiv papers: {e}")

                spec_clause = f"Researcher specialities: {specialities}.\n" if specialities.strip() else ""

                if idea_type == "paper":
                    prompt = (
                        f"Propose {n_ideas} specific, novel paper ideas on: **{topic}**.\n"
                        f"{spec_clause}\n"
                        "For each idea output EXACTLY this format (use the exact headers):\n\n"
                        "### TITLE: <concise paper title>\n"
                        "### DESCRIPTION: <2-3 sentences: what you'd do, key method, expected result>\n"
                        "### MOTIVATION: <why this is timely and important>\n"
                        "### NOVELTY: <what makes it distinct from existing work>\n"
                        "---\n\n"
                        "Be specific and concrete. Avoid vague suggestions.\n\n"
                        + base_context + fresh_ctx
                    )
                else:
                    prompt = (
                        f"Propose {n_ideas} ambitious, fundable research programme ideas centred on: **{topic}**.\n"
                        f"{spec_clause}\n\n"
                        "These should be ideas that could anchor a major fellowship or programme grant — "
                        "ERC Starting/Consolidator, UKRI Future Leaders Fellowship, ARC DECRA/Future Fellowship, "
                        "NASA ATP, NSF CAREER, Royal Society URF, or similar. "
                        "Think boldly. A strong grant answers: why this question, why you, why now, why does it matter. "
                        "It should be scientifically transformative — not just the next incremental paper — "
                        "involving multiple interconnected work packages over 3-5 years with a small team.\n\n"
                        "For each idea output EXACTLY this format:\n\n"
                        "### TITLE: <bold, memorable programme title>\n"
                        "### DESCRIPTION: <4-5 sentences covering: overarching vision, 2-3 concrete work packages, "
                        "key deliverables, and why this requires a programme not just a single paper>\n"
                        "### MOTIVATION: <the big open question this addresses — field-defining scale. "
                        "What fundamentally changes in the field if this succeeds?>\n"
                        "### NOVELTY: <what makes this distinctive and fundable only by this researcher — "
                        "unique combination of expertise, timing, methods, data access, or perspective>\n"
                        "---\n\n"
                        "Do not propose safe or obvious ideas. Aim for ideas a review panel would remember.\n\n"
                        + base_context + fresh_ctx
                    )

                system = (
                    "You are a highly experienced research grant advisor who has helped astrophysicists win "
                    "major fellowships and programme grants at ERC, UKRI, ARC, NASA, and NSF level. "
                    "You understand what review panels look for: bold scientific vision, clear feasibility, "
                    "a compelling narrative, and a strong 'only you can do this' argument. "
                    "You do not produce safe or incremental ideas — you push researchers to think bigger."
                )
                with st.spinner("Generating ideas..."):
                    raw = _assistant_call(system, [{"role": "user", "content": prompt}])

                # Parse ideas from response and offer save buttons
                st.session_state[f"{idea_type}_gen_raw"] = raw
                st.session_state[f"{idea_type}_gen_parsed"] = _parse_ideas(raw)

        # Show generated ideas with save buttons
        if st.session_state.get(f"{idea_type}_gen_parsed"):
            parsed = st.session_state[f"{idea_type}_gen_parsed"]
            st.markdown("---")
            for i, idea in enumerate(parsed):
                with st.container():
                    st.markdown(f"**{idea['title']}**")
                    st.markdown(idea["description"])
                    save_col, _ = st.columns([1, 4])
                    if save_col.button("Save this idea", key=f"{idea_type}_save_{i}"):
                        idea_store.save_idea(idea_type, idea["title"], idea["description"] + "\n\n**Motivation:** " + idea.get("motivation", "") + "\n\n**Novelty:** " + idea.get("novelty", ""))
                        st.success(f"Saved: {idea['title']}")
                    st.markdown("---")

    # ---- Saved ideas ----
    saved = idea_store.load_ideas(idea_type)
    if not saved:
        st.info(f"No saved {label.lower()} ideas yet. Generate some above.")
        return

    st.markdown(f"### Saved {label} Ideas ({len(saved)})")

    # Status filter
    statuses = ["all", "draft", "active", "archived"]
    status_filter = st.selectbox("Filter by status:", statuses, key=f"{idea_type}_status_filter")

    for idea in sorted(saved, key=lambda x: x.get("created", ""), reverse=True):
        if status_filter != "all" and idea.get("status") != status_filter:
            continue

        status = idea.get("status", "draft")
        status_emoji = {"draft": "📝", "active": "🔬", "archived": "📦"}.get(status, "📝")
        created = idea.get("created", "")[:10]

        with st.expander(f"{status_emoji} **{idea['title']}** — {status} · {created}", expanded=False):
            # Status control
            scol1, scol2 = st.columns([2, 3])
            new_status = scol1.selectbox(
                "Status", ["draft", "active", "archived"],
                index=["draft", "active", "archived"].index(status),
                key=f"{idea_type}_status_{idea['id']}",
            )
            if new_status != status:
                idea_store.update_idea(idea_type, idea["id"], {"status": new_status})

            if scol2.button("Delete idea", key=f"{idea_type}_del_{idea['id']}"):
                idea_store.delete_idea(idea_type, idea["id"])
                st.rerun()

            st.markdown(idea.get("description", ""))
            st.markdown("---")
            _idea_workspace(idea_type, idea)


def _parse_ideas(raw: str) -> List[Dict]:
    """Parse structured ideas from LLM output."""
    ideas = []
    # Split on --- separator
    blocks = [b.strip() for b in raw.split("---") if b.strip()]
    for block in blocks:
        idea = {"title": "", "description": "", "motivation": "", "novelty": ""}
        for line in block.split("\n"):
            for key, prefix in [("title", "### TITLE:"), ("description", "### DESCRIPTION:"),
                                  ("motivation", "### MOTIVATION:"), ("novelty", "### NOVELTY:")]:
                if line.startswith(prefix):
                    idea[key] = line[len(prefix):].strip()
        if idea["title"]:
            ideas.append(idea)
    return ideas


def render_paper_ideas():
    _render_ideas_tab("paper")


def render_grant_ideas():
    _render_ideas_tab("grant")


def render_projects():
    """Render project sources, literature links, weekly briefings, and proposed actions."""
    st.header("Projects")
    with st.expander("Create project", expanded=False):
        with st.form("create_project"):
            title = st.text_input("Project title")
            description = st.text_area("Goal and scientific context")
            methods = st.text_input("Methods and tools")
            if st.form_submit_button("Create", type="primary"):
                if title.strip():
                    project_store.save_project(title=title, description=description, methods=methods)
                    st.rerun()
                else:
                    st.warning("Enter a project title.")

    projects = project_store.load_projects()
    if not projects:
        st.info("Create a project or save a paper/grant idea first.")
        return
    labels = {f"{project['title']} [{project['project_id']}]": project for project in projects}
    selected_label = st.selectbox("Project", list(labels), key="project_selector")
    project = labels[selected_label]
    project_id = project["project_id"]

    overview_tab, sources_tab, papers_tab, briefing_tab, actions_tab = st.tabs(
        ["Overview", "Sources", "Papers", "Weekly briefing", "Actions"]
    )
    with overview_tab:
        with st.form(f"project_overview_{project_id}"):
            project_title = st.text_input("Title", value=project.get("title", ""))
            project_description = st.text_area("Goal and context", value=project.get("description", ""), height=160)
            project_methods = st.text_area("Methods and tools", value=project.get("methods", ""), height=100)
            project_status = st.selectbox(
                "Status",
                ["active", "paused", "archived"],
                index=["active", "paused", "archived"].index(project.get("status", "active")),
            )
            if st.form_submit_button("Save project"):
                project_store.save_project(
                    project_id=project_id,
                    title=project_title,
                    description=project_description,
                    methods=project_methods,
                    kind=project.get("kind", "research"),
                    status=project_status,
                    repository_paths=project.get("repository_paths", []),
                    notion_url=project.get("notion_url", ""),
                    notion_export_path=project.get("notion_export_path", ""),
                )
                st.success("Project saved.")

    with sources_tab:
        with st.form(f"project_sources_{project_id}"):
            repositories = st.text_area(
                "Git repositories (one absolute path per line)",
                value="\n".join(project.get("repository_paths", [])),
                height=120,
            )
            notion_url = st.text_input("Notion page URL", value=project.get("notion_url", ""))
            notion_export = st.text_input(
                "Local Notion Markdown export path",
                value=project.get("notion_export_path", ""),
            )
            if st.form_submit_button("Save sources"):
                project_store.save_project(
                    project_id=project_id,
                    title=project.get("title", ""),
                    description=project.get("description", ""),
                    methods=project.get("methods", ""),
                    kind=project.get("kind", "research"),
                    status=project.get("status", "active"),
                    repository_paths=[line.strip() for line in repositories.splitlines() if line.strip()],
                    notion_url=notion_url,
                    notion_export_path=notion_export,
                )
                st.success("Sources saved.")
        if st.button("Capture weekly changes", key=f"capture_sources_{project_id}"):
            refreshed = project_store.get_project(project_id) or project
            result = project_store.capture_project_sources(refreshed)
            if result["git"] or result["notion_export"]:
                st.success(f"Captured {result['git']} Git and {result['notion_export']} Notion-export snapshots.")
            else:
                st.info("No new source changes were found.")
            for error in result["errors"]:
                st.warning(error)
        snapshots = project_store.latest_snapshots(project_id)
        for snapshot in snapshots:
            with st.expander(f"{snapshot['created_at'][:16]} - {snapshot['source']}"):
                st.code(snapshot["content"][:12000])
        with st.form(f"meeting_note_{project_id}", clear_on_submit=True):
            meeting_title = st.text_input("Meeting title")
            meeting_text = st.text_area("Meeting notes", height=140)
            if st.form_submit_button("Save and index meeting") and meeting_text.strip():
                meeting_id = f"{project_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
                corpus_index.index_text_record(
                    owner_type="meeting",
                    owner_id=meeting_id,
                    kind="meeting",
                    title=meeting_title or f"Meeting for {project.get('title', project_id)}",
                    text=meeting_text,
                    locator={"project_id": project_id, "created_at": datetime.now(timezone.utc).isoformat()},
                    vector_db=_session_vector_db(),
                )
                st.success("Meeting notes saved and indexed.")

    with papers_tab:
        all_papers = paper_store.load_all_papers()
        paper_options = {_paper_label(paper): paper for paper in all_papers}
        current_ids = set(project_store.linked_paper_ids(project_id))
        defaults = [label for label, paper in paper_options.items() if paper.get("arxiv_id") in current_ids]
        selected = st.multiselect(
            "Linked evidence",
            list(paper_options),
            default=defaults,
            key=f"project_papers_{project_id}",
        )
        if st.button("Save paper links", key=f"save_project_papers_{project_id}"):
            project_store.set_linked_papers(
                project_id,
                [paper_options[label].get("arxiv_id", "") for label in selected],
            )
            st.success("Paper links saved.")

        ranked = relevance.score_papers(
            all_papers,
            interest_text=" ".join(
                [project.get("title", ""), project.get("description", ""), project.get("methods", "")]
            ),
        )[:10]
        st.markdown("**Suggested from your library**")
        for paper in ranked:
            st.markdown(
                f"- {paper['relevance_score']:.0f}% [{paper.get('title', paper.get('arxiv_id', ''))}]"
                f"(https://arxiv.org/abs/{paper.get('arxiv_id', '')}) - {paper['relevance_reason']}"
            )

    with briefing_tab:
        if st.button("Generate weekly project briefing", type="primary", key=f"project_briefing_{project_id}"):
            snapshots = project_store.latest_snapshots(project_id, limit=6)
            linked_ids = project_store.linked_paper_ids(project_id)
            linked = [paper for paper in paper_store.load_all_papers() if paper.get("arxiv_id") in linked_ids]
            system = (
                "You are preparing a private weekly research-project briefing. Distinguish repository/Notion changes "
                "from literature evidence and do not invent progress.\n\n"
                f"Project: {project.get('title')}\n{project.get('description')}\nMethods: {project.get('methods')}\n\n"
                + _build_selected_papers_context(linked, include_notes=True)
                + "\n\nSnapshots:\n"
                + "\n\n".join(f"[{item['source']}]\n{item['content'][:16000]}" for item in snapshots)
            )
            prompt = (
                "Produce: ## Progress, ## Relevant Literature, ## Risks or Blockers, ## Decisions Needed, "
                "and ## Proposed Next Actions. Cite paper titles for literature claims and snapshot source labels for progress claims."
            )
            with st.spinner("Generating private project briefing..."):
                st.session_state[f"project_briefing_result_{project_id}"] = _assistant_call(
                    system,
                    [{"role": "user", "content": prompt}],
                    contains_private_data=True,
                )
        if st.session_state.get(f"project_briefing_result_{project_id}"):
            st.markdown(st.session_state[f"project_briefing_result_{project_id}"])

    with actions_tab:
        st.caption("Actions are proposals only. Approval never executes an external tool automatically.")
        handoff = st.file_uploader(
            "Import TheLocalWhisperer action handoff (JSON)",
            type=["json"],
            key=f"action_handoff_{project_id}",
        )
        if handoff and st.button("Validate and import proposals", key=f"import_actions_{project_id}"):
            try:
                imported = action_contract.import_proposals(
                    "TheLocalWhisperer",
                    json.loads(handoff.getvalue()),
                    project_id=project_id,
                )
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(f"Invalid handoff: {exc}")
            else:
                st.success(f"Imported {len(imported)} proposals. Review and approve each one below.")
                st.rerun()
        with st.form(f"propose_action_{project_id}"):
            action_title = st.text_input("Proposed action")
            action_type = st.selectbox("Type", ["task.create", "calendar.create", "notion.update", "agent.invoke"])
            action_details = st.text_area("Payload or instructions")
            if st.form_submit_button("Propose action") and action_title.strip():
                action_contract.propose(
                    "nsarxiv-app",
                    action_type,
                    action_title,
                    {"instructions": action_details},
                    project_id=project_id,
                )
                st.rerun()
        for action in action_contract.list_actions(limit=100):
            if action.get("project_id") != project_id:
                continue
            cols = st.columns([5, 1, 1])
            cols[0].markdown(f"**{action['title']}**  \n`{action['action_type']}` - {action['status']}")
            if action["status"] == "proposed":
                if cols[1].button("Approve", key=f"approve_{action['action_id']}"):
                    action_contract.update_status(action["action_id"], "approved")
                    st.rerun()
                if cols[2].button("Reject", key=f"reject_{action['action_id']}"):
                    action_contract.update_status(action["action_id"], "rejected")
                    st.rerun()


def render_research_ops():
    """Render corpus trends, background indexing, and measured model routes."""
    st.header("Library Health")
    trends_tab, indexing_tab, eval_tab = st.tabs(["Trends", "Search index", "Model checks"])

    with trends_tab:
        window = st.slider("Recent window (days)", 30, 365, 90, 30)
        rows = trends.emerging_themes(paper_store.load_all_papers(), recent_days=window)
        if rows:
            trend_rows = [{
                "theme": row["theme"],
                "evidence": row["basis"],
                "title support": row["title_support"],
                "growth": f"{row['lift']:.1f}x",
                "representative papers": " | ".join(row["examples"]),
            } for row in rows]
            st.dataframe(pd.DataFrame(trend_rows), width="stretch", hide_index=True)
        else:
            st.info("Not enough papers occur in both comparison windows yet.")

    with indexing_tab:
        health = research_db.health_snapshot()
        vector_count = _session_vector_db().collection.count()
        cols = st.columns(4)
        cols[0].metric("Papers", health["papers"])
        cols[1].metric("Vector records", vector_count)
        cols[2].metric("Queued/running", health["pending_jobs"])
        cols[3].metric("Failed", health["failed_jobs"])
        st.caption(f"Search documents: {health['documents']}")
        if vector_count != health["papers"]:
            st.warning(
                f"Vector index count differs from the library by "
                f"{abs(health['papers'] - vector_count)} records."
            )
        start_col, repair_col = st.columns(2)
        if start_col.button("Index missing full text", type="primary", use_container_width=True):
            job_id = index_jobs.enqueue_library_index(
                arxiv_client=st.session_state.arxiv,
                pdf_extractor=st.session_state.pdf_extractor,
                vector_db=_session_vector_db(),
            )
            st.success(f"Queued indexing job {job_id[:8]}.")
        if repair_col.button("Repair interrupted jobs", use_container_width=True):
            repaired = index_jobs.repair_stale_jobs()
            st.success(f"Marked {repaired} interrupted jobs for rerun.")
        jobs = index_jobs.list_jobs()
        if jobs:
            st.dataframe(pd.DataFrame([{
                "job": job["job_id"][:8], "status": job["status"],
                "papers": job.get("result", {}).get("papers_examined", ""),
                "chunks": job.get("result", {}).get("chunks_written", ""),
                "error": job.get("error", ""), "updated": job["updated_at"][:19],
            } for job in jobs]), use_container_width=True, hide_index=True)

    with eval_tab:
        provider = st.selectbox("Provider", ["ollama", "gemini", "anthropic", "openai"], key="eval_provider")
        model = st.text_input("Model", value=st.session_state.summarizer._active_model(), key="eval_model")
        if st.button("Run baseline suite", type="primary", key="run_eval"):
            evaluator = PaperSummarizer(provider=provider, model=model.strip() or None)
            with st.spinner("Running three evaluation prompts..."):
                try:
                    result = evaluation.run_suite(
                        suite="research-baseline-v1",
                        provider=provider,
                        model=model,
                        complete=lambda question: evaluator.complete(
                            "Answer concisely. Include the key noun from the question in the answer.", question
                        ),
                    )
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")
                else:
                    st.success(f"Completed run {result['run_id'][:8]}.")
        runs = evaluation.list_runs()
        route = evaluation.recommend_route(runs, contains_private_data=False)
        if route:
            st.caption(f"Best measured public route: {route['provider']} / {route['model']}")
        if runs:
            st.dataframe(pd.DataFrame([{
                "provider": run["provider"], "model": run["model"],
                "coverage": round(run["metrics"].get("task_coverage", 0), 2),
                "citation precision": round(run["metrics"].get("citation_precision", 0), 2),
                "latency (s)": round(run["metrics"].get("mean_latency_seconds", 0), 2),
                "created": run["created_at"][:19],
            } for run in runs]), use_container_width=True, hide_index=True)


def render_citation_opportunities():
    """Render the automatically populated evidence-review and export queue."""
    st.header("Citation Opportunities")
    orchestrator_endpoint = os.getenv(
        "LOCAL_ORCHESTRATOR_URL", "http://127.0.0.1:8775/v1/import-bundles"
    )
    orchestrator_ui = os.getenv(
        "LOCAL_ORCHESTRATOR_UI_URL", orchestrator_endpoint.split("/v1/", 1)[0]
    )
    st.info(
        "Review the scientific match here. If it is worth contacting the authors, "
        "send it to Email drafts; LocalOrchestrator will prepare editable text and never send it."
    )
    st.link_button("Open Email drafts →", orchestrator_ui)
    try:
        catalogue = contribution_catalogue.load()
    except contribution_catalogue.CatalogueError as exc:
        st.error(str(exc))
        return
    contributions = [item for item in catalogue["contributions"] if item.get("enabled", True)]
    papers = paper_store.load_all_papers()
    if not papers or not contributions:
        st.info("A stored paper and at least one enabled catalogue contribution are required.")
        return

    discovery_jobs = [job for job in index_jobs.list_jobs() if job["kind"] == "citation_discovery"]
    pending_count = sum(job["status"] in {"queued", "running"} for job in discovery_jobs)
    failed_count = sum(job["status"] == "failed" for job in discovery_jobs)
    all_opportunities = citation_opportunity_store.list_actionable()
    reviewable = [
        item for item in all_opportunities
        if item["classification"] in {"strong_citation_opportunity", "potentially_useful"}
        and item["status"] in {"proposed", "needs_review"}
    ]
    strong_count = sum(item["classification"] == "strong_citation_opportunity" for item in reviewable)
    potential_count = sum(item["classification"] == "potentially_useful" for item in reviewable)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Strong", strong_count)
    metric_cols[1].metric("Potential", potential_count)
    metric_cols[2].metric("Checks running", pending_count)
    metric_cols[3].metric("Failed checks", failed_count)
    metric_cols[4].metric("Your works", len(contributions))

    by_id = {item["id"]: item for item in catalogue["contributions"]}
    papers_by_id = {paper.get("arxiv_id", ""): paper for paper in papers}
    paper_lookup = st.text_input(
        "Find a paper",
        placeholder="ArXiv ID, URL, or title",
        key="citation_paper_lookup",
    ).strip()

    with st.expander("Recheck a paper", expanded=False):
        paper_options = {_paper_label(paper): paper for paper in sorted(papers, key=_published_sort_key, reverse=True)}
        paper_label = st.selectbox("Paper", list(paper_options), key="citation_paper")
        paper = paper_options[paper_label]
        if st.button("Queue recheck", key="analyse_citation"):
            try:
                paper_id = paper.get("arxiv_id", "")
                full_text = _get_paper_text(paper_id, paper)
                job_id = citation_discovery.enqueue_paper(paper, full_text, force=True)
            except Exception as exc:
                st.error(f"Could not queue recheck: {exc}")
            else:
                st.success(f"Queued {job_id[:8]}.")

    completed = [
        item for item in all_opportunities
        if item not in reviewable and item["status"] in {"confirmed", "exported"}
    ]
    if paper_lookup:
        lookup_id = citation_opportunities.arxiv_id_from_input(paper_lookup)
        if lookup_id:
            displayed = citation_opportunity_store.list_actionable(
                limit=500,
                paper_id=lookup_id,
            )
        else:
            query = paper_lookup.casefold()
            displayed = [
                item for item in all_opportunities
                if query in item["paper_id"].casefold()
                or query in str(
                    papers_by_id.get(item["paper_id"], {}).get("title", "")
                ).casefold()
            ]
        empty_message = f"No citation opportunities found for {paper_lookup}."
    else:
        view = st.radio(
            "View",
            ["Strong matches", "All reviewable", "Sent to Email drafts"],
            horizontal=True,
            key="citation_opportunity_view",
        )
        if view == "Strong matches":
            displayed = [
                item for item in reviewable
                if item["classification"] == "strong_citation_opportunity"
            ]
        elif view == "All reviewable":
            displayed = reviewable
        else:
            displayed = completed
        empty_message = f"No opportunities in {view.lower()}."

    if displayed:
        summary_rows = []
        for opportunity in displayed:
            source_paper = papers_by_id.get(
                opportunity["paper_id"], {"title": opportunity["paper_id"]}
            )
            matched = by_id.get(
                opportunity["contribution_id"], {"name": opportunity["contribution_id"]}
            )
            summary_rows.append({
                "Match": "Strong" if opportunity["classification"] == "strong_citation_opportunity" else "Potential",
                "Paper": source_paper.get("title", opportunity["paper_id"]),
                "Your work": matched.get("name", opportunity["contribution_id"]),
                "Confidence": f"{opportunity['confidence']:.0%}",
                "Status": opportunity["status"].replace("_", " ").title(),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    if not displayed:
        st.info(empty_message)
    for opportunity in displayed:
        source_paper = papers_by_id.get(opportunity["paper_id"], {"title": opportunity["paper_id"], "authors": []})
        matched = by_id.get(opportunity["contribution_id"], {"name": opportunity["contribution_id"], "canonical_citations": []})
        match_label = "Strong" if opportunity["classification"] == "strong_citation_opportunity" else "Potential"
        panel_title = f"{match_label}: {source_paper.get('title')} | {matched.get('name')}"
        with st.expander(panel_title, expanded=match_label == "Strong"):
            st.markdown(f"[{source_paper.get('title')}](https://arxiv.org/abs/{opportunity['paper_id']})")
            st.metric("Confidence", f"{opportunity['confidence']:.0%}")
            citation = citation_opportunities.preferred_citation(matched)
            st.markdown(f"**Contribution:** {matched.get('name')}  \n**Preferred citation:** {citation.get('preferred_text') or citation.get('title', 'Not published')}" )
            st.markdown("**Paper evidence**")
            for evidence in opportunity.get("evidence", []):
                st.caption(evidence["locator"])
                st.info(evidence["quote"])
            st.markdown(f"**Rationale:** {opportunity['rationale']}")
            st.markdown(f"**Strongest counterargument:** {opportunity['counterargument']}")
            with st.expander("Reference check"):
                st.json(opportunity["reference_check"])
            contact = source_paper.get("corresponding_author")
            if isinstance(contact, dict) and contact.get("email"):
                st.success(
                    f"Public corresponding-author contact: {contact.get('name', 'Corresponding author')} "
                    f"({contact['email']})"
                )
            else:
                st.caption("A public corresponding-author email will be looked up from the arXiv source when you continue.")
            tone_note = st.text_input(
                "Optional drafting note",
                placeholder="For example: keep it brief; mention that I also develop Bilby",
                key=f"citation_tone_{opportunity['opportunity_id']}",
            )
            confirm_col, reject_col, review_col = st.columns(3)
            can_export = (
                opportunity["classification"] in {"strong_citation_opportunity", "potentially_useful"}
                and opportunity["status"] in {"proposed", "needs_review"}
            )
            if confirm_col.button(
                "Send to Email drafts →",
                key=f"confirm_citation_{opportunity['opportunity_id']}",
                disabled=not can_export,
            ):
                try:
                    if not contact:
                        with st.spinner("Looking for a public corresponding-author email in the arXiv source..."):
                            try:
                                contact = citation_contacts.find_public_contact(
                                    _get_paper_text(opportunity["paper_id"], source_paper),
                                    opportunity["paper_id"],
                                )
                            except Exception:
                                contact = None
                        if contact:
                            source_paper = {**source_paper, "corresponding_author": contact}
                            paper_store.update_paper(
                                opportunity["paper_id"], {"corresponding_author": contact}
                            )
                    citation_opportunity_store.update_status(opportunity["opportunity_id"], "confirmed")
                    confirmed = {**opportunity, "status": "confirmed"}
                    citation_opportunities.export_bundle(confirmed, source_paper, matched, tone_note=tone_note)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    contact_note = (
                        f" Public contact found: {contact['name']} <{contact['email']}>."
                        if contact else
                        " No public email was found, so you can enter one manually later."
                    )
                    st.success(f"Added to Email drafts.{contact_note}")
                    st.link_button("Continue in Email drafts →", orchestrator_ui)
            if reject_col.button("Not relevant", key=f"reject_citation_{opportunity['opportunity_id']}"):
                citation_opportunity_store.update_status(opportunity["opportunity_id"], "not_relevant")
                st.rerun()
            if review_col.button("Needs review", key=f"review_citation_{opportunity['opportunity_id']}"):
                citation_opportunity_store.update_status(opportunity["opportunity_id"], "needs_review")
                st.rerun()
            st.caption(f"Status: {opportunity['status']} | export: {opportunity.get('export_status') or 'not exported'}")
            if opportunity.get("export_error"):
                st.warning(opportunity["export_error"])


def main():
    init_session_state()
    _auto_sync_papers_from_store()
    render_header()

    workspaces = {
        "Inbox": render_inbox,
        "Semantic Search": render_vector_search,
        "Knowledge Graph": render_knowledge_graph,
        "Library": render_papers_list,
        "Assistant": render_assistant,
        "Lit Review": render_lit_review_builder,
        "Projects": render_projects,
        "Paper Ideas": render_paper_ideas,
        "Grant Ideas": render_grant_ideas,
        "Citation Opportunities": render_citation_opportunities,
        "Library Health": render_research_ops,
        "Schedule": render_schedule,
        "Profile": render_profile,
    }
    st.sidebar.header("Workspace")
    selected_workspace = st.sidebar.selectbox(
        "View",
        list(workspaces),
        key="workspace_view",
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    query, author, categories, max_results, date_from = render_sidebar()

    if query is not None:  # None means Search button was not pressed
        render_search_results(query, author, categories, max_results, date_from)

    workspaces[selected_workspace]()


if __name__ == "__main__":
    main()
