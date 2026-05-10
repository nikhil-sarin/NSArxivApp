"""Streamlit frontend for the paper wiki application."""

import os
import re
import json
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.arxiv_client import ArxivClient
from app.pdf_extractor import PDFExtractor
from app.summarizer import PaperSummarizer
from app.vector_db import PaperVectorDB
from app.knowledge_graph import KnowledgeGraph
from app import paper_store
from app.tex_extractor import fetch_html_text
from app import researcher_profile
from app import idea_store
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


def init_session_state():
    """Initialize session state variables, loading persisted papers on first run."""
    st.session_state.vdb = get_vector_db()
    st.session_state.summarizer = get_summarizer()
    st.session_state.arxiv = get_arxiv_client()
    st.session_state.pdf_extractor = get_pdf_extractor()

    if "knowledge_graph" not in st.session_state:
        st.session_state.kg = KnowledgeGraph()

    # Load persisted papers into session state on first run
    if "papers" not in st.session_state:
        stored = paper_store.load_all_papers()
        st.session_state.papers = stored
        # Rebuild knowledge graph from stored papers
        for p in stored:
            pid = p.get("arxiv_id", p.get("id", ""))
            if pid:
                st.session_state.kg.add_paper(pid, p)
                st.session_state.kg.connect_by_category(pid, p.get("categories", []))
                st.session_state.kg.connect_by_author(pid, p.get("authors", []))


def render_header():
    st.title("ArXiv Paper Wiki")
    st.markdown("Discover, summarize, and explore connections between research papers.")


def render_sidebar():
    """Render sidebar search controls. Returns (query, author, categories, max_results, date_from) or Nones."""
    st.sidebar.header("Search & Filters")

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

    max_results = st.sidebar.slider("Max results", 5, 100, 20)

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

    if st.sidebar.button("Search", type="primary"):
        return query, author, categories, max_results, date_from

    # Provider status
    st.sidebar.markdown("---")
    summ_provider = os.getenv("SUMMARIZER_PROVIDER", "ollama")
    summ_model = os.getenv("LLM_MODEL", "") or {
        "ollama": os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
        "gemini": "gemini-2.0-flash",
        "anthropic": "claude-3-5-haiku-20241022",
        "openai": "gpt-4o-mini",
    }.get(summ_provider, summ_provider)
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    chat_model = os.getenv("CHAT_LLM_MODEL", "gemini-2.0-flash") if has_gemini else summ_model
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
            _ingest_by_arxiv_id(arxiv_input.strip())
        else:
            st.sidebar.warning("Please enter an ArXiv URL or ID.")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Summarization:** {summ_provider} / {summ_model}")
    st.sidebar.caption(f"**Chat:** {'gemini' if has_gemini else summ_provider} / {chat_model}")

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


def _format_notes_for_display(paper: Dict) -> str:
    lines = _notes_lines(paper)
    return "\n".join(f"- {line}" for line in lines)


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
                    import arxiv as _arxiv
                    result = next(st.session_state.arxiv.client.results(
                        _arxiv.Search(query=f"id:{pid}", max_results=1)
                    ), None)
                    if result:
                        pdf_path = st.session_state.arxiv.get_pdf_path(result)
                        st.success(f"Saved to {pdf_path}")


def _store_paper(pid: str, metadata: Dict, summary: str):
    """Persist a paper to JSON store, vector DB, and knowledge graph."""
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
        st.session_state.vdb.collection.delete(ids=[pid])
    except Exception:
        pass
    st.session_state.vdb.add_paper(
        paper_id=pid,
        title=metadata["title"],
        summary=summary,
        metadata=chroma_meta,
    )

    st.session_state.kg.add_paper(pid, metadata)
    st.session_state.kg.connect_by_category(pid, metadata.get("categories", []))
    st.session_state.kg.connect_by_author(pid, metadata.get("authors", []))


def _parse_arxiv_id(raw: str) -> str:
    """Extract a clean arxiv ID from a URL or bare ID string."""
    raw = raw.strip().rstrip("/")
    # Handle URLs like https://arxiv.org/abs/2301.12345 or arxiv.org/pdf/2301.12345v2
    for prefix in ("abs/", "pdf/", "html/", "src/"):
        if prefix in raw:
            raw = raw.split(prefix)[-1]
    # Strip version suffix for lookup but keep original
    return raw.split("v")[0] if re.match(r"^\d{4}\.\d{4,5}", raw.split("v")[0]) else raw


def _ingest_by_arxiv_id(raw_input: str):
    """Fetch, summarize, and store a paper given an ArXiv URL or ID."""
    import arxiv as _arxiv

    arxiv_id = _parse_arxiv_id(raw_input)

    if paper_store.paper_exists(arxiv_id) or paper_store.paper_exists(arxiv_id + "v1"):
        st.sidebar.info("Paper already in library.")
        return

    with st.sidebar:
        with st.spinner("Fetching paper..."):
            result = next(
                st.session_state.arxiv.client.results(
                    _arxiv.Search(id_list=[arxiv_id], max_results=1)
                ),
                None,
            )
        if result is None:
            st.error(f"Could not find paper {arxiv_id} on ArXiv.")
            return

        metadata = st.session_state.arxiv.get_paper_metadata(result)
        pid = metadata["arxiv_id"]

        with st.spinner("Loading paper text..."):
            text = _fetch_text(result, pid)

        with st.spinner("Summarizing..."):
            summary = st.session_state.summarizer.summarize(text)

        metadata["summary"] = summary
        _store_paper(pid, metadata, summary)
        st.session_state.papers.append(metadata)
        st.success(f"Added: {metadata['title'][:60]}...")


def _fetch_text(result, arxiv_id: str) -> str:
    """Get paper text — HTML source preferred, PDF fallback (safe to run in a thread)."""
    cache_dir = Path("data/papers")
    text = fetch_html_text(arxiv_id, cache_dir)
    if text:
        return text
    pdf_path = st.session_state.arxiv.get_pdf_path(result)
    return st.session_state.pdf_extractor.extract_first_n_pages(pdf_path, n_pages=6)


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
    new_results = []
    stored_data = paper_store._load()
    all_metadata = []
    for result in papers:
        metadata = st.session_state.arxiv.get_paper_metadata(result)
        pid = metadata["arxiv_id"]
        if pid in stored_data:
            metadata["summary"] = stored_data[pid].get("summary", "")
        else:
            new_results.append((result, metadata))
        all_metadata.append(metadata)

    st.markdown(f"### Found {len(papers)} papers ({len(new_results)} new)")

    if new_results:
        # Step 1: download all PDFs in parallel
        progress = st.progress(0, text="Downloading PDFs...")
        texts = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_fetch_text, result, meta["arxiv_id"]): (result, meta) for result, meta in new_results}
            for i, future in enumerate(as_completed(futures)):
                _, meta = futures[future]
                try:
                    texts[meta["arxiv_id"]] = future.result()
                except Exception as e:
                    texts[meta["arxiv_id"]] = ""
                progress.progress((i + 1) / len(new_results), text=f"Downloaded {i+1}/{len(new_results)} PDFs")

        # Step 2: summarize sequentially (Ollama is single-threaded)
        progress.progress(0, text="Summarizing...")
        for i, (result, metadata) in enumerate(new_results):
            pid = metadata["arxiv_id"]
            progress.progress((i + 1) / len(new_results), text=f"Summarizing {i+1}/{len(new_results)}: {metadata['title'][:50]}...")
            summary = st.session_state.summarizer.summarize(texts.get(pid, ""))
            metadata["summary"] = summary
            _store_paper(pid, metadata, summary)
            st.session_state.papers.append(metadata)

        progress.empty()
        st.success(f"Added {len(new_results)} new papers to your library.")

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
            has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            for p in selected_papers:
                pid = p.get("arxiv_id", "")
                text = _get_paper_text(pid, p)
                parts.append(_paper_source_context(pid, p, text, max_chunks=None if has_gemini else 4))
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
                    reply = st.session_state.summarizer.dispatch_chat_gemini(system_prompt, messages)
                except Exception as e:
                    reply = f"Error: {e}"
            st.write(reply)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        st.session_state["multi_chat_history"] = history


def render_vector_search():
    st.header("Semantic Search")
    semantic_query = st.text_input(
        "Describe what you're looking for...",
        placeholder="e.g., papers about gravitational wave detection methods",
    )
    if semantic_query:
        with st.spinner("Searching..."):
            results = st.session_state.vdb.search(semantic_query, top_k=10)
        if results:
            st.markdown(f"### Found {len(results)} relevant papers")
            for result in results:
                metadata = result["metadata"]
                metadata["summary"] = result["summary"]
                render_paper_card(metadata, show_actions=False)
        else:
            st.info("No papers in the database yet. Use the ArXiv search to add papers.")

    st.markdown("---")
    render_multi_paper_chat()


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

    filtered = sorted(filtered, key=lambda p: str(p.get("published", "")), reverse=True)[:max_nodes]
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
                vec = st.session_state.vdb.get_embedding(pid)
                if vec is None:
                    continue
                for result in st.session_state.vdb.search_by_vector(vec, top_k=5, exclude_id=pid):
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
    """Get full paper text. Tries ArXiv HTML first, falls back to PDF extraction."""
    cache_dir = Path("data/papers")
    # Try HTML source first — full paper, clean text, no page limit
    text = fetch_html_text(pid, cache_dir)
    if text:
        return text
    # Fallback: PDF extraction (may be incomplete)
    print(f"[chat] HTML unavailable for {pid}, falling back to PDF")
    pdf_path = st.session_state.arxiv.get_pdf_path_by_id(pid)
    if pdf_path is None or not pdf_path.exists():
        import arxiv as _arxiv
        result = next(st.session_state.arxiv.client.results(
            _arxiv.Search(query=f"id:{pid}", max_results=1)
        ), None)
        if result is None:
            return ""
        pdf_path = st.session_state.arxiv.get_pdf_path(result)
    return st.session_state.pdf_extractor.extract_text(pdf_path)


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
    chat_key = f"chat_text_{pid}"
    source_key = f"chat_text_source_{pid}"
    if chat_key not in st.session_state or st.session_state.get(f"{chat_key}_v") != 2:
        with st.spinner("Loading full paper text..."):
            text = _get_paper_text(pid, paper)
            st.session_state[chat_key] = text
            st.session_state[f"{chat_key}_v"] = 2  # bump to invalidate old PDF-only cache
            # Record source for display
            clean_id = pid.split("v")[0]
            from pathlib import Path as _Path
            html_cached = (_Path("data/papers") / f"{clean_id.replace('.','_')}_html.txt").exists()
            st.session_state[source_key] = "HTML (full paper)" if html_cached else "PDF"

    paper_text = st.session_state[chat_key]
    # Gemini (1M token context) gets the full text.
    # Ollama fallback is limited to its context window.
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if not has_gemini:
        char_limit = int(os.getenv("OLLAMA_NUM_CTX", "32768")) * 3
        if len(paper_text) > char_limit:
            paper_text = paper_text[:char_limit] + "\n...[truncated]"
    max_chunks = None if has_gemini else 10

    system_prompt = (
        "You are a research assistant. Answer using only the supplied paper source unless the user explicitly asks "
        "for outside knowledge. Cite evidence for substantive claims using the provided source labels, for example "
        f"[arXiv:{pid} chunk 3]. If the source does not support an answer, say that clearly. "
        "When useful, end with a short 'Evidence' section listing the cited chunks.\n\n"
        + _paper_source_context(pid, paper, paper_text, max_chunks=max_chunks)
    )

    history_key = f"chat_history_{pid}"
    history = st.session_state.get(history_key, [])
    messages = history + [{"role": "user", "content": user_message}]

    try:
        reply = st.session_state.summarizer.dispatch_chat_gemini(system_prompt, messages)
    except Exception as e:
        reply = f"Error: {e}"

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    st.session_state[history_key] = history

    return reply


def render_paper_chat(pid: str, paper: Dict):
    """Render an inline chat panel for a paper."""
    history_key = f"chat_history_{pid}"
    history = st.session_state.get(history_key, [])

    source = st.session_state.get(f"chat_text_source_{pid}", "")
    chars = len(st.session_state.get(f"chat_text_{pid}", ""))
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    chat_model = os.getenv("CHAT_LLM_MODEL", "gemini-2.0-flash") if has_gemini else st.session_state.summarizer._active_model()
    st.markdown("**Chat with this paper**")
    if chars:
        st.caption(f"Context: {chars:,} chars — source: {source} — model: {chat_model}")

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

    # Try HTML first (full paper), fall back to PDF
    text = fetch_html_text(pid, Path("data/papers"))
    if not text:
        pdf_path = st.session_state.arxiv.get_pdf_path_by_id(pid)
        if pdf_path is None or not pdf_path.exists():
            import arxiv as _arxiv
            result = next(st.session_state.arxiv.client.results(
                _arxiv.Search(query=f"id:{pid}", max_results=1)
            ), None)
            if result is None:
                st.error(f"Could not find paper {pid} on ArXiv.")
                return
            pdf_path = st.session_state.arxiv.get_pdf_path(result)
        n_pages = 6 if detailed else 3
        text = st.session_state.pdf_extractor.extract_first_n_pages(pdf_path, n_pages=n_pages)

    summary = st.session_state.summarizer.summarize(text, max_length=500 if detailed else 300, detailed=detailed)
    _store_paper(pid, paper, summary)

    # Update session state papers list
    for p in st.session_state.papers:
        if p.get("arxiv_id") == pid:
            p["summary"] = summary
            break
    return summary


def render_papers_list():
    st.header("All Papers")
    papers = st.session_state.papers
    if not papers:
        st.info("No papers stored yet. Use the search to add papers.")
        return

    df = pd.DataFrame(papers)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Papers", len(df))
    col2.metric("Categories", df["categories"].explode().nunique() if "categories" in df.columns else 0)

    # Bulk summary buttons
    detailed_all = col3.checkbox("Detailed mode", key="regen_detailed_all")
    btn_col1, btn_col2 = col3.columns(2)
    if btn_col1.button("Regenerate all"):
        progress = st.progress(0, text="Regenerating summaries...")
        all_papers = paper_store.load_all_papers()
        for i, p in enumerate(all_papers):
            progress.progress((i + 1) / len(all_papers), text=f"Summarizing {i+1}/{len(all_papers)}: {p.get('title','')[:50]}...")
            _regenerate_summary(p, detailed=detailed_all)
        progress.empty()
        st.session_state.papers = paper_store.load_all_papers()
        st.success("All summaries regenerated.")
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
        df["_pub_sort"] = pd.to_datetime(df["published"], errors="coerce")
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
        paper = row.to_dict()
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
                        st.success("Done. Summary updated above.")
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
                        st.session_state.vdb.delete_paper(pid)
                        st.session_state.papers = [
                            p for p in st.session_state.papers
                            if p.get("arxiv_id") != pid
                        ]
                        st.session_state.pop(confirm_key, None)
                        st.success("Deleted.")
                    if dcol2.button("Cancel", key=f"del_cancel_{pid}"):
                        st.session_state[confirm_key] = False
            if st.session_state.get(f"show_mlt_{pid}", False):
                vec = st.session_state.vdb.get_embedding(pid)
                if vec is not None:
                    similar = st.session_state.vdb.search_by_vector(vec, top_k=5, exclude_id=pid)
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


def render_schedule():
    """Show cron/launchd setup for daily automated fetch."""
    st.header("Scheduled Daily Fetch")
    st.markdown(
        "Set up an automated daily search that runs even when the app is closed. "
        "Results will be added to your local library and available next time you open the app."
    )

    col1, col2 = st.columns(2)
    with col1:
        sched_query = st.text_input("Search query for scheduled job", placeholder="neutron star kilonova")
        sched_cats = st.multiselect(
            "Categories",
            ["cs.LG", "cs.CL", "cs.CV", "astro-ph.HE", "astro-ph.CO", "gr-qc"],
        )
        sched_max = st.slider("Max results per run", 5, 50, 10)
        sched_hour = st.slider("Run at hour (24h, local time)", 0, 23, 7)

    with col2:
        app_dir = Path(".").resolve()
        python_path = subprocess.run(
            ["which", "python"], capture_output=True, text=True
        ).stdout.strip() or "python"

        cats_arg = " ".join(sched_cats) if sched_cats else ""
        fetch_cmd = (
            f"cd {app_dir} && {python_path} -m app.fetch_job"
            f" --query \"{sched_query}\""
            + (f" --categories {cats_arg}" if cats_arg else "")
            + f" --max-results {sched_max}"
        )

        st.markdown("**cron entry** (paste into `crontab -e`)")
        cron_line = f"0 {sched_hour} * * * {fetch_cmd} >> {app_dir}/data/fetch.log 2>&1"
        st.code(cron_line, language="bash")

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
                <string>{fetch_cmd} >> {app_dir}/data/fetch.log 2>&1</string>
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


def _build_library_context() -> str:
    """Build a compact context string from all stored papers for the assistant."""
    papers = paper_store.load_all_papers()
    if not papers:
        return ""
    lines = [f"The user has a library of {len(papers)} research papers:\n"]
    for p in papers:
        pid = p.get("arxiv_id", "")
        title = p.get("title", "Unknown")
        authors = _authors_str(p)
        published = str(p.get("published", ""))[:10]
        cats = ", ".join(p.get("categories", [])) if isinstance(p.get("categories"), list) else str(p.get("categories", ""))
        summary = p.get("summary", "").strip()
        notes = "\n".join(_notes_lines(p))
        lines.append(
            f"---\n"
            f"ID: {pid}\n"
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Published: {published}  Categories: {cats}\n"
            f"Summary: {summary}\n"
            + (f"Research notes:\n{notes}\n" if notes else "")
        )
    return "\n".join(lines)


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


def _assistant_call(system: str, messages: list) -> str:
    """Call Gemini (preferred) or fall back to configured provider."""
    try:
        return st.session_state.summarizer.dispatch_chat_gemini(system, messages)
    except Exception as e:
        return f"Error: {e}"


def render_lit_review_builder():
    st.header("Literature Review Builder")
    papers = paper_store.load_all_papers()
    if not papers:
        st.info("Your library is empty. Search for papers first.")
        return

    options = {_paper_label(p): p for p in sorted(papers, key=lambda x: str(x.get("published", "")), reverse=True)}
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

    if st.button("Generate literature review", type="primary", key="generate_lit_review"):
        profile_ctx = researcher_profile.to_context_string(researcher_profile.load())
        paper_ctx = _build_selected_papers_context(selected, include_notes=include_notes)
        focus_clause = f"Focus specifically on: {focus.strip()}." if focus.strip() else "Use the strongest common themes in the selected papers."
        system = (
            "You are a research assistant helping write accurate, useful literature reviews for an active researcher. "
            "Use the selected papers and the user's notes as the source of truth. Be specific about paper titles and authors. "
            "Do not invent claims that are not supported by the supplied summaries or notes.\n\n"
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
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    model_name = os.getenv("CHAT_LLM_MODEL", "gemini-2.0-flash") if has_gemini else st.session_state.summarizer._active_model()
    st.caption(f"{n} papers in library — model: {model_name}")

    mode = st.radio(
        "Mode:",
        ["Cross-library chat", "Research briefing", "Gap finder", "Project ideas"],
        horizontal=True,
        key="assistant_mode",
    )

    # Build library context (cached, invalidated when paper count changes)
    ctx_key = "assistant_library_context"
    ctx_n_key = "assistant_library_n"
    if st.session_state.get(ctx_n_key) != n:
        with st.spinner("Indexing library..."):
            st.session_state[ctx_key] = _build_library_context()
            st.session_state[ctx_n_key] = n

    library_context = st.session_state[ctx_key]

    # Prepend researcher profile to all assistant prompts
    profile = researcher_profile.load()
    profile_context = researcher_profile.to_context_string(profile)
    if profile_context:
        library_context = profile_context + "\n\n" + library_context

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
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = _assistant_call(system, history + [{"role": "user", "content": user_input}])
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
                    + library_context
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
            library_ctx = _build_library_context()
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

    library_ctx = _build_library_context()
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
    library_ctx = _build_library_context()
    base_context = (profile_ctx + "\n\n" if profile_ctx else "") + library_ctx

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


def main():
    init_session_state()
    render_header()

    query, author, categories, max_results, date_from = render_sidebar()

    if query is not None:  # None means Search button was not pressed
        render_search_results(query, author, categories, max_results, date_from)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Semantic Search", "Knowledge Graph", "Library",
        "Assistant", "Lit Review", "Paper Ideas", "Grant Ideas",
        "Schedule", "Profile",
    ])

    with tab1:
        render_vector_search()
    with tab2:
        render_knowledge_graph()
    with tab3:
        render_papers_list()
    with tab4:
        render_assistant()
    with tab5:
        render_lit_review_builder()
    with tab6:
        render_paper_ideas()
    with tab7:
        render_grant_ideas()
    with tab8:
        render_schedule()
    with tab9:
        render_profile()


if __name__ == "__main__":
    main()
