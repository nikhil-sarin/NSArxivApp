"""Streamlit frontend for the paper wiki application."""

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
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
    """Render sidebar search controls. Returns (query, categories, max_results, date_from) or Nones."""
    st.sidebar.header("Search & Filters")

    query = st.sidebar.text_input("Search query", placeholder="e.g., neutron star merger")

    st.sidebar.markdown("**Categories**")
    categories = st.sidebar.multiselect(
        "Select categories:",
        options=[
            "cs.LG", "cs.CL", "cs.CV", "cs.AI",
            "astro-ph.HE", "astro-ph.CO", "astro-ph.GA",
            "physics.hep-th", "gr-qc",
            "q-bio.QM", "q-fin.CP",
        ],
        default=[],
    )

    max_results = st.sidebar.slider("Max results", 5, 50, 20)

    st.sidebar.markdown("**Date filter**")
    date_option = st.sidebar.selectbox(
        "Submitted since:",
        ["All time", "Today", "Last 7 days", "Last 30 days"],
        index=0,
    )

    date_from: Optional[datetime] = None
    now = datetime.now(timezone.utc)
    if date_option == "Today":
        date_from = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif date_option == "Last 7 days":
        date_from = now - timedelta(days=7)
    elif date_option == "Last 30 days":
        date_from = now - timedelta(days=30)

    if st.sidebar.button("Search", type="primary"):
        return query, categories, max_results, date_from

    return None, categories, max_results, date_from


def _authors_str(paper: Dict, max_shown: int = 3) -> str:
    """Return a display string for authors, handling both list and string formats."""
    authors = paper.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    shown = ", ".join(authors[:max_shown])
    if len(authors) > max_shown:
        shown += f" +{len(authors) - max_shown}"
    return shown


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

    chroma_meta = {
        k: v for k, v in {
            **metadata,
            "summary": summary,
            "authors": ", ".join(metadata.get("authors", [])),
            "categories": ", ".join(metadata.get("categories", [])),
        }.items()
        if v is not None and not isinstance(v, list)
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


def _fetch_text(result, n_pages: int = 3) -> str:
    """Download PDF and extract text (safe to run in a thread)."""
    pdf_path = st.session_state.arxiv.get_pdf_path(result)
    return st.session_state.pdf_extractor.extract_first_n_pages(pdf_path, n_pages=n_pages)


def render_search_results(query: str, categories: List[str], max_results: int, date_from: Optional[datetime]):
    with st.spinner("Searching ArXiv..."):
        papers = st.session_state.arxiv.search(
            query=query, max_results=max_results, categories=categories, date_from=date_from
        )

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
            futures = {pool.submit(_fetch_text, result): (result, meta) for result, meta in new_results}
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


def render_knowledge_graph():
    st.header("Knowledge Graph")
    if st.button("Visualize Connections"):
        all_ids = st.session_state.kg.get_all_paper_ids()
        if len(all_ids) < 2:
            st.info("Add more papers to see connections.")
            return

        G = st.session_state.kg.graph
        nodes = list(G.nodes())[:50]
        subgraph = G.subgraph(nodes)
        degrees = dict(subgraph.degree())

        node_df = pd.DataFrame({
            "id": list(subgraph.nodes()),
            "size": [max(degrees[n] * 500, 100) for n in subgraph.nodes()],
            "label": [G.nodes[n].get("title", n)[:40] for n in subgraph.nodes()],
        })
        edges = list(subgraph.edges())

        fig = px.treemap(
            node_df,
            path=["id"],
            values="size",
            color_continuous_scale="Viridis",
            title="Paper Connections (by category / author)",
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Papers shown", len(nodes))
        col2.metric("Connections", len(edges))
        col3.metric("Avg connections", f"{len(edges)/len(nodes):.1f}" if nodes else "0")


def _regenerate_summary(paper: Dict, detailed: bool = False):
    """Re-summarize a single paper and update all stores."""
    pid = paper.get("arxiv_id", paper.get("id", ""))
    n_pages = 6 if detailed else 3
    # Find the PDF
    pdf_path = st.session_state.arxiv.get_pdf_path_by_id(pid)
    if pdf_path is None or not pdf_path.exists():
        # Re-download
        import arxiv as _arxiv
        result = next(st.session_state.arxiv.client.results(
            _arxiv.Search(query=f"id:{pid}", max_results=1)
        ), None)
        if result is None:
            st.error(f"Could not find paper {pid} on ArXiv.")
            return
        pdf_path = st.session_state.arxiv.get_pdf_path(result)

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

    # Regenerate all button
    detailed_all = col3.checkbox("Detailed mode", key="regen_detailed_all")
    if col3.button("Regenerate all summaries"):
        progress = st.progress(0, text="Regenerating summaries...")
        all_papers = paper_store.load_all_papers()
        for i, p in enumerate(all_papers):
            progress.progress((i + 1) / len(all_papers), text=f"Summarizing {i+1}/{len(all_papers)}: {p.get('title','')[:50]}...")
            _regenerate_summary(p, detailed=detailed_all)
        progress.empty()
        st.session_state.papers = paper_store.load_all_papers()
        st.success("All summaries regenerated.")

    st.markdown("---")

    if "categories" in df.columns:
        all_cats = df["categories"].explode().dropna().unique().tolist()
        selected = st.multiselect("Filter by category:", options=all_cats, default=[])
        if selected:
            mask = df["categories"].apply(lambda x: any(c in x for c in selected))
            df = df[mask]

    # Per-paper cards with regenerate button
    for _, row in df.head(50).iterrows():
        paper = row.to_dict()
        pid = paper.get("arxiv_id", "")
        with st.expander(f"**{paper.get('title', pid)}**", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"Authors: {_authors_str(paper)}")
                st.caption(f"Published: {paper.get('published', 'N/A')}")
                st.markdown("**Summary**")
                st.write(paper.get("summary", "No summary available."))
            with col2:
                detailed = st.checkbox("Detailed", key=f"det_{pid}")
                if st.button("Regenerate summary", key=f"regen_{pid}"):
                    with st.spinner("Summarizing..."):
                        new_summary = _regenerate_summary(paper, detailed=detailed)
                    if new_summary:
                        paper["summary"] = new_summary
                        st.success("Done.")

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


def main():
    init_session_state()
    render_header()

    query, categories, max_results, date_from = render_sidebar()

    if query is not None:  # None means button was not pressed
        render_search_results(query, categories, max_results, date_from)

    tab1, tab2, tab3, tab4 = st.tabs(["Semantic Search", "Knowledge Graph", "Library", "Schedule"])

    with tab1:
        render_vector_search()
    with tab2:
        render_knowledge_graph()
    with tab3:
        render_papers_list()
    with tab4:
        render_schedule()


if __name__ == "__main__":
    main()
