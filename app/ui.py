"""Streamlit frontend for the paper wiki application."""

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
from pathlib import Path
from typing import List, Dict
import time

from app.arxiv_client import ArxivClient
from app.pdf_extractor import PDFExtractor
from app.summarizer import PaperSummarizer
from app.vector_db import PaperVectorDB
from app.knowledge_graph import KnowledgeGraph


def init_session_state():
    """Initialize session state variables."""
    if "papers" not in st.session_state:
        st.session_state.papers = []
    if "knowledge_graph" not in st.session_state:
        st.session_state.kg = KnowledgeGraph()
    if "vector_db" not in st.session_state:
        st.session_state.vdb = PaperVectorDB()
    if "arxiv_client" not in st.session_state:
        st.session_state.arxiv = ArxivClient()
    if "summarizer" not in st.session_state:
        st.session_state.summarizer = PaperSummarizer()
    if "pdf_extractor" not in st.session_state:
        st.session_state.pdf_extractor = PDFExtractor()


def render_header():
    """Render the app header."""
    st.title("📚 ArXiv Paper Wiki")
    st.markdown(
        """
        Discover, summarize, and explore connections between research papers.
        Search semantically, browse by category, and visualize knowledge connections.
        """
    )


def render_sidebar():
    """Render the sidebar with search and filters."""
    st.sidebar.header("🔍 Search & Filters")

    # Search query
    query = st.sidebar.text_input("Search papers...", placeholder="e.g., transformer attention mechanism")

    # Category filters
    st.sidebar.markdown("**Categories**")
    categories = st.sidebar.multiselect(
        "Select categories:",
        options=[
            "cs.LG",
            "cs.CL",
            "cs.CV",
            "cs.AI",
            "physics.hep-th",
            "astro-ph.CO",
            "q-bio.QM",
            "q-fin.CP",
        ],
        default=[],
    )

    # Max results
    max_results = st.sidebar.slider("Max results", 5, 50, 20)

    # Action button
    if st.sidebar.button("🚀 Search", type="primary"):
        return query, categories, max_results

    return None, categories, max_results


def render_paper_card(paper: Dict, show_actions: bool = True):
    """Render a paper card."""
    with st.expander(
        f"**{paper['title']}**",
        expanded=False,
    ):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"### {paper['title']}")
            authors = ", ".join(paper.get("authors", [])[:3])
            if len(paper.get("authors", [])) > 3:
                authors += f" +{len(paper['authors']) - 3}"
            st.caption(f"👤 {authors}")
            st.caption(f"📅 {paper.get('published', 'N/A')}")
            st.caption(f"📂 {' | '.join(paper.get('categories', []))}")

            st.markdown("**Summary**")
            st.write(paper.get("summary", "No summary available."))

        with col2:
            if show_actions:
                if st.button("📥 Download PDF", key=f"download_{paper['id']}"):
                    pdf_path = st.session_state.arxiv.get_pdf_path(
                        type("Result", (), {"pdf_url": paper.get("pdf_url", "")})
                    )
                    st.success(f"PDF saved to {pdf_path}")

                if st.button("🔗 View Connections", key=f"connections_{paper['id']}"):
                    st.session_state.selected_paper = paper["id"]


def render_search_results(query: str, categories: List[str], max_results: int):
    """Search and display results from ArXiv."""
    with st.spinner("Searching ArXiv..."):
        papers = st.session_state.arxiv.search(
            query=query, max_results=max_results, categories=categories
        )

        if not papers:
            st.info("No papers found. Try different keywords or categories.")
            return

        st.markdown(f"### Found {len(papers)} papers")

        for result in papers:
            metadata = st.session_state.arxiv.get_paper_metadata(result)

            # Check if paper already exists
            if not st.session_state.vdb.paper_exists(metadata["arxiv_id"]):
                # Summarize the paper
                pdf_path = st.session_state.arxiv.get_pdf_path(result)
                full_text = st.session_state.pdf_extractor.extract_first_n_pages(
                    pdf_path, n_pages=3
                )
                summary = st.session_state.summarizer.summarize(full_text)

                # Add to vector DB and knowledge graph
                st.session_state.vdb.add_paper(
                    paper_id=metadata["arxiv_id"],
                    title=metadata["title"],
                    summary=summary,
                    metadata=metadata,
                )

                st.session_state.kg.add_paper(metadata["arxiv_id"], metadata)
                st.session_state.kg.connect_by_category(
                    metadata["arxiv_id"], metadata["categories"]
                )
                st.session_state.kg.connect_by_author(
                    metadata["arxiv_id"], metadata["authors"]
                )

                st.session_state.papers.append(metadata)

            # Display paper card
            render_paper_card(metadata)


def render_vector_search():
    """Render semantic search from vector database."""
    st.markdown("---")
    st.header("🔎 Semantic Search")

    semantic_query = st.text_input(
        "Describe what you're looking for...", placeholder="e.g., papers about climate change modeling"
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
                st.info("No papers found in the database. Try searching ArXiv first.")


def render_knowledge_graph():
    """Render the knowledge graph visualization."""
    st.markdown("---")
    st.header("🕸️ Knowledge Graph")

    if st.button("🔄 Visualize Connections"):
        if len(st.session_state.kg.get_all_paper_ids()) < 2:
            st.info("Add more papers to see connections.")
            return

        # Create subgraph of recent papers
        G = st.session_state.kg.graph
        nodes = list(G.nodes())[:50]  # Limit for performance
        subgraph = G.subgraph(nodes)

        # Calculate node sizes based on degree
        degrees = dict(subgraph.degree())
        node_sizes = [degrees[n] * 500 for n in subgraph.nodes()]

        # Create edge list
        edges = list(subgraph.edges())

        # Prepare data for visualization
        node_df = pd.DataFrame(
            {
                "id": list(subgraph.nodes()),
                "size": node_sizes,
                "label": [G.nodes[n].get("title", n)[:30] for n in subgraph.nodes()],
            }
        )

        edge_df = pd.DataFrame(
            edges, columns=["source", "target"]
        )

        # Create figure
        fig = px.treemap(
            node_df,
            path=["id"],
            values="size",
            color_continuous_scale="Viridis",
            title="Paper Connections (by category/author)",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show connection details
        st.markdown("**Connection Statistics**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Papers", len(nodes))
        with col2:
            st.metric("Connections", len(edges))
        with col3:
            st.metric("Avg Connections", f"{len(edges)/len(nodes):.1f}" if nodes else "0")


def render_papers_list():
    """Render list of all stored papers."""
    st.markdown("---")
    st.header("📋 All Papers")

    if not st.session_state.papers:
        st.info("No papers stored yet. Use the search to add papers.")
        return

    # Create dataframe
    df = pd.DataFrame(st.session_state.papers)

    # Display stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Papers", len(df))
    with col2:
        st.metric("Categories", df["categories"].explode().nunique())

    # Filter by category
    selected_category = st.multiselect(
        "Filter by category:",
        options=df["categories"].explode().unique().tolist(),
        default=[],
    )

    if selected_category:
        mask = df["categories"].apply(
            lambda x: any(cat in x for cat in selected_category)
        )
        df = df[mask]

    # Display papers in table
    display_df = df[["title", "authors", "published"]].head(20)
    st.dataframe(display_df, use_container_width=True)


def main():
    """Main application."""
    init_session_state()
    render_header()

    # Render sidebar and get search parameters
    query, categories, max_results = render_sidebar()

    if query or categories:
        render_search_results(query, categories, max_results)

    # Render different sections
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "🕸️ Connections", "📚 Library"])

    with tab1:
        render_vector_search()

    with tab2:
        render_knowledge_graph()

    with tab3:
        render_papers_list()


if __name__ == "__main__":
    main()
