"""Semantic trend-cluster workspace."""

from __future__ import annotations

import streamlit as st

from app import paper_store, researcher_profile, trends


def render(vector_db) -> None:
    window = st.slider("Recent window (days)", 30, 365, 90, 30)
    profile_context = researcher_profile.to_context_string(researcher_profile.load())
    data_mtime = paper_store.STORE_PATH.stat().st_mtime_ns if paper_store.STORE_PATH.exists() else 0
    cache_key = (data_mtime, window, profile_context, vector_db.embedding_backend)
    if st.button("Refresh clusters", use_container_width=False):
        st.session_state.pop("semantic_trend_cache", None)
    cached = st.session_state.get("semantic_trend_cache")
    if not cached or cached.get("key") != cache_key:
        with st.spinner("Clustering recent papers..."):
            rows = trends.semantic_theme_clusters(
                paper_store.load_reading_papers(),
                interest_text=profile_context,
                encode=vector_db.embedder.encode,
                recent_days=window,
            )
        st.session_state.semantic_trend_cache = {"key": cache_key, "rows": rows}
    else:
        rows = cached["rows"]
    feedback = trends.load_feedback()
    rows = [row for row in rows if feedback.get(row["theme"]) != "mute"]
    if not rows:
        st.info("No profile-relevant paper clusters have enough support in this window.")
        return
    for row in rows:
        followed = feedback.get(row["theme"]) == "follow"
        heading = f"{row['theme']} - {row['paper_count']} papers - {row['relevance']:.0%} profile match"
        if followed:
            heading = "Following - " + heading
        with st.expander(heading):
            for example in row["examples"]:
                st.markdown(f"- [{example['title']}](https://arxiv.org/abs/{example['arxiv_id']})")
            follow_col, mute_col = st.columns(2)
            if follow_col.button(
                "Unfollow" if followed else "Follow",
                key=f"trend_follow_{row['theme']}",
                use_container_width=True,
            ):
                trends.save_feedback(row["theme"], "" if followed else "follow")
                st.rerun()
            if mute_col.button("Mute", key=f"trend_mute_{row['theme']}", use_container_width=True):
                trends.save_feedback(row["theme"], "mute")
                st.rerun()
