"""Reusable Library presentation controls."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st


def paginated_rows(frame: pd.DataFrame, *, state_key: str = "library") -> pd.DataFrame:
    page_size = st.selectbox(
        "Papers per page",
        [10, 20, 50],
        index=1,
        key=f"{state_key}_page_size",
    )
    total_pages = max(1, math.ceil(len(frame) / page_size))
    page_state = f"{state_key}_page"
    current_page = min(int(st.session_state.get(page_state, 1)), total_pages)
    st.session_state[page_state] = current_page
    previous, page_label, following = st.columns([1, 3, 1])
    if previous.button("<", help="Previous page", disabled=current_page <= 1, use_container_width=True):
        st.session_state[page_state] = current_page - 1
        st.rerun()
    page_label.markdown(
        f"<p style='text-align:center'>Page {current_page} of {total_pages} &middot; {len(frame)} papers</p>",
        unsafe_allow_html=True,
    )
    if following.button(">", help="Next page", disabled=current_page >= total_pages, use_container_width=True):
        st.session_state[page_state] = current_page + 1
        st.rerun()
    start = (current_page - 1) * page_size
    return frame.iloc[start:start + page_size]


def render_summary_provenance(paper: dict) -> None:
    provenance = paper.get("summary_provenance", {})
    if not provenance:
        st.caption("Legacy summary - generation provenance unavailable")
        return
    generated = str(provenance.get("generated_at", ""))[:19].replace("T", " ")
    st.caption(
        f"{provenance.get('status', 'unknown')} - "
        f"{provenance.get('provider', 'unknown')} / {provenance.get('model', 'unknown')}"
        + (f" - {generated} UTC" if generated else "")
    )
    if provenance.get("status") == "fallback":
        st.warning("The model failed and this summary used deterministic fallback text.")
