"""Evidence-first literature synthesis artifacts."""

from __future__ import annotations

import re

from app import retrieval


def _bibtex_key(paper: dict) -> str:
    authors = paper.get("authors") or ["unknown"]
    first = authors[0] if isinstance(authors, list) else str(authors).split(",")[0]
    surname = re.sub(r"[^A-Za-z0-9]", "", first.split()[-1]) or "unknown"
    year = str(paper.get("published", "n.d."))[:4]
    word = next(iter(re.findall(r"[A-Za-z0-9]+", paper.get("title", "paper"))), "paper")
    return f"{surname}{year}{word}"


def bibtex(papers: list[dict]) -> str:
    """Export selected library records as reproducible ArXiv BibTeX."""
    entries = []
    for paper in papers:
        paper_id = paper.get("arxiv_id", "")
        authors = paper.get("authors") or []
        author_text = " and ".join(authors) if isinstance(authors, list) else str(authors)
        title = str(paper.get("title", "Untitled")).replace("{", "").replace("}", "")
        year = str(paper.get("published", ""))[:4]
        entries.append(
            "\n".join([
                f"@misc{{{_bibtex_key(paper)},",
                f"  title = {{{title}}},",
                f"  author = {{{author_text}}},",
                f"  year = {{{year}}},",
                f"  eprint = {{{paper_id}}},",
                "  archivePrefix = {arXiv}",
                "}",
            ])
        )
    return "\n\n".join(entries)


def claim_matrix(query: str, *, vector_db, paper_ids: list[str], limit: int = 20) -> list[dict]:
    """Build an auditable evidence matrix from the hybrid corpus."""
    rows = []
    for item in retrieval.hybrid_search(query, vector_db=vector_db, paper_ids=paper_ids, limit=limit):
        sentences = re.split(r"(?<=[.!?])\s+", item.get("text", "").strip())
        claim = next((sentence for sentence in sentences if len(sentence) >= 40), item.get("text", ""))
        rows.append({
            "paper_id": item.get("paper_id", ""),
            "source_type": item.get("kind", ""),
            "claim_or_evidence": claim[:700],
            "citation": retrieval.citation_label(item),
            "source_url": retrieval.source_url(item),
            "retrieval_score": round(float(item.get("score", 0)), 5),
        })
    return rows
