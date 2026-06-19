"""Generate cached guided-reading paper reports with the configured LLM."""

from __future__ import annotations

import base64
import math
import os
import re
from html import escape
from pathlib import Path

from app.paper_text import get_paper_text
from app.source_extractor import download_source, extract_abstract, extract_figures, extract_text

DEFAULT_REPORT_WORDS = 1800
MAX_SOURCE_CHARS = 80_000
MAX_FIGURES = 3
MAX_REPORT_ATTEMPTS = 2


class ReportUnavailable(RuntimeError):
    """The paper does not have enough readable content for a report."""


def _is_missing(value) -> bool:
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def _paper_title(paper: dict) -> str:
    title = paper.get("title")
    if _is_missing(title):
        title = paper.get("arxiv_id", "Untitled")
    return str(title)


def _paper_authors(paper: dict) -> str:
    authors = paper.get("authors", [])
    if _is_missing(authors):
        return ""
    if isinstance(authors, str):
        return authors
    return ", ".join(str(author) for author in authors)


def _paper_abs_url(paper: dict) -> str:
    arxiv_id = str(paper.get("arxiv_id", "")).split("v")[0]
    abs_url = paper.get("abs_url")
    if _is_missing(abs_url):
        return f"https://arxiv.org/abs/{arxiv_id}"
    return str(abs_url)


def _paper_abstract(paper: dict) -> str:
    abstract = paper.get("abstract", "")
    return "" if _is_missing(abstract) else str(abstract)


def _best_abstract(paper: dict, source_abstract: str) -> str:
    metadata_abstract = _paper_abstract(paper)
    if source_abstract:
        if not metadata_abstract:
            return source_abstract
        # Source abstracts from unfinished drafts can contain ~unit placeholder values
        # (e.g. "~mag", "~erg") where the author hadn't yet filled in numbers.
        # In that case the arXiv metadata abstract is always more useful.
        if _LATEX_PLACEHOLDER_RE.search(source_abstract):
            return metadata_abstract
        if len(metadata_abstract) < 0.7 * len(source_abstract):
            return source_abstract
        if metadata_abstract and metadata_abstract[-1].isalnum() and not metadata_abstract.endswith((".", "!", "?")):
            return source_abstract
    return metadata_abstract


def _trim_source(text: str, max_chars: int = MAX_SOURCE_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return (
        text[:head]
        + "\n\n[... report input truncated for length ...]\n\n"
        + text[-tail:]
    )


def _build_prompt(paper: dict, source_text: str, fig_list: str, abstract: str) -> tuple[str, str]:
    system = """I'm going to give you the source of a scientific paper together with a numbered list of its figures.
Produce a short guided-reading report of the paper.

Begin immediately with the first <h2> section heading. Do not write any preamble, greeting, summary of what you are about to do, or explanation of your approach.

Output the report as an HTML <body> fragment only — no <html>, <head>, <style>, or <body> tags, no code fences, no markdown.

Structure:
- Do NOT output the title, authors, or abstract. They are rendered separately.
- Output only the report body: roughly three to five sections that follow the paper's logical arc:
  motivation → background or methods → results or findings → take-aways.
- Adapt the section names to the paper itself.
- Always finish with a Take-aways section using a short <ul> with around 3-5 bullets in your own words.

Style rules:
- Be concise.
- The body should consist mostly of information-dense verbatim quotes from the paper inside <q>...</q>, joined by short framing paragraphs.
- Use one or two strong quotes per section rather than many weak quotes.
- Preserve inline LaTeX math and symbols exactly as they appear in the source.
- Do not include the references, acknowledgements, or appendix.
- Use <h2> for section headings, <h3> for sub-headings, <p> for framing, <q> for verbatim quotes, and <ul>/<li> for take-aways.
- Do not explain your process, do not ask clarifying questions, do not mention analysis, outline reconstruction, recommendations, or manuscript drafting.
- IMPORTANT: Every section must contain at least one direct verbatim quote from the paper source text, placed inside <q>...</q>. Do not use general knowledge — only content that appears in the provided source.

Figures:
- To embed a figure, place <!-- FIGURE:N --> inline where it should appear.
- Use at most 3 figures total.
- Choose figures that genuinely help the reader understand the paper.
- If figures are available, use at least one when it clarifies the paper."""
    user = (
        f"Title: {_paper_title(paper)}\n"
        f"Authors: {_paper_authors(paper)}\n"
        f"Abstract: {abstract}\n\n"
        f"Available figures:\n{fig_list or '(none)'}\n\n"
        f"Paper source/text:\n{source_text}"
    )
    return system, user


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$")
_FIGURE_MARKER_RE = re.compile(r"<!--\s*FIGURE:(\d+)\s*-->")
_WRAPPER_TAG_RE = re.compile(r"</?(?:html|head|body)[^>]*>", re.IGNORECASE)
_H1_RE = re.compile(r"<h1\b[^>]*>.*?</h1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<(?:h1|h2|h3|p|q|ul|li|figure|figcaption)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"\\section\{([^}]*)\}(.*?)(?=(?:\\section\{|\\end\{document\}))", re.DOTALL)
# Matches ~unit only when NOT preceded by a number, closing math delimiter ($, }), or dash.
# "$-19.2$~mag" is a real value; "of ~mag" with no preceding number is a placeholder.
_LATEX_PLACEHOLDER_RE = re.compile(r"(?<![0-9}$\-])~\s*(?:mag|erg|K|yr|pc|cm|Jy|Hz|eV|keV|MeV|GeV|TeV)\b", re.IGNORECASE)
_META_PHRASES = [
    "here is a detailed analysis",
    "analysis and structure reconstruction",
    "reconstructed manuscript outline",
    "recommendation for the author",
    "to be written last",
    "assuming its context",
    "i will organize this",
    "example draft",
    "the user has provided",
    "latex document",
    "scientific manuscript",
    "editorial suggestions",
    "overall grade",
    "summary of action items",
    "units consistency",
    "figure labels",
    "technical refinements",
    "structural strengths",
    "critical editorial suggestions",
    "since there is no question",
    "please provide the question",
    "here is a summary and analysis",
    "if you have a specific question",
    "no specific question was asked",
    "please let me know",
    "what would you like",
    "are you interested in",
    "a critique or suggestion",
    "i'd be happy to help",
    "could you please clarify",
    "main topics covered",
    "key details and findings",
    "inferred",
    "here is a summary of",
    "organized by theme",
    "note: the provided text",
    "provided text snippets",
    "grouped into logical",
    "groups these points",
]

_H2_RE_BODY = re.compile(r"<h2\b", re.IGNORECASE)


def _render_intro(paper: dict, abstract: str) -> str:
    authors = escape(_paper_authors(paper))
    # Replace LaTeX non-breaking spaces and strip common inline commands for display
    abstract = re.sub(r"~", " ", abstract)
    abstract = escape(abstract)
    intro = [
        f"<h1>{escape(_paper_title(paper))}</h1>",
    ]
    if authors:
        intro.append(f"<p>{authors}</p>")
    if abstract:
        intro.append(f"<p>{abstract}</p>")
    return "\n".join(intro)


def _markdownish_to_html(body: str) -> str:
    lines = [line.rstrip() for line in body.splitlines()]
    parts: list[str] = []
    bullets: list[str] = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            items = "".join(f"<li>{escape(item)}</li>" for item in bullets)
            parts.append(f"<ul>{items}</ul>")
            bullets = []

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_bullets()
            continue
        if line.startswith("### "):
            flush_bullets()
            parts.append(f"<h3>{escape(line[4:].strip())}</h3>")
            continue
        if line.startswith("## "):
            flush_bullets()
            parts.append(f"<h2>{escape(line[3:].strip())}</h2>")
            continue
        if line.startswith("* ") or line.startswith("- "):
            bullets.append(line[2:].strip())
            continue
        if re.match(r"^\d+\.\s+", line):
            bullets.append(re.sub(r"^\d+\.\s+", "", line))
            continue
        flush_bullets()
        parts.append(f"<p>{escape(line)}</p>")
    flush_bullets()
    return "\n".join(parts)


def _auto_insert_figures(body: str, figures: list[dict]) -> str:
    if not figures or _FIGURE_MARKER_RE.search(body):
        return body
    blocks = "".join(_figure_block(figure) for figure in figures[: min(2, len(figures))])
    return blocks + body


def _looks_like_meta_commentary(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _META_PHRASES)


def _clean_source_excerpt(text: str) -> str:
    """Lightly clean LaTeX-heavy source text for inline display."""
    text = re.sub(r"(?m)%.*$", "", text)
    text = re.sub(r"\\(cite|citep|citet|citealp|citeauthor)\*?(?:\[[^\]]*\]){0,2}\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:label|ref|pageref)\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection)\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:textit|textbf|emph|texttt|mathrm)\{([^{}]*)\}", r"\1", text)
    text = text.replace("~", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_sections(source_text: str) -> dict[str, str]:
    """Map top-level LaTeX section headings to their content."""
    return {
        title.strip().lower(): content.strip()
        for title, content in _SECTION_RE.findall(source_text)
    }


def _paragraphs(section_text: str) -> list[str]:
    """Return substantial prose-like paragraphs from one source section."""
    section_text = re.sub(r"\\subsection\{[^}]*\}", "\n\n", section_text)
    candidates = []
    for chunk in re.split(r"\n\s*\n", section_text):
        cleaned = _clean_source_excerpt(chunk)
        if len(cleaned) < 140:
            continue
        if cleaned.startswith(("\\begin{", "\\end{", "\\caption", "\\includegraphics", "\\author", "\\affiliation")):
            continue
        candidates.append(cleaned)
    return candidates


def _sentences(text: str) -> list[str]:
    cleaned = _clean_source_excerpt(text)
    if not cleaned:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]


def _pick_quote(section_text: str, fallback_text: str = "") -> str:
    """Pick one dense quote from a section or fallback text."""
    for paragraph in _paragraphs(section_text):
        if 160 <= len(paragraph) <= 900:
            return paragraph
    fallback = _clean_source_excerpt(fallback_text)
    return fallback[:900].strip()


def _takeaway_items(abstract: str, summary_text: str) -> list[str]:
    """Build concise takeaway bullets from abstract/summary text."""
    items: list[str] = []
    for sentence in _sentences(abstract) + _sentences(summary_text):
        if sentence not in items:
            items.append(sentence)
        if len(items) >= 4:
            break
    return items[:4]


def _fallback_report_body(paper: dict, source_text: str, figures: list[dict], abstract: str) -> str:
    """Deterministic ArxivSella-style fallback when the LLM returns meta-commentary."""
    sections = _extract_sections(source_text)
    intro = sections.get("introduction", "")
    observations = sections.get("observations and data reduction", "")
    photometry = sections.get("photometric properties", "")
    spectroscopy = sections.get("spectroscopic properties", "")
    discussion = sections.get("discussion", "")
    conclusions = sections.get("summary and conclusions", "")

    abstract_sentences = _sentences(abstract)
    motivation_text = " ".join(abstract_sentences[:2]) or _clean_source_excerpt(intro)[:320]
    approach_text = " ".join(abstract_sentences[2:4]) or _clean_source_excerpt(observations)[:320]
    results_text = " ".join(abstract_sentences[4:]) or _clean_source_excerpt(conclusions or discussion)[:360]

    figure_lines = []
    for index, figure in enumerate(figures[:2]):
        figure_lines.append(f"<!-- FIGURE:{index} -->")
        figure_lines.append(f"<q>{escape(_clean_source_excerpt(figure['caption']))}</q>")
    if not figure_lines:
        figure_lines.append("<p>No extracted figures were available for this paper, so the reading guide relies on the text alone.</p>")

    takeaway_items = _takeaway_items(abstract, results_text or motivation_text)
    if not takeaway_items:
        takeaway_items = ["This paper is best read by following the methodology, then the results, then the authors' interpretation and conclusions."]

    parts = [
        "<h2>Motivation</h2>",
        f"<p>{escape(motivation_text)}</p>",
        f"<q>{escape(_pick_quote(intro, abstract))}</q>",
        "<h2>Approach</h2>",
        f"<p>{escape(approach_text)}</p>",
        f"<q>{escape(_pick_quote(observations or photometry, abstract))}</q>",
        "<h2>Key Results</h2>",
        f"<p>{escape(results_text)}</p>",
        f"<q>{escape(_pick_quote(discussion or conclusions or spectroscopy, abstract))}</q>",
        "<h2>Figure Reading Guide</h2>",
        "<p>Follow the figures in document order to trace the paper's key measurements, model comparisons, and observational diagnostics.</p>",
        *figure_lines,
        "<h2>Take-aways</h2>",
        "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in takeaway_items) + "</ul>",
    ]
    return "\n".join(parts)


_Q_TAG_RE = re.compile(r"<q\b", re.IGNORECASE)


def _is_structurally_valid(body: str) -> bool:
    """Body must have at least one <h2> heading and at least one <q> verbatim quote."""
    converted = body if _HTML_TAG_RE.search(body) else _markdownish_to_html(body)
    return bool(_H2_RE_BODY.search(converted)) and bool(_Q_TAG_RE.search(converted))


def _build_context_prompt(paper: dict, abstract: str) -> tuple[str, str]:
    system = """You are a research assistant helping a scientist read a new paper.
Write a short 'Wider Context' section (HTML fragment, no wrapper tags) that places this paper's results in the broader scientific landscape.

Begin immediately with <h2>Wider Context</h2>.

You may draw on your general knowledge — this section is explicitly NOT limited to what is in the paper.
Cover 2–4 of the following as relevant: how this result fits into or challenges the current consensus; related open questions it bears on; connections to other recent work or subfields; implications or follow-up directions a reader might pursue.

Use <h2>Wider Context</h2> as the heading, <h3> for any sub-headings, and <p> for prose.
Be concise (3–5 paragraphs). No bullet lists. No code fences. No markdown."""
    user = (
        f"Title: {_paper_title(paper)}\n"
        f"Abstract: {abstract}"
    )
    return system, user


def _generate_context_section(summarizer, paper: dict, abstract: str) -> str:
    """Generate a free-form wider-context commentary section using the LLM."""
    system, user = _build_context_prompt(paper, abstract)
    try:
        text = summarizer.complete(system, user, max_length=600, detailed=True).strip()
        # Strip any accidental fences/wrappers
        text = _FENCE_RE.sub("", text)
        text = _WRAPPER_TAG_RE.sub("", text)
        # Must start with the expected heading; if not, prepend it
        if not re.search(r"<h2\b[^>]*>\s*Wider Context", text, re.IGNORECASE):
            text = "<h2>Wider Context</h2>\n" + text
        return text
    except Exception:
        return ""


def _generate_body_with_retry(summarizer, system: str, user: str) -> str:
    last = ""
    for attempt in range(MAX_REPORT_ATTEMPTS):
        if attempt == 0:
            prompt_user = user
        else:
            prompt_user = (
                user
                + "\n\nYour previous response was not a valid report. Begin immediately with <h2>Section Name</h2> "
                "and write only HTML report sections. "
                "Do not ask questions, do not summarise what you are about to do, do not produce analysis, "
                "outlines, recommendations, or manuscript-writing advice."
            )
        last = summarizer.complete(system, prompt_user, max_length=DEFAULT_REPORT_WORDS, detailed=True).strip()
        if not _looks_like_meta_commentary(last) and _is_structurally_valid(last):
            return last
    return ""


def _figure_block(figure: dict) -> str:
    path = figure["path"]
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(data).decode("ascii")
    caption = escape(figure["caption"])
    return (
        f'<figure style="margin:18px 0;text-align:center;">'
        f'<img src="data:{mime};base64,{b64}" '
        f'style="max-width:100%;max-height:460px;border:1px solid #ddd;border-radius:4px;" />'
        f'<figcaption style="font-size:13px;color:#555;font-style:italic;margin-top:6px;">'
        f"{caption}</figcaption></figure>"
    )


def render_report(paper: dict, body_html: str, figures: list[dict], abstract: str) -> str:
    """Embed figure markers and wrap the report in a MathJax-enabled document."""
    body = _FENCE_RE.sub("", body_html.strip())
    body = _WRAPPER_TAG_RE.sub("", body)
    body = _H1_RE.sub("", body)
    if not _HTML_TAG_RE.search(body):
        body = _markdownish_to_html(body)
    if _looks_like_meta_commentary(body):
        raise ReportUnavailable("the configured LLM returned meta-commentary instead of the report")

    def _embed(match: re.Match) -> str:
        index = int(match.group(1))
        return _figure_block(figures[index]) if 0 <= index < len(figures) else ""

    body = _FIGURE_MARKER_RE.sub(_embed, body)
    body = _auto_insert_figures(body, figures)
    intro = _render_intro(paper, abstract)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{escape(_paper_title(paper))}</title>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
</head>
<body style="font-family:Helvetica,Arial,sans-serif;max-width:760px;margin:0 auto;padding:24px;color:#1a1a2e;line-height:1.6;">
<p style="font-size:13px;color:#666;"><a href="{escape(_paper_abs_url(paper), quote=True)}">{escape(_paper_abs_url(paper))}</a></p>
{intro}
{body}
</body></html>"""


def generate_report(
    paper: dict,
    *,
    summarizer,
    arxiv_client,
    pdf_extractor,
    reports_dir: Path,
    sources_dir: Path,
    force: bool = False,
) -> Path:
    """Generate or reuse a cached detailed report for one paper."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    arxiv_id = str(paper.get("arxiv_id", "")).split("v")[0]
    if not arxiv_id:
        raise ReportUnavailable("paper has no arXiv id")
    out_path = reports_dir / f"{arxiv_id.replace('/', '_')}.html"
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    paper_dir = download_source(arxiv_id, sources_dir)
    source_text = extract_text(paper_dir) if paper_dir else ""
    source_abstract = extract_abstract(paper_dir) if paper_dir else ""
    figures = [figure for figure in extract_figures(paper_dir) if figure["path"]] if paper_dir else []
    figures = figures[:MAX_FIGURES]

    if not source_text:
        source_text = get_paper_text(
            arxiv_id,
            arxiv_client,
            pdf_extractor,
            title=paper.get("title"),
            pdf_url=paper.get("pdf_url"),
            cache_dir=Path("data/papers"),
        )
    source_text = _trim_source(source_text)
    if len(source_text) < 1000:
        raise ReportUnavailable(f"no readable source text for {arxiv_id}")

    abstract = _best_abstract(paper, source_abstract)
    fig_list = "\n".join(
        f"[{index}] {figure['caption'][:200]}"
        for index, figure in enumerate(figures)
    )
    system, user = _build_prompt(paper, source_text, fig_list, abstract)
    # REPORT_LLM_MODEL / REPORT_LLM_PROVIDER let you route reports to a
    # more capable model without changing the default summarizer.
    report_model = os.getenv("REPORT_LLM_MODEL", "").strip()
    report_provider = os.getenv("REPORT_LLM_PROVIDER", "").strip()
    if report_model or report_provider:
        from app.summarizer import PaperSummarizer
        report_summarizer = PaperSummarizer(
            provider=report_provider or None,
            model=report_model or None,
        )
    else:
        report_summarizer = summarizer
    body = _generate_body_with_retry(report_summarizer, system, user)
    if not body:
        body = _fallback_report_body(paper, source_text, figures, abstract)
    context_section = _generate_context_section(report_summarizer, paper, abstract)
    if context_section:
        body = body + "\n" + context_section
    html = render_report(paper, body, figures, abstract)
    tmp = out_path.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.rename(out_path)
    return out_path
