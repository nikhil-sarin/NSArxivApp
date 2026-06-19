"""Download ArXiv sources and extract text/figures for detailed reports."""

from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import requests

EPRINT_URL = "https://arxiv.org/e-print"
USER_AGENT = "NSArxivApp/1.0 (+https://github.com/nikhil-sarin/NSArxivApp)"
_WEB_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_GS_CONVERTIBLE_EXTENSIONS = {".pdf", ".eps", ".ps"}
_ALL_FIG_EXTENSIONS = _WEB_EXTENSIONS | _GS_CONVERTIBLE_EXTENSIONS
_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_FIGURE_ENV_RE = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.DOTALL)


def _extract_braced_content(text: str, start_brace: int) -> str:
    depth = 0
    chars: list[str] = []
    for index in range(start_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
            if depth == 1:
                continue
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars)
        if depth >= 1:
            chars.append(char)
    return ""


def _extract_command_argument(text: str, command: str) -> str:
    marker = f"\\{command}"
    index = text.find(marker)
    if index == -1:
        return ""
    brace = text.find("{", index + len(marker))
    if brace == -1:
        return ""
    return _extract_braced_content(text, brace)


def download_source(
    arxiv_id: str,
    output_dir: Path,
    *,
    session: requests.Session | None = None,
) -> Path | None:
    """Download and extract one paper's source tree."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = output_dir / arxiv_id.replace("/", "_")
    if paper_dir.exists():
        return paper_dir

    own_session = session is None
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
    try:
        response = session.get(
            f"{EPRINT_URL}/{arxiv_id}",
            timeout=60,
            allow_redirects=True,
        )
        if response.status_code != 200:
            return None

        tmp_dir = Path(tempfile.mkdtemp(dir=output_dir))
        try:
            try:
                with tarfile.open(fileobj=BytesIO(response.content), mode="r:*") as tar:
                    safe = [
                        member for member in tar.getmembers()
                        if not member.name.startswith("/") and ".." not in member.name
                    ]
                    tar.extractall(tmp_dir, members=safe)
                tmp_dir.rename(paper_dir)
                return paper_dir
            except tarfile.TarError:
                pass

            try:
                text = response.content.decode("utf-8", errors="strict")
                if "\\document" in text or "\\begin" in text:
                    (tmp_dir / "main.tex").write_bytes(response.content)
                    tmp_dir.rename(paper_dir)
                    return paper_dir
            except UnicodeDecodeError:
                pass

            (tmp_dir / "source.bin").write_bytes(response.content)
            tmp_dir.rename(paper_dir)
            return paper_dir
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    finally:
        if own_session:
            session.close()


def _sorted_tex_files(paper_dir: Path) -> list[Path]:
    main: list[Path] = []
    aux: list[Path] = []
    for path in paper_dir.rglob("*.tex"):
        try:
            head = path.read_bytes()[:2048].decode("utf-8", errors="replace")
        except OSError:
            continue
        if "\\documentclass" in head:
            main.append(path)
        else:
            aux.append(path)
    return sorted(main) + sorted(aux)


def extract_text(paper_dir: Path, *, max_chars: int = 120_000) -> str:
    """Concatenate TeX sources into one bounded text blob."""
    chunks: list[str] = []
    total = 0
    for tex_file in _sorted_tex_files(paper_dir):
        try:
            text = tex_file.read_text(errors="replace")
        except OSError:
            continue
        chunks.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(chunks)[:max_chars]


def extract_abstract(paper_dir: Path) -> str:
    """Extract the LaTeX abstract from the main source if present."""
    for tex_file in _sorted_tex_files(paper_dir):
        try:
            text = tex_file.read_text(errors="replace")
        except OSError:
            continue
        abstract = _extract_command_argument(text, "abstract")
        if abstract:
            return " ".join(abstract.split())
    return ""


def _convert_to_png(path: Path) -> Path | None:
    png_path = path.with_suffix(".png")
    if png_path.exists():
        return png_path
    if shutil.which("gs") is None:
        return None
    try:
        args = [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            "-sDEVICE=png16m",
            "-r100",
        ]
        if path.suffix.lower() in {".eps", ".ps"}:
            args.append("-dEPSCrop")
        subprocess.run(
            args + [f"-sOutputFile={png_path}", str(path)],
            capture_output=True,
            timeout=15,
            check=False,
        )
        return png_path if png_path.exists() else None
    except Exception:
        return None


def _resolve_figure(tex_dir: Path, ref: str, root_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for base_dir in dict.fromkeys([tex_dir, root_dir]):
        base = base_dir / ref
        candidates.append(base)
        if base.suffix.lower() not in _ALL_FIG_EXTENSIONS:
            for ext in (".png", ".jpg", ".jpeg", ".pdf", ".eps", ".ps"):
                candidates.append(base.with_suffix(ext))
    return candidates


def extract_figures(paper_dir: Path) -> list[dict]:
    """Extract figures in document order with captions and image paths."""
    figures: list[dict] = []
    for tex_file in _sorted_tex_files(paper_dir):
        try:
            text = tex_file.read_text(errors="replace")
        except OSError:
            continue
        for env in _FIGURE_ENV_RE.finditer(text):
            body = env.group(1)
            graphics = _INCLUDEGRAPHICS_RE.search(body)
            if not graphics:
                continue
            caption = " ".join(_extract_command_argument(body, "caption").split())
            path = None
            for candidate in _resolve_figure(tex_file.parent, graphics.group(1).strip(), paper_dir):
                if not candidate.is_file():
                    continue
                suffix = candidate.suffix.lower()
                if suffix in _WEB_EXTENSIONS:
                    path = candidate
                    break
                if suffix in _GS_CONVERTIBLE_EXTENSIONS:
                    converted = _convert_to_png(candidate)
                    if converted is not None:
                        path = converted
                        break
            figures.append({"caption": caption, "path": path})
    return figures
