"""File loaders (design §3): text, Markdown and — through extras — PDF and DOCX.

A loader returns the text plus where each page starts, so chunks can cite a page. Structure is
kept where the loader can see it: Markdown headings stay as they are, and the `docling` loader
turns a PDF or DOCX into Markdown so the markdown chunker gets its headings. The `pypdf` loader
extracts plain text only (fast, no models).
"""

import hashlib
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

from lite_rag.errors import ConfigError
from lite_rag.extras import require_extra

PdfLoader = Literal["pypdf", "docling"]
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx"}
# An ATX Markdown heading: group 1 the level marks, group 2 the title (also the chunkers' rule).
ATX = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_PAGE_BREAK = "<!-- lite-rag:page-break -->"  # asked of docling between pages
PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Loaded:
    title: str
    text: str
    pages: tuple[int, ...] = ()  # where each page starts in `text`; empty: no pages


def doc_id_for(source: str) -> str:
    """A stable document id for a source (a path, an upload's file name): the same source
    replaces the same document."""
    return "file:" + hashlib.sha256(source.encode()).hexdigest()[:16]


def file_doc_id(path: Path) -> str:
    return doc_id_for(str(path.resolve()))


def join_pages(pages: list[str]) -> tuple[str, tuple[int, ...]]:
    """The pages as one text, and the offset where each starts."""
    parts = [page.strip() for page in pages]
    starts, at = [], 0
    for part in parts:
        starts.append(at)
        at += len(part) + len(PAGE_SEPARATOR)
    return PAGE_SEPARATOR.join(parts), tuple(starts)


def load_file(path: Path, pdf: PdfLoader = "pypdf", name: str | None = None) -> Loaded:
    """The text of a supported file. The title is the PDF's own title, else the first heading
    (`# ` in Markdown, any level from docling), else the file name — `name`'s, when the file
    sits at a temporary path (an upload). `.docx` always goes through docling."""
    stem = Path(name).stem if name else path.stem
    suffix = path.suffix.lower()
    if suffix not in SUFFIXES:
        raise ConfigError(
            f"{path}: unsupported file type (supported: {', '.join(sorted(SUFFIXES))})"
        )
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8")
        title = _title(text, max_level=1) if suffix != ".txt" else None
        return Loaded(title or stem, text)
    if suffix == ".pdf" and pdf == "pypdf":
        title, pages = _pypdf_pages(path)
    else:
        pages = docling_markdown(path).split(_PAGE_BREAK)
        title = None
    text, starts = join_pages(pages)
    return Loaded(title or _title(text, max_level=6) or stem, text, starts)


def _title(markdown: str, max_level: int) -> str | None:
    """The first heading of at most `max_level` (docling writes every level as `##`)."""
    heading = next((h for h in ATX.finditer(markdown) if len(h.group(1)) <= max_level), None)
    return heading.group(2).strip() if heading else None


def _pypdf_pages(path: Path) -> tuple[str | None, list[str]]:
    pypdf = require_extra("pypdf", "pdf")
    reader = pypdf.PdfReader(path)
    title = reader.metadata.title if reader.metadata else None
    return (title or None), [page.extract_text() or "" for page in reader.pages]


@cache
def _docling_converter() -> Any:
    """One converter per process: building it loads the layout models (seconds)."""
    require_extra("docling", "docling")
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # No OCR: scanned documents are out of scope (design §1); this keeps it fast and small.
    options = PdfPipelineOptions(do_ocr=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def docling_markdown(path: Path) -> str:
    """The document as Markdown, pages separated by `_PAGE_BREAK`."""
    document = _docling_converter().convert(path).document
    return str(document.export_to_markdown(page_break_placeholder=_PAGE_BREAK))


def find_files(paths: list[Path]) -> list[Path]:
    """The supported files among `paths`, descending into directories, in a stable order."""
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            found.extend(f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUFFIXES)
        else:
            found.append(p)
    return found
