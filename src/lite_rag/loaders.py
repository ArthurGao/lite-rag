"""File loaders (design §3): plain text and Markdown. PDF text extraction is a later extra."""

import hashlib
import re
from pathlib import Path

from lite_rag.errors import ConfigError

SUFFIXES = {".txt", ".md", ".markdown"}
_H1 = re.compile(r"^#[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def file_doc_id(path: Path) -> str:
    """A stable id for a file: re-ingesting the same path replaces the same document."""
    return "file:" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def load_file(path: Path) -> tuple[str, str]:
    """(title, text) of a supported file. The title is the first `# heading`, else the file name."""
    if path.suffix.lower() not in SUFFIXES:
        raise ConfigError(
            f"{path}: unsupported file type (supported: {', '.join(sorted(SUFFIXES))})"
        )
    text = path.read_text(encoding="utf-8")
    heading = _H1.search(text) if path.suffix.lower() != ".txt" else None
    return (heading.group(1).strip() if heading else path.stem), text


def find_files(paths: list[Path]) -> list[Path]:
    """The supported files among `paths`, descending into directories, in a stable order."""
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            found.extend(f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUFFIXES)
        else:
            found.append(p)
    return found
