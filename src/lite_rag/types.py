"""Value types shared by every component (design §3)."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """One piece of a document. `chunk_id` is `"{doc_id}#{n}"`; `start` / `end` are character
    offsets into the document text, for citations."""

    chunk_id: str
    doc_id: str
    n: int
    text: str
    start: int
    end: int
    tokens: int
    heading_path: tuple[str, ...] = ()


def chunk_id(doc_id: str, n: int) -> str:
    return f"{doc_id}#{n}"
