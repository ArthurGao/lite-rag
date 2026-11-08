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


def model_id(provider: str, model: str) -> str:
    """What indexes, rerank scores and answers are recorded against, e.g. `ollama/bge-m3:567m`."""
    return f"{provider}/{model}"


@dataclass(frozen=True)
class Hit:
    """One search result. `scores` / `ranks` keep each stage's view (`vector`, `bm25`, `fused`,
    later `rerank`) so the effect of every stage can be seen and compared (design §5)."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    start: int
    end: int
    heading_path: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
