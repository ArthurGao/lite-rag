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
    pages: tuple[int, ...] = ()  # where each page starts in `text` (PDF); empty: no pages


@dataclass(frozen=True)
class Chunk:
    """One piece of a document. `chunk_id` is `"{doc_id}#{n}"`; `start` / `end` are character
    offsets into the document text, for citations. `page` (1-based) is the page the chunk starts
    on, None when the document has no pages; `parent` is the `n` of the parent chunk it belongs to
    (strategy `parent_child`), else None."""

    chunk_id: str
    doc_id: str
    n: int
    text: str
    start: int
    end: int
    tokens: int
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    parent: int | None = None


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
    page: int | None = None
    # parent_child: `text` / `start` / `end` are the parent's; this is the child span that matched.
    matched: tuple[int, int] | None = None
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    kb: str = field(kw_only=True)  # the knowledge base it came from (several searched at once)


# Graph retrieval (design §5, S11): what an LLM found in one chunk.


@dataclass(frozen=True)
class Entity:
    key: str  # entity_key(name): what merges mentions across chunks and documents
    name: str  # as the passage writes it (the first spelling seen in the chunk)
    type: str  # free text from the model, "" when it gave none


@dataclass(frozen=True)
class Relation:
    source: str  # entity keys
    target: str
    description: str


@dataclass(frozen=True)
class ChunkGraph:
    entities: tuple[Entity, ...] = ()
    relations: tuple[Relation, ...] = ()
