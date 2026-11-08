"""Store Protocols (design §3, §4): documents, chunks, vectors and BM25 postings behind one
interface, so a backend that can (PostgreSQL) writes a whole document in one transaction; and a
separate vector cache, which a backend may or may not offer."""

import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol

from lite_rag.types import Chunk, Document


@dataclass(frozen=True)
class KbRef:
    """A knowledge base inside a namespace. `key` is the one place the physical key is formed."""

    namespace: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.name}"


@dataclass(frozen=True)
class KbInfo:
    """What a knowledge base was built with. Opening it with anything else is refused: one model,
    one set of prefixes and one chunker per index (KB rag-production-guide §4 rule 2)."""

    ref: KbRef
    embedding_provider: str
    embedding_model: str
    dim: int
    query_prefix: str
    document_prefix: str
    chunker: dict[str, Any]

    def differences(self, other: "KbInfo") -> list[str]:
        return [
            f"{f.name}: built with {getattr(self, f.name)!r}, configured {getattr(other, f.name)!r}"
            for f in dataclasses.fields(self)
            if f.name != "ref" and getattr(self, f.name) != getattr(other, f.name)
        ]


@dataclass(frozen=True)
class IndexedChunk:
    chunk: Chunk
    vector: list[float]
    terms: list[str]


@dataclass(frozen=True)
class KbStats:
    namespace: str
    kb: str
    documents: int
    chunks: int
    avg_length: float
    embedding: str
    dim: int


class Store(Protocol):
    def get_or_create_kb(self, info: KbInfo) -> KbInfo:
        """The kb as stored — created from `info` if it did not exist. The caller compares."""
        ...

    def document_hash(self, kb: KbRef, doc_id: str) -> str | None: ...

    def upsert_document(
        self, kb: KbRef, doc: Document, content_hash: str, chunks: list[IndexedChunk]
    ) -> None: ...

    def delete_document(self, kb: KbRef, doc_id: str) -> bool: ...

    def delete_kb(self, kb: KbRef) -> bool: ...

    def stats(self, kb: KbRef) -> KbStats | None: ...

    def close(self) -> None: ...


class VectorCache(Protocol):
    """Vectors by (model, input key). A vector depends only on the model and its exact input, so
    the cache is shared by every kb and namespace."""

    def get(self, model_id: str, dim: int, keys: list[str]) -> dict[str, list[float]]: ...

    def put(self, model_id: str, dim: int, vectors: dict[str, list[float]]) -> None: ...
