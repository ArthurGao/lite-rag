"""Store Protocols (design §3, §4): documents, chunks, vectors and BM25 postings behind one
interface, so a backend that can (PostgreSQL) writes a whole document in one transaction; and a
separate vector cache, which a backend may or may not offer."""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from lite_rag.errors import InvalidRequest
from lite_rag.types import Chunk, Document

SCALARS = (str, int, float, bool)


def check_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """A metadata filter is key → scalar (str, int, float, bool), compared for equality — the
    same rule on every backend (design §5). The engine checks it before any work; stores only
    translate it."""
    for key, value in (where or {}).items():
        if not isinstance(value, SCALARS):
            raise InvalidRequest(
                f"metadata filter {key!r}: values must be scalars (str, int, float, bool)"
            )
    return where or None


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
    context: str | None = None  # contextual header: indexed with the chunk, never returned


@dataclass(frozen=True)
class StoredParent:
    """The larger chunk a `parent_child` hit returns instead of the child that matched."""

    n: int  # its number in the document (key with doc_id)
    text: str
    start: int
    end: int
    heading_path: tuple[str, ...]
    page: int | None


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start: int
    end: int
    heading_path: tuple[str, ...]
    page: int | None = None
    parent: StoredParent | None = None


@dataclass(frozen=True)
class StoredDocument:
    """A document as it was upserted. `text` is None for documents stored before lite-rag kept
    the original text (migration 0005)."""

    doc_id: str
    title: str
    text: str | None
    metadata: dict[str, Any]
    source: str | None
    pages: tuple[int, ...]


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

    def create_kb(self, info: KbInfo) -> bool:
        """Create the kb only if it does not exist, atomically; False: it already did."""
        ...

    def get_kb(self, kb: KbRef) -> KbInfo | None:
        """The kb as stored, or None; never creates one."""
        ...

    def document_hash(self, kb: KbRef, doc_id: str) -> str | None: ...

    def upsert_document(
        self,
        kb: KbRef,
        doc: Document,
        content_hash: str,
        chunks: list[IndexedChunk],
        parents: list[Chunk] | None = None,
    ) -> None:
        """Replace the document with `chunks` (embedded, indexed) and `parents` (stored only),
        in one transaction."""
        ...

    def delete_document(self, kb: KbRef, doc_id: str) -> bool: ...

    def delete_kb(self, kb: KbRef) -> bool: ...

    def list_kbs(self, namespace: str) -> list[str]:
        """The names of the namespace's kbs, sorted."""
        ...

    def documents(self, kb: KbRef, page: int = 100) -> Iterator[StoredDocument]:
        """Every document of `kb`, in doc_id order, read a page at a time."""
        ...

    def count_missing_text(self, kb: KbRef) -> int:
        """Documents stored without their text (before migration 0005)."""
        ...

    def stats(self, kb: KbRef) -> KbStats | None: ...

    # --- retrieval (design §5) ---------------------------------------------------------------

    def vector_query(
        self, kb: KbRef, vector: list[float], n: int, where: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """The `n` nearest chunks as (chunk_id, cosine similarity), best first."""
        ...

    def bm25_query(
        self,
        kb: KbRef,
        terms: list[str],
        n: int,
        k1: float,
        b: float,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """The `n` best chunks by BM25 (`keyword.py`) as (chunk_id, score), best first. N, n(t)
        and avgdl are the whole kb's; `where` only narrows which chunks may be returned."""
        ...

    def chunks(self, kb: KbRef, chunk_ids: list[str]) -> dict[str, StoredChunk]: ...

    def close(self) -> None: ...


class VectorCache(Protocol):
    """Vectors by (model, input key). A vector depends only on the model and its exact input, so
    the cache is shared by every kb and namespace."""

    def get(self, model_id: str, dim: int, keys: list[str]) -> dict[str, list[float]]: ...

    def put(self, model_id: str, dim: int, vectors: dict[str, list[float]]) -> None: ...


class CachedStore(Store, VectorCache, Protocol):
    """A store that is also the vector cache — what every backend implements."""
