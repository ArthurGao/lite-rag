"""`RagEngine` — the one entry point for the library, the CLI and the HTTP service (design §3).

An engine is bound to ONE namespace (e.g. `tenderted/tenant_42`) at construction; every knowledge
base it touches lives inside that namespace, so one project or tenant can never read or write
another's index — isolation is enforced here, not left to callers (design §4).

It also owns the index-identity rule: a kb records the embedding model, its prefixes and the
chunker it was built with, and opening it with anything else raises (design §3.2).
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lite_rag.chunking import build_chunker, embedding_text
from lite_rag.config import Settings
from lite_rag.embedding import Embedder, build_embedder
from lite_rag.errors import ConfigError
from lite_rag.loaders import file_doc_id, load_file
from lite_rag.stores.base import IndexedChunk, KbInfo, KbRef, KbStats, Store
from lite_rag.stores.pg import PgStore
from lite_rag.text import build_counter, terms
from lite_rag.types import Document

NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}(/[a-z0-9][a-z0-9_.-]{0,62}){0,3}$")
KB_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")


@dataclass(frozen=True)
class Upserted:
    doc_id: str
    chunks: int
    skipped: bool  # unchanged: nothing was re-chunked or re-embedded


def content_hash(title: str, text: str, metadata: dict[str, Any]) -> str:
    payload = json.dumps([title, text, metadata], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def check_namespace(namespace: str | None) -> str:
    if not namespace:
        raise ConfigError("a namespace is required (e.g. 'myproject' or 'myproject/tenant_42')")
    if not NAMESPACE.match(namespace):
        raise ConfigError(
            f"invalid namespace {namespace!r}: lowercase letters, digits, '_', '-', '.', "
            "up to 4 levels separated by '/'"
        )
    return namespace


class RagEngine:
    def __init__(
        self, settings: Settings, store: Store, embedder: Embedder, *, namespace: str
    ) -> None:
        self.namespace = check_namespace(namespace)
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._counter = build_counter(settings.chunking, settings.embedding)
        self._chunker = build_chunker(settings.chunking, self._counter)
        self._ready: set[KbRef] = set()  # kbs checked against the configuration

    @classmethod
    def from_settings(cls, settings: Settings, namespace: str | None = None) -> "RagEngine":
        ns = check_namespace(namespace or settings.namespace)
        store = PgStore(settings.database_url, settings.store.schema_name)
        embedder = build_embedder(settings.embedding, cache=store)
        return cls(settings, store, embedder, namespace=ns)

    def _ref(self, kb: str) -> KbRef:
        if not KB_NAME.match(kb):
            raise ConfigError(
                f"invalid knowledge base name {kb!r}: lowercase letters, digits, '_', '-', '.'"
            )
        return KbRef(self.namespace, kb)

    def kb_info(self, kb: str) -> KbInfo:
        """What `kb` must have been built with to be opened by this engine."""
        return KbInfo(
            self._ref(kb),
            self._embedder.provider,
            self._embedder.model,
            self._embedder.dim,
            self._embedder.prefixes.query,
            self._embedder.prefixes.document,
            {**self._settings.chunking.identity(), "counter": self._counter.identity},
        )

    def _open(self, kb: str) -> KbRef:
        """Check (once) that `kb` was built with this configuration."""
        ref = self._ref(kb)
        if ref not in self._ready:
            wanted = self.kb_info(kb)
            stored = self._store.get_or_create_kb(wanted)
            if diffs := stored.differences(wanted):
                raise ConfigError(
                    f"knowledge base {kb!r} in namespace {self.namespace!r} was built differently"
                    f" ({'; '.join(diffs)}). Use the settings it was built with, or a new kb name"
                    " (switching model = a new kb + re-embedding)."
                )
            self._ready.add(ref)
        return ref

    # --- writes ------------------------------------------------------------------------------

    def upsert_text(
        self,
        kb: str,
        doc_id: str,
        title: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Upserted:
        """Store a document, replacing any earlier version. Unchanged content is a no-op."""
        ref = self._open(kb)
        meta = metadata or {}
        digest = content_hash(title, text, meta)
        if self._store.document_hash(ref, doc_id) == digest:
            return Upserted(doc_id, 0, skipped=True)
        doc = Document(doc_id, title, text, source, meta)
        chunks = self._chunker.split(doc)
        inputs = [embedding_text(c) for c in chunks]
        vectors = self._embedder.embed_documents(inputs) if inputs else []
        indexed = [
            IndexedChunk(c, v, terms(t)) for c, v, t in zip(chunks, vectors, inputs, strict=True)
        ]
        self._store.upsert_document(ref, doc, digest, indexed)
        return Upserted(doc_id, len(indexed), skipped=False)

    def ingest(self, kb: str, path: Path) -> Upserted:
        title, text = load_file(path)
        return self.upsert_text(kb, file_doc_id(path), title, text, source=str(path))

    def delete(self, kb: str, doc_id: str) -> bool:
        return self._store.delete_document(self._open(kb), doc_id)

    def delete_kb(self, kb: str) -> bool:
        ref = self._ref(kb)
        self._ready.discard(ref)
        return self._store.delete_kb(ref)

    # --- reads -------------------------------------------------------------------------------

    def stats(self, kb: str) -> KbStats | None:
        return self._store.stats(self._ref(kb))

    def close(self) -> None:
        self._store.close()
