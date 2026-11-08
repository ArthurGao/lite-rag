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

from lite_rag.answer import Answer, check_citations, prompt
from lite_rag.chunking import build_chunker, chunk_context, embedding_text
from lite_rag.config import Mode, Settings
from lite_rag.embedding import Embedder, build_embedder
from lite_rag.errors import ConfigError, KbNotFound
from lite_rag.fusion import rrf, weighted
from lite_rag.llm import LLM, build_llm
from lite_rag.loaders import file_doc_id, load_file
from lite_rag.ranking import ranked
from lite_rag.rerank import Reranker, build_reranker
from lite_rag.stores.base import IndexedChunk, KbInfo, KbRef, KbStats, Store, StoredChunk
from lite_rag.stores.pg import PgStore
from lite_rag.text import build_counter, terms
from lite_rag.types import Document, Hit

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
        self,
        settings: Settings,
        store: Store,
        embedder: Embedder,
        *,
        namespace: str,
        reranker: Reranker | None = None,
        llm: LLM | None = None,
    ) -> None:
        """`reranker` and `llm` are built from `settings` on first use when not given, so a
        search-only engine never needs an LLM key or the rerank model."""
        self.namespace = check_namespace(namespace)
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._llm = llm
        self._counter = build_counter(settings.chunking, settings.embedding)
        self._chunker = build_chunker(settings.chunking, self._counter)
        self._ready: set[KbRef] = set()  # kbs checked against the configuration

    @classmethod
    def from_settings(cls, settings: Settings, namespace: str | None = None) -> "RagEngine":
        ns = check_namespace(namespace or settings.namespace)
        store = PgStore(
            settings.database_url, settings.store.schema_name, bm25_impl=settings.store.bm25_impl
        )
        embedder = build_embedder(settings.embedding, cache=store)
        # Built now when enabled, so a misconfigured reranker fails at startup (its model still
        # loads on first use). The LLM stays lazy: a search-only engine needs no key.
        reranker = build_reranker(settings.rerank) if settings.rerank.enabled else None
        return cls(settings, store, embedder, namespace=ns, reranker=reranker)

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

    def _open(self, kb: str, *, create: bool) -> KbRef:
        """Check that `kb` was built with this configuration. Writes create a missing kb and
        check once per engine (a kb deleted later fails their locked write with `KbNotFound`);
        reads check every time and raise `KbNotFound`, so a mistyped name — or a kb dropped by
        another process — is an error, never an empty result."""
        ref = self._ref(kb)
        if not create or ref not in self._ready:
            wanted = self.kb_info(kb)
            stored = self._store.get_or_create_kb(wanted) if create else self._store.get_kb(ref)
            if stored is None:
                raise KbNotFound(f"knowledge base {kb!r} does not exist in {self.namespace!r}")
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
        ref = self._open(kb, create=True)
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
        """False if the document (or its kb) does not exist: deleting is idempotent, and needs
        no identity check (nothing is embedded)."""
        return self._store.delete_document(self._ref(kb), doc_id)

    def delete_kb(self, kb: str) -> bool:
        ref = self._ref(kb)
        self._ready.discard(ref)
        return self._store.delete_kb(ref)

    # --- reads -------------------------------------------------------------------------------

    def search(
        self,
        kb: str,
        query: str,
        *,
        top_k: int | None = None,
        mode: Mode | None = None,
        rerank: bool | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """The best `top_k` chunks for `query` (design §5). `mode`: `vector`, `bm25` or `hybrid`
        (both, fused). `rerank` (default: config) re-scores the best `rerank.top_n` candidates
        with a cross-encoder. `where` keeps only documents whose metadata contains those
        key-values. Each hit carries every stage's score and rank; `score` is the final one."""
        cfg = self._settings.retrieval
        k, mode = top_k or cfg.top_k, mode or cfg.mode
        reranking = self._settings.rerank.enabled if rerank is None else rerank
        # `pool` candidates reach the end (all of them re-scored when reranking); each path
        # returns `n` — more in hybrid, where fusion needs candidates from both lists.
        pool = max(self._settings.rerank.top_n, k) if reranking else k
        n = max(pool, k * cfg.candidate_factor) if mode == "hybrid" else pool
        ref = self._open(kb, create=False)
        rankings: dict[str, list[tuple[str, float]]] = {}
        if mode in ("vector", "hybrid"):
            vector = self._embedder.embed_query(query)
            rankings["vector"] = self._store.vector_query(ref, vector, n, where)
        if mode in ("bm25", "hybrid"):
            bm = cfg.bm25
            rankings["bm25"] = self._store.bm25_query(ref, terms(query), n, bm.k1, bm.b, where)
        if mode == "hybrid":
            rankings["fused"] = self._fuse(rankings)
        ranking = rankings["fused" if mode == "hybrid" else mode]
        # Look up every ranked chunk: one deleted since it was ranked makes room for the next.
        stored = self._store.chunks(ref, [chunk for chunk, _ in ranking])
        candidates = [(c, s) for c, s in ranking if c in stored][:pool]
        if reranking and candidates:
            candidates = rankings["rerank"] = self._rerank(query, candidates, stored)
        ranks = {s: {c: i for i, (c, _) in enumerate(r, start=1)} for s, r in rankings.items()}
        scores = {s: dict(r) for s, r in rankings.items()}
        return [
            Hit(
                **vars(stored[chunk]),
                score=score,
                scores={s: m[chunk] for s, m in scores.items() if chunk in m},
                ranks={s: m[chunk] for s, m in ranks.items() if chunk in m},
            )
            for chunk, score in candidates[:k]
        ]

    def ask(
        self,
        kb: str,
        question: str,
        *,
        top_k: int | None = None,
        mode: Mode | None = None,
        rerank: bool | None = None,
        where: dict[str, Any] | None = None,
    ) -> Answer:
        """Search, then let the LLM answer from the hits only, citing them as [n] (design §6).
        Citations to numbers that were not given are removed and reported in `dropped`. With
        no hits the LLM is not called."""
        hits = self.search(kb, question, top_k=top_k, mode=mode, rerank=rerank, where=where)
        if not hits:
            return Answer(
                question=question,
                status="no_sources",
                text="",
                citations=[],
                sources=[],
                model_id=None,
            )
        completion = self._get_llm().complete(prompt(question, hits))
        text, citations, dropped = check_citations(completion.text, hits)
        return Answer(
            question=question,
            status="answered",
            text=text,
            citations=citations,
            sources=hits,
            model_id=completion.model_id,
            usage=completion.usage,
            dropped=dropped,
        )

    def _rerank(
        self, query: str, candidates: list[tuple[str, float]], stored: dict[str, StoredChunk]
    ) -> list[tuple[str, float]]:
        """The candidates re-scored by the cross-encoder, which reads each chunk with its
        title and heading path."""
        ids = [chunk for chunk, _ in candidates]
        texts = [
            f"{chunk_context(stored[c].title, stored[c].heading_path)}\n{stored[c].text}"
            for c in ids
        ]
        return ranked(dict(zip(ids, self._get_reranker().score(query, texts), strict=True)))

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = build_reranker(self._settings.rerank)
        return self._reranker

    def _get_llm(self) -> LLM:
        if self._llm is None:
            self._llm = build_llm(self._settings.llm)
        return self._llm

    def _fuse(self, rankings: dict[str, list[tuple[str, float]]]) -> list[tuple[str, float]]:
        cfg = self._settings.retrieval
        if cfg.fusion == "rrf":
            return rrf(rankings, cfg.rrf_k)
        return weighted(rankings, {"vector": cfg.vector_weight, "bm25": 1 - cfg.vector_weight})

    def stats(self, kb: str) -> KbStats | None:
        return self._store.stats(self._ref(kb))

    def close(self) -> None:
        self._store.close()
