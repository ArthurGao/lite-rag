"""`RagEngine` — the one entry point for the library, the CLI and the HTTP service (design §3).

An engine is bound to ONE namespace (e.g. `tenderted/tenant_42`) at construction; every knowledge
base it touches lives inside that namespace, so one project or tenant can never read or write
another's index — isolation is enforced here, not left to callers (design §4).

It also owns the index-identity rule: a kb records the embedding model, its prefixes and the
chunker it was built with, and opening it with anything else raises (design §3.2).
"""

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lite_rag.answer import Answer, check_citations, prompt
from lite_rag.chunking import build_chunker, chunk_context, embedding_text
from lite_rag.config import Mode, Settings
from lite_rag.contextual import Contextualizer
from lite_rag.embedding import Embedder, build_embedder
from lite_rag.errors import ConfigError, KbMismatch, KbNotFound, StoreError
from lite_rag.fusion import rrf, weighted
from lite_rag.llm import LLM, build_llm
from lite_rag.loaders import doc_id_for, file_doc_id, load_file
from lite_rag.ranking import ranked
from lite_rag.rerank import Reranker, build_reranker
from lite_rag.stores.base import IndexedChunk, KbInfo, KbRef, KbStats, Store, StoredChunk
from lite_rag.stores.factory import open_store
from lite_rag.text import build_counter, terms
from lite_rag.types import Document, Hit

NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}(/[a-z0-9][a-z0-9_.-]{0,62}){0,3}$")
KB_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")


@dataclass
class _Models:
    """Models built on first use, shared by an engine and its `for_namespace` copies."""

    reranker: Reranker | None
    llm: LLM | None
    context_llm: LLM | None


@dataclass(frozen=True)
class Upserted:
    doc_id: str
    chunks: int
    skipped: bool  # unchanged: nothing was re-chunked or re-embedded


def content_hash(
    title: str, text: str, metadata: dict[str, Any], pages: tuple[int, ...] = ()
) -> str:
    # Pages join only when present, so documents without pages keep the hashes they had.
    parts: list[Any] = [title, text, metadata, *([list(pages)] if pages else [])]
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False)
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
        context_llm: LLM | None = None,
    ) -> None:
        """`reranker` and `llm` are built from `settings` on first use when not given, so a
        search-only engine never needs an LLM key or the rerank model."""
        self.namespace = check_namespace(namespace)
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._models = _Models(reranker, llm, context_llm)
        self._counter = build_counter(settings.chunking, settings.embedding)
        # semantic chunking embeds sentences with the kb's own embedder (and its cache)
        self._chunker = build_chunker(settings.chunking, self._counter, embedder.embed_documents)
        self._ready: set[KbRef] = set()  # kbs checked against the configuration

    @classmethod
    def from_settings(cls, settings: Settings, namespace: str | None = None) -> "RagEngine":
        ns = check_namespace(namespace or settings.namespace)
        store = open_store(
            settings.store,
            settings.database_url,
            # one connection per request the server runs at once: none waits on the pool
            max_connections=max(4, settings.server.max_workers),
        )
        embedder = build_embedder(settings.embedding, cache=store)
        # Built now when enabled, so a misconfigured reranker fails at startup (its model still
        # loads on first use). The LLM stays lazy: a search-only engine needs no key.
        reranker = build_reranker(settings.rerank) if settings.rerank.enabled else None
        return cls(settings, store, embedder, namespace=ns, reranker=reranker)

    def for_namespace(self, namespace: str) -> "RagEngine":
        """This engine for another namespace: the same store, embedder, chunker and models
        (built once, also when built later), its own record of the kbs it has checked. A shallow
        copy: any per-namespace state added to the engine must be reset here too."""
        other = copy.copy(self)
        other.namespace = check_namespace(namespace)
        other._ready = set()
        return other

    def create_kb(self, kb: str) -> KbInfo:
        """Create `kb` with this engine's settings (or check an existing one against them)."""
        self._open(kb, create=True)
        info = self._store.get_kb(self._ref(kb))
        assert info is not None  # just created or checked
        return info

    def warm_up(self) -> None:
        """Load lazily built models now (the reranker, when enabled): the first request should
        not pay for it."""
        if self._settings.rerank.enabled:
            self._get_reranker().score("warm up", ["warm up"])

    def health(self, deep: bool = False) -> None:
        """Raises when the store (and, `deep`, the embedding service) cannot be reached. The
        deep probe is a cached query embedding: free after the first call."""
        self._store.list_kbs(self.namespace)
        if deep:
            self._embedder.embed_query("lite-rag health check")

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
                raise KbNotFound.of(kb, self.namespace)
            if diffs := stored.differences(wanted):
                raise KbMismatch(
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
        pages: tuple[int, ...] = (),
    ) -> Upserted:
        """Store a document, replacing any earlier version. Unchanged content is a no-op.
        `pages`: where each page starts in `text`, so chunks can cite their page."""
        ref = self._open(kb, create=True)
        meta = metadata or {}
        digest = content_hash(title, text, meta, pages)
        if self._store.document_hash(ref, doc_id) == digest:
            return Upserted(doc_id, 0, skipped=True)
        doc = Document(doc_id, title, text, source, meta, pages)
        parents, chunks = self._chunker.split_parents(doc)
        contexts: list[str | None] = (
            list(self._get_contextualizer().contexts(doc, chunks))
            if self._settings.chunking.contextual.enabled
            else [None] * len(chunks)
        )
        inputs = [embedding_text(c, ctx) for c, ctx in zip(chunks, contexts, strict=True)]
        vectors = self._embedder.embed_documents(inputs) if inputs else []
        indexed = [
            IndexedChunk(c, v, terms(t), ctx)
            for c, v, t, ctx in zip(chunks, vectors, inputs, contexts, strict=True)
        ]
        self._store.upsert_document(ref, doc, digest, indexed, parents)
        return Upserted(doc_id, len(indexed), skipped=False)

    def ingest(
        self, kb: str, path: Path, *, doc_id: str | None = None, source: str | None = None
    ) -> Upserted:
        """Load and upsert a file. `source` (default: the path) names where it came from — an
        upload's file name when `path` is a temporary copy — and gives the default `doc_id` and
        the fallback title."""
        loaded = load_file(path, self._settings.loader.pdf, name=source)
        doc = doc_id or (doc_id_for(source) if source else file_doc_id(path))
        return self.upsert_text(
            kb, doc, loaded.title, loaded.text, source=source or str(path), pages=loaded.pages
        )

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
        k = cfg.top_k if top_k is None else top_k
        if k < 1:
            raise ConfigError(f"top_k must be at least 1, got {k}")
        mode = mode or cfg.mode
        reranking = self._settings.rerank.enabled if rerank is None else rerank
        # `pool` candidates reach the end (all of them re-scored when reranking); each path
        # returns `n` — more in hybrid, where fusion needs candidates from both lists, and more
        # when several chunks share a parent (parent_child), which count once.
        pool = max(self._settings.rerank.top_n, k) if reranking else k
        wanted = pool * self._chunker.children_per_parent  # chunks that make `pool` parents
        n = max(wanted, k * cfg.candidate_factor) if mode == "hybrid" else wanted
        ref = self._open(kb, create=False)
        vector = self._embedder.embed_query(query) if mode in ("vector", "hybrid") else None
        while True:
            rankings = self._rank(ref, query, vector, mode, n, where)
            ranking = rankings["fused" if mode == "hybrid" else mode]
            # Look up every ranked chunk: one deleted since it was ranked makes room for the
            # next. Short of `pool` while a path came back full (more exists): look deeper —
            # `children_per_parent` is an estimate, and deletes can race a search.
            stored = self._store.chunks(ref, [chunk for chunk, _ in ranking])
            candidates = _best_per_parent([(c, s) for c, s in ranking if c in stored], stored)
            full = any(len(r) == n for name, r in rankings.items() if name != "fused")
            if len(candidates) >= pool or not full:
                break
            n *= 2
        candidates = candidates[:pool]
        if reranking and candidates:
            candidates = rankings["rerank"] = self._rerank(query, candidates, stored)
        ranks = {s: {c: i for i, (c, _) in enumerate(r, start=1)} for s, r in rankings.items()}
        scores = {s: dict(r) for s, r in rankings.items()}
        return [self._hit(stored[c], score, scores, ranks) for c, score in candidates[:k]]

    def _rank(
        self,
        ref: KbRef,
        query: str,
        vector: list[float] | None,
        mode: Mode,
        n: int,
        where: dict[str, Any] | None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Each path's top `n` (and their fusion, in hybrid mode)."""
        rankings: dict[str, list[tuple[str, float]]] = {}
        if vector is not None:
            rankings["vector"] = self._store.vector_query(ref, vector, n, where)
        if mode in ("bm25", "hybrid"):
            bm = self._settings.retrieval.bm25
            rankings["bm25"] = self._store.bm25_query(ref, terms(query), n, bm.k1, bm.b, where)
        if mode == "hybrid":
            rankings["fused"] = self._fuse(rankings)
        return rankings

    @staticmethod
    def _hit(
        c: StoredChunk,
        score: float,
        scores: dict[str, dict[str, float]],
        ranks: dict[str, dict[str, int]],
    ) -> Hit:
        """A hit for chunk `c`; a parent_child child returns its parent's text and position."""
        view = c.parent or c
        return Hit(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            title=c.title,
            text=view.text,
            score=score,
            start=view.start,
            end=view.end,
            heading_path=view.heading_path,
            page=view.page,
            matched=(c.start, c.end) if c.parent is not None else None,
            scores={s: m[c.chunk_id] for s, m in scores.items() if c.chunk_id in m},
            ranks={s: m[c.chunk_id] for s, m in ranks.items() if c.chunk_id in m},
        )

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
        if self._models.reranker is None:
            self._models.reranker = build_reranker(self._settings.rerank)
        return self._models.reranker

    def _get_contextualizer(self) -> Contextualizer:
        cfg = self._settings.chunking.contextual
        if self._models.context_llm is None:
            self._models.context_llm = build_llm(cfg.llm)
        return Contextualizer(self._models.context_llm, self._counter, cfg.max_doc_tokens)

    def _get_llm(self) -> LLM:
        if self._models.llm is None:
            self._models.llm = build_llm(self._settings.llm)
        return self._models.llm

    def _fuse(self, rankings: dict[str, list[tuple[str, float]]]) -> list[tuple[str, float]]:
        cfg = self._settings.retrieval
        if cfg.fusion == "rrf":
            return rrf(rankings, cfg.rrf_k)
        return weighted(rankings, {"vector": cfg.vector_weight, "bm25": 1 - cfg.vector_weight})

    def list_kbs(self) -> list[str]:
        return self._store.list_kbs(self.namespace)

    def reindex(self, source: str, target: str) -> int:
        """Build kb `target` from `source`'s stored documents under this engine's settings (a new
        model or chunker); returns the number of documents. `source` is left as it is: switching
        means pointing clients at `target`. Refuses — writing nothing — when `target` exists or a
        document has no stored text; a failure halfway drops the half-built `target`."""
        source_ref, target_ref = self._ref(source), self._ref(target)
        if self._store.get_kb(source_ref) is None:
            raise KbNotFound.of(source, self.namespace)
        missing = self._store.count_missing_text(source_ref)
        if missing:
            raise ConfigError(
                f"{missing} document(s) in {source!r} have no stored text (ingested before "
                "lite-rag kept it): re-ingest them, then reindex"
            )
        # Created here, atomically: a kb someone else creates meanwhile is never written into
        # or deleted by this reindex.
        if not self._store.create_kb(self.kb_info(target)):
            raise ConfigError(f"knowledge base {target!r} already exists: reindex into a new name")
        copied = 0
        try:
            for d in self._store.documents(source_ref):  # page by page: never all in memory
                if d.text is None:  # added since the check: refuse rather than skip it
                    raise ConfigError(f"{d.doc_id!r} in {source!r} has no stored text")
                self.upsert_text(target, d.doc_id, d.title, d.text, d.metadata, d.source, d.pages)
                copied += 1
            stats = self._store.stats(target_ref)
            if stats is None or stats.documents != copied:
                raise StoreError(f"reindex of {source!r}: {target!r} does not hold every document")
        except BaseException:
            self.delete_kb(target)
            raise
        return copied

    def stats(self, kb: str) -> KbStats | None:
        return self._store.stats(self._ref(kb))

    def close(self) -> None:
        self._store.close()


def _best_per_parent(
    ranking: list[tuple[str, float]], stored: dict[str, StoredChunk]
) -> list[tuple[str, float]]:
    """The ranking with one chunk per parent (parent_child): the best child stands for its
    parent, so k hits are k distinct parents. Chunks without a parent all stay."""
    seen: set[tuple[str, int]] = set()
    out = []
    for chunk, score in ranking:
        parent = stored[chunk].parent
        if parent is not None:
            key = (stored[chunk].doc_id, parent.n)
            if key in seen:
                continue
            seen.add(key)
        out.append((chunk, score))
    return out
