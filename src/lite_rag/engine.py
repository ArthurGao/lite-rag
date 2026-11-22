"""`RagRuntime` and `RagEngine` — the entry point for the library, the CLI and the HTTP service
(design §3).

A `RagRuntime` holds what does not depend on a namespace: settings, store, embedder, token
counter, chunker, and models built on first use. It is built once (the server keeps one for its
lifetime) and hands out engines.

A `RagEngine` is that runtime seen from ONE namespace (e.g. `tenderted/tenant_42`): every
knowledge base it touches lives inside that namespace, so one project or tenant can never read
or write another's index — isolation is enforced here, not left to callers (design §4). An
engine is cheap: one per request is fine.

It also owns the index-identity rule: a kb records the embedding model, its prefixes and the
chunker it was built with, and opening it with anything else raises (design §3.2).
"""

import hashlib
import json
import logging
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import lite_model
from lite_model import LiteModel
from lite_model.hooks import Registration

from lite_rag.answer import Answer, check_citations, prompt
from lite_rag.cache import CacheBinding, cache_context, check_cache, log_question, open_cache
from lite_rag.chunking import Chunker, build_chunker, chunk_context, embedding_text
from lite_rag.config import EmbeddingConfig, Mode, RetrievalPath, Role, Settings
from lite_rag.contextual import Contextualizer
from lite_rag.embedding import Embedder, build_embedder
from lite_rag.errors import ConfigError, InvalidRequest, KbMismatch, KbNotFound, StoreError
from lite_rag.fusion import rrf, weighted
from lite_rag.graph import GraphExtractor, entity_key, seeds, walk
from lite_rag.llm import LLM, RoleLLM, build_model
from lite_rag.loaders import doc_id_for, file_doc_id, load_file
from lite_rag.model_profiles import Prefixes, prefixes_for
from lite_rag.ranking import Key, ranked
from lite_rag.rerank import Reranker, build_reranker
from lite_rag.route import Router
from lite_rag.stores.base import (
    CachedStore,
    GraphStore,
    IndexedChunk,
    KbInfo,
    KbRef,
    KbStats,
    StoredChunk,
    check_where,
)
from lite_rag.stores.factory import open_store
from lite_rag.text import TokenCounter, build_counter, terms
from lite_rag.types import DEFAULT_EMBEDDING, ChunkGraph, Document, Hit, Route, RouteMode

NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}(/[a-z0-9][a-z0-9_.-]{0,62}){0,3}$")
KB_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Indexer:
    """One named embedder and what is bound to it (S22): the token counter sized by its model's
    tokenizer, and the chunker that counts with it (semantic chunking embeds with it too)."""

    embedder: Embedder
    counter: TokenCounter
    chunker: Chunker


@dataclass(frozen=True)
class Upserted:
    doc_id: str
    chunks: int
    # Unchanged: nothing was re-chunked or re-embedded. Best effort: two identical upserts
    # racing may both report False (the stored result is the same either way).
    skipped: bool


def content_hash(
    title: str, text: str, metadata: dict[str, Any], pages: tuple[int, ...] = ()
) -> str:
    # Pages join only when present, so documents without pages keep the hashes they had.
    parts: list[Any] = [title, text, metadata, *([list(pages)] if pages else [])]
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def check_namespace(namespace: str | None) -> str:
    if not namespace:
        raise InvalidRequest("a namespace is required (e.g. 'myproject' or 'myproject/tenant_42')")
    if not NAMESPACE.match(namespace):
        raise InvalidRequest(
            f"invalid namespace {namespace!r}: lowercase letters, digits, '_', '-', '.', "
            "up to 4 levels separated by '/'"
        )
    return namespace


class RagRuntime:
    def __init__(
        self,
        settings: Settings,
        store: CachedStore,
        embedder: Embedder,
        *,
        reranker: Reranker | None = None,
        llm: LLM | None = None,
        context_llm: LLM | None = None,
        graph_llm: LLM | None = None,
        route_llm: LLM | None = None,
        model: LiteModel | None = None,
        embedders: Mapping[str, Embedder] | None = None,
    ) -> None:
        """`reranker` and the LLMs are built from `settings` on first use when not given, so a
        search-only runtime never needs an LLM key or the rerank model. `model`: the lite-model
        every role not given uses (default: built from `settings.llm`); the cache hook goes on
        it."""
        self.settings = settings
        self.store = store
        # S22: the default plus `settings.embeddings`, each built on first use (a search of
        # default kbs never needs another embedder's key); `embedders` supplies instances.
        self._indexers = {DEFAULT_EMBEDDING: self._indexer_of(settings.embedding, embedder)}
        self._given = dict(embedders or {})
        self._warned: set[str] = set()  # warn_once
        unknown = sorted(set(self._given) - set(settings.embeddings))
        if unknown:
            raise ConfigError(f"embedders given for names not in settings.embeddings: {unknown}")
        self._reranker = reranker
        self._model: LiteModel | None = model  # lite-model, for every role not given
        self._cache: CacheBinding | None = None  # lite-cache, when settings.cache.enabled (S18)
        self._cache_hook: Registration | None = None
        self._llm = llm
        self._context_llm = context_llm
        self._contextualizer: Contextualizer | None = None
        self._graph_llm = graph_llm
        self._graph_extractor: GraphExtractor | None = None
        self._route_llm = route_llm
        self._router: Router | None = None
        self._lock = threading.Lock()  # each model is built once, also under concurrent requests

    @classmethod
    def from_settings(cls, settings: Settings, *, max_connections: int = 4) -> "RagRuntime":
        """`max_connections`: the store's pool (the server passes its worker count, so no
        request waits on the pool). An enabled cache is checked here (its extra and prompt
        secret), like the reranker: a server must not start healthy and fail on a request."""
        if settings.cache.enabled:
            check_cache(settings)
        store = open_store(settings.store, settings.database_url, max_connections=max_connections)
        try:
            embedder = build_embedder(settings.embedding, cache=store)
            # Built now when enabled, so a misconfigured reranker fails at startup (its model
            # still loads on first use). The LLMs stay lazy: search needs no key.
            reranker = build_reranker(settings.rerank) if settings.rerank.enabled else None
            return cls(settings, store, embedder, reranker=reranker)  # builds the chunker too
        except BaseException:
            store.close()
            raise

    def engine(self, namespace: str) -> "RagEngine":
        return RagEngine(self, namespace)

    def warm_up(self) -> None:
        """Load lazily built models now — jieba's dictionary (about a second), and the reranker
        when enabled: the first request should not pay for them. The embedding model is not
        touched: that could be a paid call on every start."""
        terms("warm up")
        if self.settings.rerank.enabled:
            self.reranker().score("warm up", ["warm up"])

    def health(self, deep: bool = False) -> None:
        """Raises when the store (and, `deep`, the embedding service) cannot be reached. The
        deep probe is a cached query embedding: free after the first call."""
        self.store.list_kbs("lite-rag-health")
        if deep:
            self.embedder.embed_query("lite-rag health check")

    def _indexer_of(self, config: EmbeddingConfig, embedder: Embedder) -> Indexer:
        counter = build_counter(self.settings.chunking, config)
        # semantic chunking embeds sentences with the kb's own embedder (and its cache)
        chunker = build_chunker(self.settings.chunking, counter, embedder.embed_documents)
        return Indexer(embedder, counter, chunker)

    @property
    def embedder(self) -> Embedder:
        """The default embedder (`settings.embedding`)."""
        return self._indexers[DEFAULT_EMBEDDING].embedder

    @property
    def counter(self) -> TokenCounter:
        return self._indexers[DEFAULT_EMBEDDING].counter

    @property
    def chunker(self) -> Chunker:
        return self._indexers[DEFAULT_EMBEDDING].chunker

    def warn_once(self, key: str, message: str, *args: object) -> None:
        """Log `message` the first time `key` is seen by this runtime."""
        with self._lock:
            if key in self._warned:
                return
            self._warned.add(key)
        log.warning(message, *args)

    @property
    def embedding_names(self) -> list[str]:
        return [DEFAULT_EMBEDDING, *self.settings.embeddings]

    def indexer(self, name: str) -> Indexer:
        """The named embedder with its counter and chunker; an unknown name is InvalidRequest."""
        if (found := self._indexers.get(name)) is not None:
            return found
        if name not in self.settings.embeddings:
            raise InvalidRequest(
                f"no embedder {name!r}: configured are {', '.join(self.embedding_names)}"
            )
        with self._lock:
            if name not in self._indexers:
                config = self.settings.embeddings[name]
                embedder = self._given.get(name) or build_embedder(config, cache=self.store)
                self._indexers[name] = self._indexer_of(config, embedder)
            return self._indexers[name]

    def signature(self, name: str) -> tuple[str, str, Prefixes]:
        """What a kb records of an embedder besides its dim — provider, model, prefixes — known
        without building it (so opening a kb never needs another embedder's key)."""
        built = self._indexers.get(name)
        embedder = built.embedder if built else self._given.get(name)
        if embedder is not None:
            return embedder.provider, embedder.model, embedder.prefixes
        c: EmbeddingConfig = self.settings.embeddings[name]
        return c.provider, c.model, prefixes_for(c.model, c.query_prefix, c.document_prefix)

    def reranker(self) -> Reranker:
        with self._lock:
            if self._reranker is None:
                self._reranker = build_reranker(self.settings.rerank)
            return self._reranker

    def llm(self) -> LLM:
        with self._lock:
            if self._llm is None:
                self._llm = self._role("answer")
            return self._llm

    def _role(self, role: Role) -> LLM:
        """`role` of the runtime's one lite-model, built (and its whole config checked) on first
        use; the caller holds the lock."""
        if self._model is None:
            self._model = build_model(self.settings.llm)
        if self.settings.cache.enabled and self._cache is None:
            self._cache = open_cache(self.settings)
            try:
                self._cache_hook = self._model.hooks.register(self._cache.hook)
            except lite_model.ConfigError as exc:  # e.g. the host registered its own CacheHook
                raise ConfigError(f"cannot register lite-cache on this model: {exc}") from exc
        return RoleLLM(self._model, role)

    def contextualizer(self) -> Contextualizer:
        """One for the runtime: its `concurrency` bounds the context calls of every ingest
        running at once, not of each."""
        with self._lock:
            if self._contextualizer is None:
                cfg = self.settings.chunking.contextual
                llm = self._context_llm or self._role("contextual")
                self._contextualizer = Contextualizer(
                    llm, self.counter, cfg.max_doc_tokens, cfg.concurrency
                )
            return self._contextualizer

    def graph_extractor(self) -> GraphExtractor:
        """One for the runtime, like the contextualizer: its `concurrency` bounds every ingest."""
        with self._lock:
            if self._graph_extractor is None:
                cfg = self.settings.graph
                llm = self._graph_llm or self._role("graph")
                self._graph_extractor = GraphExtractor(llm, cfg.concurrency)
            return self._graph_extractor

    def router(self) -> Router:
        """The retrieval router (S21), built on first use: a search that does not route never
        needs its model."""
        if self._router is not None:  # every routed search: no lock once built
            return self._router
        with self._lock:
            if self._router is None:
                cfg = self.settings.retrieval
                llm = self._route_llm or self._role("route")
                self._router = Router(llm, cfg.router.modes, cfg.hybrid_paths)
            return self._router

    def close(self) -> None:
        if self._cache_hook is not None:  # an injected model may outlive this runtime
            self._cache_hook.remove()
        if self._cache is not None:
            self._cache.close()
        self.store.close()


class RagEngine:
    def __init__(self, runtime: RagRuntime, namespace: str) -> None:
        self.namespace = check_namespace(namespace)
        self.runtime = runtime
        self._settings = runtime.settings
        self._store = runtime.store
        # the default's chunker: `children_per_parent` comes from the chunking settings, the same
        # for every embedder (tests lower it to force a deeper search)
        self._chunker = runtime.chunker
        self._ready: set[KbRef] = set()  # kbs checked against the configuration
        self._embedding: dict[KbRef, str] = {}  # each opened kb's embedder (S22)

    @classmethod
    def from_settings(cls, settings: Settings, namespace: str | None = None) -> "RagEngine":
        """An engine on its own runtime (the library's shortcut); `close` closes that runtime."""
        ns = check_namespace(namespace or settings.namespace)  # before connecting to anything
        return RagRuntime.from_settings(settings).engine(ns)

    def create_kb(self, kb: str, *, embedding: str | None = None) -> KbInfo:
        """Create `kb` with this engine's settings and the named embedder (default: `default`),
        or check an existing one against them; `embedding` names an embedder of
        `settings.embeddings` (S22). The result says which embedder the kb is opened with."""
        if embedding is not None:
            self.runtime.indexer(embedding)  # an unknown name: InvalidRequest, before any write
        return self._check(kb, self._ref(kb), create=True, embedding=embedding)

    def _ref(self, kb: str) -> KbRef:
        if not KB_NAME.match(kb):
            raise InvalidRequest(
                f"invalid knowledge base name {kb!r}: lowercase letters, digits, '_', '-', '.'"
            )
        return KbRef(self.namespace, kb)

    def kb_info(self, kb: str, embedding: str = DEFAULT_EMBEDDING) -> KbInfo:
        """What `kb` must have been built with to be opened with the named embedder."""
        ix = self.runtime.indexer(embedding)
        e = ix.embedder
        return KbInfo(
            self._ref(kb),
            e.provider,
            e.model,
            e.dim,
            e.prefixes.query,
            e.prefixes.document,
            {
                **self._settings.chunking.identity(self._settings.llm),
                "counter": ix.counter.identity,
                **self._settings.graph.index_identity(self._settings.llm),
            },
            embedding,
        )

    def _resolve(self, kb: str, stored: KbInfo, asked: str | None) -> str:
        """The configured embedder that built `stored` (S22): the one whose provider, model and
        prefixes it recorded — and dim, when that is still ambiguous. None is KbMismatch."""
        recorded = stored.signature()
        names = [n for n in self.runtime.embedding_names if self.runtime.signature(n) == recorded]
        if len(names) > 1:  # same model, other dimensions: building them tells
            names = [n for n in names if self.runtime.indexer(n).embedder.dim == stored.dim]
        if asked is not None and names and asked not in names:
            raise KbMismatch(
                f"knowledge base {kb!r} in namespace {self.namespace!r} was built with embedder"
                f" {names[0]!r}, not {asked!r} (switching model = a new kb + reindex)"
            )
        if not names:  # what the default would need, as before S22
            configured = ", ".join(self.runtime.embedding_names)
            raise self._mismatch(
                kb, stored.differences(self.kb_info(kb)), f"no configured embedder ({configured})"
            )
        return asked or names[0]

    def _mismatch(self, kb: str, diffs: list[str], why: str = "") -> KbMismatch:
        return KbMismatch(
            f"knowledge base {kb!r} in namespace {self.namespace!r} was built differently"
            f" ({'; '.join(diffs)}){f': {why} matches it' if why else ''}. Use the settings it"
            " was built with, or a new kb name (switching model = a new kb + re-embedding)."
        )

    def _open(self, kb: str, *, create: bool, embedding: str | None = None) -> KbRef:
        """Check that `kb` was built with this configuration. Writes create a missing kb and
        check once per engine (a kb deleted later fails their locked write with `KbNotFound`);
        reads check every time and raise `KbNotFound`, so a mistyped name — or a kb dropped by
        another process — is an error, never an empty result."""
        ref = self._ref(kb)
        if embedding is not None:
            self.runtime.indexer(embedding)  # an unknown name: InvalidRequest, before any write
        if not (create and ref in self._ready and embedding in (None, self._embedding[ref])):
            self._check(kb, ref, create=create, embedding=embedding)
        return ref

    def _check(self, kb: str, ref: KbRef, *, create: bool, embedding: str | None) -> KbInfo:
        """The stored kb, checked against the embedder that built it (S22: resolved from what
        it recorded; a write creates a missing kb with the asked one, default `default`)."""
        if create:
            stored = self._store.get_or_create_kb(self.kb_info(kb, embedding or DEFAULT_EMBEDDING))
        elif (found := self._store.get_kb(ref)) is not None:
            stored = found
        else:
            raise KbNotFound.of(kb, self.namespace)
        name = self._resolve(kb, stored, embedding)
        if diffs := stored.differences(self.kb_info(kb, name)):
            raise self._mismatch(kb, diffs)
        self._embedding[ref] = name
        self._ready.add(ref)
        return replace(stored, embedding=name)

    def _indexer(self, ref: KbRef) -> Indexer:
        """The embedder an opened kb was built with."""
        return self.runtime.indexer(self._embedding[ref])

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
        ix = self._indexer(ref)
        parents, chunks = ix.chunker.split_parents(doc)
        contexts: list[str | None] = [None] * len(chunks)
        if self._settings.chunking.contextual.enabled:
            with cache_context(self._settings, self.namespace, "contextual"):
                contexts = list(self.runtime.contextualizer().contexts(doc, chunks))
        graphs = [ChunkGraph()] * len(chunks)
        if self._settings.graph.enabled:
            with cache_context(self._settings, self.namespace, "graph"):
                graphs = self.runtime.graph_extractor().extract(chunks)
        inputs = [embedding_text(c, ctx) for c, ctx in zip(chunks, contexts, strict=True)]
        vectors = ix.embedder.embed_documents(inputs) if inputs else []
        indexed = [
            IndexedChunk(c, v, terms(t), ctx, g)
            for c, v, t, ctx, g in zip(chunks, vectors, inputs, contexts, graphs, strict=True)
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
        """The best `top_k` chunks for `query` (design §5). `mode`: `vector`, `bm25`, `graph`
        (needs graph.enabled) or `hybrid` (the `retrieval.hybrid_paths`, fused). `rerank`
        (default: config) re-scores the best `rerank.top_n` candidates with a cross-encoder.
        `where` keeps only documents whose metadata has each key equal to its scalar value. Each
        hit carries every stage's score and rank; `score` is the final one."""
        refs = [self._open(kb, create=False)]
        return self._search(refs, query, top_k=top_k, mode=mode, rerank=rerank, where=where)

    def search_kbs(
        self,
        kbs: Sequence[str],
        query: str,
        *,
        top_k: int | None = None,
        mode: Mode | None = None,
        rerank: bool | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """`search` over several kbs of this namespace as one (design §5: one query embedding,
        one fusion, one rerank; each hit names its `kb`). A kb that does not exist yet is
        skipped, so a mistyped (valid) name returns nothing rather than an error."""
        if not kbs:
            raise InvalidRequest("search_kbs needs at least one knowledge base")
        names = [self._ref(kb).name for kb in dict.fromkeys(kbs)]  # all checked before any read
        refs = []
        for kb in names:
            try:
                refs.append(self._open(kb, create=False))
            except KbNotFound:
                pass
        return self._search(refs, query, top_k=top_k, mode=mode, rerank=rerank, where=where)

    def _search(
        self,
        refs: list[KbRef],
        query: str,
        *,
        top_k: int | None,
        mode: Mode | None,
        rerank: bool | None,
        where: dict[str, Any] | None,
    ) -> list[Hit]:
        cfg = self._settings.retrieval
        where = check_where(where)
        k = cfg.top_k if top_k is None else top_k
        if k < 1:
            raise InvalidRequest(f"top_k must be at least 1, got {k}")
        mode = mode or cfg.mode
        route = None
        if mode == "route":  # one routing call per search, before any retrieval (S21)
            if not refs:  # no kb to search: no model call either (Codex S21)
                return []
            with cache_context(self._settings, self.namespace, "route", query):
                route = self.runtime.router().choose(query)
            mode = route.mode
        paths = self._paths(mode)
        reranking = self._settings.rerank.enabled if rerank is None else rerank
        if not refs:
            return []
        # `pool` candidates reach the end (all of them re-scored when reranking); each path
        # returns `n` — more in hybrid, where fusion needs candidates from both lists, and more
        # when several chunks share a parent (parent_child), which count once.
        pool = max(self._settings.rerank.top_n, k) if reranking else k
        wanted = pool * self._chunker.children_per_parent  # chunks that make `pool` parents
        n = max(wanted, k * cfg.candidate_factor) if mode == "hybrid" else wanted
        # S22: one query vector per embedder among the kbs; kbs of one embedder share it
        names = [self._embedding[ref] for ref in refs]
        if len(set(names)) > 1 and not reranking:
            self.runtime.warn_once(
                f"mixed:{sorted(set(names))}",
                "search across embedders (%s) without rerank: each kb's best ties in the rank"
                " fusion, so the top hit is arbitrary between kbs (S22: hit@1 0.46 vs 1.00);"
                " enable rerank for comparable scores",
                ", ".join(dict.fromkeys(names)),
            )
        vectors = (
            {n: self.runtime.indexer(n).embedder.embed_query(query) for n in dict.fromkeys(names)}
            if paths & {"vector", "graph"}
            else {}
        )
        # Each kb's graph is walked once: deeper rounds below only re-rank what it reached.
        graph = [self._reach(ref, query, where) for ref in refs] if "graph" in paths else []
        while True:
            # Keyed (kb index, chunk id): a chunk id is unique within its kb only. Each path's
            # lists from every kb are merged by score into one, then fused once.
            per_kb = [
                self._rank(
                    ref, query, vectors.get(names[i]), paths, n, where, graph[i] if graph else None
                )
                for i, ref in enumerate(refs)
            ]
            full = any(len(found) == n for r in per_kb for found in r.values())
            merged: dict[str, dict[Key, float]] = {}
            for i, r in enumerate(per_kb):
                for path, found in r.items():
                    merged.setdefault(path, {}).update(((i, c), s) for c, s in found)
            rankings = {
                path: self._across_embedders(per_kb, names, path, n)
                if path in ("vector", "graph")
                else ranked(scores, n)
                for path, scores in merged.items()
            }
            if mode == "hybrid":
                rankings["fused"] = self._fuse(rankings)
            ranking = rankings["fused" if mode == "hybrid" else mode]
            stored = self._stored(refs, [key for key, _ in ranking])
            # Look up every ranked chunk: one deleted since it was ranked makes room for the
            # next. Short of `pool` while a path came back full (more exists): look deeper —
            # `children_per_parent` is an estimate, and deletes can race a search.
            candidates = _best_per_parent([(c, s) for c, s in ranking if c in stored], stored)
            if len(candidates) >= pool or not full:
                break
            n *= 2
        candidates = candidates[:pool]
        if reranking and candidates:
            candidates = rankings["rerank"] = self._rerank(query, candidates, stored)
        ranks = {s: {c: i for i, (c, _) in enumerate(r, start=1)} for s, r in rankings.items()}
        scores = {s: dict(r) for s, r in rankings.items()}
        # a hit's vector / graph score stays its similarity under its own kb's model
        scores |= {p: merged[p] for p in merged.keys() & {"vector", "graph"}}
        return [
            self._hit(refs[key[0]].name, key, stored[key], score, scores, ranks, route)
            for key, score in candidates[:k]
        ]

    def _across_embedders(
        self, per_kb: list[dict[str, list[tuple[str, float]]]], names: list[str], path: str, n: int
    ) -> list[tuple[Key, float]]:
        """`path`'s lists over every kb as one (S22): each embedder's kbs merged by similarity;
        several embedders' lists fused by RRF — ranks only, as two models' scores do not
        compare. One embedder (the usual case) is the merge by similarity alone, as before."""
        groups: dict[str, dict[Key, float]] = {}
        for i, r in enumerate(per_kb):
            groups.setdefault(names[i], {}).update(((i, c), s) for c, s in r.get(path, []))
        lists = {name: ranked(scores, n) for name, scores in groups.items()}
        if len(lists) == 1:
            return next(iter(lists.values()))
        return rrf(lists, self._settings.retrieval.rrf_k)[:n]

    def _stored(self, refs: list[KbRef], keys: list[Key]) -> dict[Key, StoredChunk]:
        """The stored chunks of `keys`: one lookup per kb."""
        wanted: dict[int, list[str]] = {}
        for i, chunk in keys:
            wanted.setdefault(i, []).append(chunk)
        return {
            (i, chunk): c
            for i, chunks in wanted.items()
            for chunk, c in self._store.chunks(refs[i], chunks).items()
        }

    def _rank(
        self,
        ref: KbRef,
        query: str,
        vector: list[float] | None,
        paths: set[RetrievalPath],
        n: int,
        where: dict[str, Any] | None,
        reached: list[str] | None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Each path's top `n` in one kb (fusion happens once, over every kb's lists).
        `reached`: the chunks the kb's graph reached (graph path only)."""
        rankings: dict[str, list[tuple[str, float]]] = {}
        if "vector" in paths:
            assert vector is not None  # embedded for these paths
            rankings["vector"] = self._store.vector_query(ref, vector, n, where)
        if "bm25" in paths:
            bm = self._settings.retrieval.bm25
            rankings["bm25"] = self._store.bm25_query(ref, terms(query), n, bm.k1, bm.b, where)
        if "graph" in paths:
            assert vector is not None and reached is not None
            # the graph narrows, the query vector orders
            rankings["graph"] = (
                self._store.vector_query(ref, vector, n, where, among=reached) if reached else []
            )
        return rankings

    def _paths(self, mode: RouteMode) -> set[RetrievalPath]:
        """The retrieval paths `mode` runs: itself, or for hybrid the configured ones."""
        paths: set[RetrievalPath] = (
            set(self._settings.retrieval.hybrid_paths) if mode == "hybrid" else {mode}
        )
        if "graph" in paths and not self._settings.graph.enabled:
            raise InvalidRequest(
                "the graph path needs graph.enabled and a kb built with it (design §5, S11)"
            )
        return paths

    def _reach(self, ref: KbRef, query: str, where: dict[str, Any] | None) -> list[str]:
        """The chunks the kb's graph reaches from the entities the query names (graph.py),
        walking only documents `where` keeps."""
        store = self._store
        if not isinstance(store, GraphStore):  # the configuration refuses this; say so anyway
            raise ConfigError("graph retrieval needs the postgres backend (design §5, S11)")
        cfg = self._settings.graph
        return walk(
            seeds(store.graph_names_in(ref, entity_key(query), where)),
            lambda keys: store.graph_chunks_of(ref, keys, where),
            lambda chunks: store.graph_entities_of(ref, chunks),
            hops=cfg.hops,
            hub_chunks=cfg.hub_chunks,
            max_neighbors=cfg.max_neighbors,
        )

    @staticmethod
    def _hit(
        kb: str,
        key: Key,
        c: StoredChunk,
        score: float,
        scores: dict[str, dict[Key, float]],
        ranks: dict[str, dict[Key, int]],
        route: Route | None = None,
    ) -> Hit:
        """A hit for chunk `c` (ranked under `key`); a parent_child child returns its parent's
        text and position."""
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
            scores={s: m[key] for s, m in scores.items() if key in m},
            ranks={s: m[key] for s, m in ranks.items() if key in m},
            kb=kb,
            route=route,
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
        no hits the LLM is not called. With `cache.question_log` set, the question is logged
        first (S18)."""
        if self._settings.cache.question_log is not None:
            log_question(self._settings.cache.question_log, self.namespace, question)
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
        with cache_context(self._settings, self.namespace, "answer", question):
            completion = self.runtime.llm().complete(prompt(question, hits))
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
            cached=completion.cached,
        )

    def _rerank(
        self, query: str, candidates: list[tuple[Key, float]], stored: dict[Key, StoredChunk]
    ) -> list[tuple[Key, float]]:
        """The candidates re-scored by the cross-encoder, which reads each chunk with its
        title and heading path."""
        ids = [chunk for chunk, _ in candidates]
        texts = [
            f"{chunk_context(stored[c].title, stored[c].heading_path)}\n{stored[c].text}"
            for c in ids
        ]
        return ranked(dict(zip(ids, self.runtime.reranker().score(query, texts), strict=True)))

    def _fuse(self, rankings: dict[str, list[tuple[Key, float]]]) -> list[tuple[Key, float]]:
        cfg = self._settings.retrieval
        if cfg.fusion == "rrf":
            return rrf(rankings, cfg.rrf_k)
        return weighted(rankings, {"vector": cfg.vector_weight, "bm25": 1 - cfg.vector_weight})

    def list_kbs(self) -> list[str]:
        return self._store.list_kbs(self.namespace)

    def reindex(self, source: str, target: str, *, embedding: str | None = None) -> int:
        """Build kb `target` from `source`'s stored documents under this engine's settings (a new
        model or chunker); returns the number of documents. `source` is left as it is: switching
        means pointing clients at `target`. Refuses — writing nothing — when `target` exists or a
        document has no stored text; a failure halfway drops the half-built `target`."""
        source_ref, target_ref = self._ref(source), self._ref(target)
        if self._store.get_kb(source_ref) is None:
            raise KbNotFound.of(source, self.namespace)
        missing = self._store.count_missing_text(source_ref)
        if missing:
            raise InvalidRequest(
                f"{missing} document(s) in {source!r} have no stored text (ingested before "
                "lite-rag kept it): re-ingest them, then reindex"
            )
        # Created here, atomically: a kb someone else creates meanwhile is never written into
        # or deleted by this reindex.
        if not self._store.create_kb(self.kb_info(target, embedding or DEFAULT_EMBEDDING)):
            raise InvalidRequest(
                f"knowledge base {target!r} already exists: reindex into a new name"
            )
        copied = 0
        try:
            for d in self._store.documents(source_ref):  # page by page: never all in memory
                if d.text is None:  # added since the check: refuse rather than skip it
                    raise InvalidRequest(f"{d.doc_id!r} in {source!r} has no stored text")
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
        """Closes the runtime — shared by every engine it handed out."""
        self.runtime.close()


def _best_per_parent(
    ranking: list[tuple[Key, float]], stored: dict[Key, StoredChunk]
) -> list[tuple[Key, float]]:
    """The ranking with one chunk per parent (parent_child): the best child stands for its
    parent, so k hits are k distinct parents. Chunks without a parent all stay."""
    seen: set[tuple[int, str, int]] = set()
    out = []
    for chunk, score in ranking:
        parent = stored[chunk].parent
        if parent is not None:
            key = (chunk[0], stored[chunk].doc_id, parent.n)
            if key in seen:
                continue
            seen.add(key)
        out.append((chunk, score))
    return out
