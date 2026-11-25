"""Runtimes and engines as `RagRuntime.from_settings` builds them, but with a fake embedder."""

from typing import Any

from lite_rag.config import load_settings
from lite_rag.embedding import CachingEmbedder, Embedder
from lite_rag.engine import RagEngine, RagRuntime
from lite_rag.llm import LLM
from lite_rag.rerank import Reranker
from lite_rag.stores.base import CachedStore
from lite_rag.testing import FakeEmbedder

NS = "proj/tenant_1"
CHUNKING = {"strategy": "paragraph", "unit": "chars", "size": 40, "overlap": 0, "min_size": 0}


def make_runtime(
    store: CachedStore,
    embedder: FakeEmbedder | None = None,
    *,
    cache: bool = True,
    retrieval: dict[str, Any] | None = None,
    rerank: dict[str, Any] | None = None,
    reranker: Reranker | None = None,
    llm: LLM | None = None,
    graph: dict[str, Any] | None = None,
    graph_llm: LLM | None = None,
    route_llm: LLM | None = None,
    llm_settings: dict[str, Any] | None = None,
    embedders: dict[str, Embedder] | None = None,
    trace: dict[str, Any] | None = None,
    **chunking: Any,
) -> RagRuntime:
    """`embedders`: named embedders besides the default (S22): each name is added to the
    settings' `embeddings` (a placeholder config) and served by the instance given."""
    settings = load_settings(
        chunking={**CHUNKING, **chunking},
        embedding={"cache": cache},
        embeddings={name: {"model": f"placeholder-{name}"} for name in embedders or {}},
        retrieval=retrieval or {},
        rerank=rerank or {},
        graph=graph or {},
        llm=llm_settings or {},
        trace=trace or {},
    )
    inner = embedder or FakeEmbedder(dim=8)
    wrapped: Embedder = CachingEmbedder(inner, store) if cache else inner
    return RagRuntime(
        settings,
        store,
        wrapped,
        reranker=reranker,
        llm=llm,
        graph_llm=graph_llm,
        route_llm=route_llm,
        embedders=embedders,
    )


def make_engine(
    store: CachedStore,
    embedder: FakeEmbedder | None = None,
    *,
    namespace: str = NS,
    **kwargs: Any,
) -> RagEngine:
    return make_runtime(store, embedder, **kwargs).engine(namespace)
