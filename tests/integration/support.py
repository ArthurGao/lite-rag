"""Engines as `RagEngine.from_settings` builds them, but with a fake embedder."""

from typing import Any

from lite_rag.config import load_settings
from lite_rag.embedding import CachingEmbedder, Embedder
from lite_rag.engine import RagEngine
from lite_rag.llm import LLM
from lite_rag.rerank import Reranker
from lite_rag.stores.base import CachedStore
from lite_rag.testing import FakeEmbedder

NS = "proj/tenant_1"
CHUNKING = {"strategy": "paragraph", "unit": "chars", "size": 40, "overlap": 0, "min_size": 0}


def make_engine(
    store: CachedStore,
    embedder: FakeEmbedder | None = None,
    *,
    namespace: str = NS,
    cache: bool = True,
    retrieval: dict[str, Any] | None = None,
    rerank: dict[str, Any] | None = None,
    reranker: Reranker | None = None,
    llm: LLM | None = None,
    **chunking: Any,
) -> RagEngine:
    settings = load_settings(
        chunking={**CHUNKING, **chunking},
        embedding={"cache": cache},
        retrieval=retrieval or {},
        rerank=rerank or {},
    )
    inner = embedder or FakeEmbedder(dim=8)
    wrapped: Embedder = CachingEmbedder(inner, store) if cache else inner
    return RagEngine(settings, store, wrapped, namespace=namespace, reranker=reranker, llm=llm)
