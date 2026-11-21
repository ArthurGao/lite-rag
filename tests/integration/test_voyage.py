"""Opt-in: the real Voyage AI API (`VOYAGE_API_KEY=... uv run pytest -m voyage`, S20). Paid per
token (a few tiny calls). Without the key it fails, not skips: an opt-in run that proved nothing
must not look green."""

import math
import os

import pytest

from lite_rag.config import EmbeddingConfig, RerankConfig
from lite_rag.embedding import build_embedder
from lite_rag.rerank import build_reranker

pytestmark = pytest.mark.voyage

DOCS = ["Refunds are paid within 14 days of the request.", "The office is closed on Sundays."]
QUERY = "how long until I get my money back"


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def test_voyage_embeds_and_reranks() -> None:
    assert os.environ.get("VOYAGE_API_KEY"), "set VOYAGE_API_KEY for -m voyage"
    embedder = build_embedder(
        EmbeddingConfig(provider="voyage", model="voyage-4-lite", dimensions=256)
    )
    docs = embedder.embed_documents(DOCS)
    query = embedder.embed_query(QUERY)
    assert [len(v) for v in docs] == [256, 256] and embedder.dim == 256
    assert _cos(query, docs[0]) > _cos(query, docs[1])
    reranker = build_reranker(RerankConfig(provider="voyage", model="rerank-2.5-lite"))
    first, second = reranker.score(QUERY, DOCS)
    assert first > second and reranker.model_id == "voyage/rerank-2.5-lite"
