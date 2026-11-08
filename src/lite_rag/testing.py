"""Deterministic fakes for tests — here and in projects that use lite-rag (e.g. llm-steward's
contract tests). They never touch the network or a model (AGENTS.md rule 6)."""

import hashlib
import math
import random

from lite_rag.model_profiles import Prefixes
from lite_rag.text import ESTIMATE_TOKEN


class FakeEmbedder:
    """Bag-of-hashed-tokens vectors, L2-normalised: texts that share words are close, so retrieval
    tests behave like the real thing without a model. Each CJK character counts as one token."""

    provider = "fake"

    def __init__(
        self, dim: int = 64, model: str = "fake-embedder", prefixes: Prefixes | None = None
    ) -> None:
        self._dim = dim
        self._model = model
        self.prefixes = prefixes or Prefixes()
        self.calls: list[list[str]] = []  # every batch embedded, for assertions
        self.queries: list[str] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % self._dim] += 1.0
        return normalise(vec)


def _tokens(text: str) -> list[str]:
    """The same tokens the estimate counter uses (CJK per character), minus punctuation."""
    return [t for t in ESTIMATE_TOKEN.findall(text.lower()) if t.isalnum()]


def normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class SpreadEmbedder(FakeEmbedder):
    """Distinct, well-spread unit vectors, seeded by the text. Use it where vector-index
    behaviour matters (e.g. HNSW recall): the bag-of-words fake gives many identical
    low-dimensional vectors, which leaves HNSW graph nodes unreachable — real models do not."""

    def _vector(self, text: str) -> list[float]:
        rng = random.Random(hashlib.sha256(text.encode()).digest())
        return normalise([rng.gauss(0, 1) for _ in range(self.dim)])
