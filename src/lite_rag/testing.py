"""Deterministic fakes for tests — here and in projects that use lite-rag (e.g. llm-steward's
contract tests). They never touch the network or a model (AGENTS.md rule 6)."""

import hashlib
import math
import re

TOKEN = re.compile(r"\w+", re.UNICODE)


class FakeEmbedder:
    """Bag-of-hashed-tokens vectors, L2-normalised: texts that share words are close, so retrieval
    tests behave like the real thing without a model. Each CJK character counts as one token."""

    provider = "fake"

    def __init__(self, dim: int = 64, model: str = "fake-embedder") -> None:
        self._dim = dim
        self._model = model
        self.calls: list[list[str]] = []  # every batch embedded, for assertions

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
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for word in TOKEN.findall(text.lower()):
        if any("一" <= ch <= "鿿" for ch in word):
            out.extend(word)
        else:
            out.append(word)
    return out
