"""Embedders (design §3.2). Documents and queries are embedded separately: many models want a query
prefix or an input type, so `embed_query` is not assumed to equal `embed_documents([q])[0]`. The
prefixes come from `model_profiles` and are applied here — callers never add them.

Failures raise `EmbeddingError` after bounded retries of the *same* model (429, 5xx, transport
errors) — never an empty or partial result, and never a different model.
"""

import hashlib
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from itertools import batched
from typing import Any, Protocol

import httpx2

from lite_rag.config import EmbeddingConfig
from lite_rag.errors import ConfigError, LiteRagError
from lite_rag.model_profiles import Prefixes, profile_for
from lite_rag.stores.base import VectorCache

Sleep = Callable[[float], None]


class EmbeddingError(LiteRagError):
    """The embedding service failed after its retries."""


def model_id(provider: str, model: str) -> str:
    """What an index is bound to, e.g. `ollama/bge-m3:567m`."""
    return f"{provider}/{model}"


class Embedder(Protocol):
    """`prefixes` are what this embedder adds by itself; they are part of the index identity,
    because vectors embedded with other prefixes are not comparable."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def prefixes(self) -> Prefixes: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx2.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx2.TransportError)


def _profile_prefixes(config: EmbeddingConfig) -> Prefixes:
    return profile_for(config.model, config.query_prefix, config.document_prefix).prefixes


class _HTTPEmbedder:
    """Batching, bounded concurrency and retry with exponential backoff, for HTTP embedding APIs.
    Subclasses set `provider` and implement `_call`."""

    provider: str

    def __init__(
        self,
        config: EmbeddingConfig,
        client: httpx2.Client | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        if not config.base_url:
            raise ConfigError(f"embedding.base_url is required for the {self.provider} provider")
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._client = client or httpx2.Client(timeout=config.timeout_s)
        self._sleep = sleep
        self._dim = config.dimensions
        self.model = config.model
        self.prefixes = _profile_prefixes(config)

    @property
    def dim(self) -> int:
        if self._dim is None:  # learn it from the model once
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = [self.prefixes.document + t for t in texts]
        batches = [list(b) for b in batched(inputs, self._config.batch_size)]
        if len(batches) <= 1 or self._config.max_concurrency == 1:
            results = [self._with_retry(b) for b in batches]
        else:
            with ThreadPoolExecutor(self._config.max_concurrency) as pool:
                results = list(pool.map(self._with_retry, batches))
        vectors = [v for batch in results for v in batch]
        self._check(vectors, len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        [vector] = self._with_retry([self.prefixes.query + text])
        return vector

    def _check(self, vectors: list[list[float]], expected: int) -> None:
        if len(vectors) != expected:
            raise EmbeddingError(f"{self.model}: asked for {expected} vectors, got {len(vectors)}")
        if vectors and self._dim is not None and len(vectors[0]) != self._dim:
            raise EmbeddingError(
                f"{self.model}: expected {self._dim}-d vectors, got {len(vectors[0])}"
            )

    def _with_retry(self, batch: list[str]) -> list[list[float]]:
        attempts = self._config.max_retries + 1
        for attempt in range(attempts):
            try:
                return self._call(batch)
            except Exception as exc:
                if not _retryable(exc) or attempt == attempts - 1:
                    raise EmbeddingError(
                        f"{model_id(self.provider, self.model)} failed after {attempt + 1} "
                        f"attempt(s): {type(exc).__name__}: {exc}"
                    ) from exc
                self._sleep(min(0.5 * 2**attempt, 8.0))
        raise AssertionError("unreachable")

    def _call(self, batch: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OllamaEmbedder(_HTTPEmbedder):
    """Ollama's `/api/embed` (local, free)."""

    provider = "ollama"

    def _call(self, batch: list[str]) -> list[list[float]]:
        response = self._client.post(
            f"{self._base_url}/api/embed", json={"model": self.model, "input": batch}
        )
        response.raise_for_status()
        embeddings: list[list[float]] = response.json()["embeddings"]
        return embeddings


class OpenAICompatEmbedder(_HTTPEmbedder):
    """`POST {base_url}/embeddings` — OpenAI, and the many services that copy its API (DashScope,
    SiliconFlow, Jina, vLLM, TEI…)."""

    provider = "openai"

    def __init__(
        self,
        config: EmbeddingConfig,
        client: httpx2.Client | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._key = os.environ.get(config.api_key_env) if config.api_key_env else None
        if config.api_key_env and not self._key:
            raise ConfigError(f"embedding.api_key_env names {config.api_key_env}, which is not set")
        super().__init__(config, client, sleep)

    def _call(self, batch: list[str]) -> list[list[float]]:
        body: dict[str, Any] = {"model": self.model, "input": batch}
        if self._config.dimensions:
            body["dimensions"] = self._config.dimensions
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        response = self._client.post(f"{self._base_url}/embeddings", json=body, headers=headers)
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


class FastEmbedEmbedder:
    """In-process ONNX models (extra `[fastembed]`); the model downloads once, then runs offline.
    Uses fastembed's plain `embed` with the profile's prefixes, so the prefixes recorded on the
    index are exactly what was embedded."""

    provider = "fastembed"

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ConfigError(
                "install the fastembed extra: pip install 'lite-rag[fastembed]'"
            ) from exc
        known = {m["model"]: m for m in TextEmbedding.list_supported_models()}
        if config.model not in known:
            raise ConfigError(f"fastembed does not support embedding model {config.model!r}")
        self._config = config
        self.model = config.model
        self.dim = int(known[config.model]["dim"])
        self.prefixes = _profile_prefixes(config)

    @cached_property
    def _impl(self) -> Any:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self.model, cache_dir=self._config.cache_dir)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._impl.embed(texts, batch_size=self._config.batch_size)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.prefixes.document + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        [vector] = self._embed([self.prefixes.query + text])
        return vector


class CachingEmbedder:
    """Reuses vectors already computed for the same model and the same exact input, across every
    kb and namespace. The key covers the role and the prefix, so a document vector is never
    served for a query."""

    def __init__(self, inner: Embedder, cache: VectorCache) -> None:
        self.inner = inner
        self._cache = cache

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def prefixes(self) -> Prefixes:
        return self.inner.prefixes

    def _id(self) -> str:
        return model_id(self.provider, self.model)

    @staticmethod
    def _key(role: str, prefix: str, text: str) -> str:
        return hashlib.sha256(f"{role}\x1f{prefix}\x1f{text}".encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key("document", self.prefixes.document, t) for t in texts]
        found = self._cache.get(self._id(), self.dim, list(dict.fromkeys(keys)))
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in found}
        if missing:
            vectors = self.inner.embed_documents(list(missing.values()))
            fresh = dict(zip(missing, vectors, strict=True))
            self._cache.put(self._id(), self.dim, fresh)
            found.update(fresh)
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        key = self._key("query", self.prefixes.query, text)
        found = self._cache.get(self._id(), self.dim, [key])
        if key not in found:
            found[key] = self.inner.embed_query(text)
            self._cache.put(self._id(), self.dim, {key: found[key]})
        return found[key]


def build_embedder(config: EmbeddingConfig, cache: VectorCache | None = None) -> Embedder:
    """The configured embedder, wrapped in the vector cache when `cache` is given and enabled."""
    embedder: Embedder
    match config.provider:
        case "ollama":
            embedder = OllamaEmbedder(config)
        case "openai":
            embedder = OpenAICompatEmbedder(config)
        case "fastembed":
            embedder = FastEmbedEmbedder(config)
    return CachingEmbedder(embedder, cache) if cache is not None and config.cache else embedder
