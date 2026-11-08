"""Embedders (design §3.2). Documents and queries are embedded separately: many models want a query
prefix or an input type, so `embed_query` is not assumed to equal `embed_documents([q])[0]`. The
prefixes come from `model_profiles` and are applied here — callers never add them.

Failures raise `EmbeddingError` after bounded retries of the *same* model (429, 5xx, transport
errors) — never an empty or partial result, and never a different model.
"""

import hashlib
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from itertools import batched
from typing import Any, ClassVar, Literal, Protocol

import httpx2

from lite_rag.config import OLLAMA_URL, EmbeddingConfig
from lite_rag.errors import ConfigError, LiteRagError
from lite_rag.extras import api_key, require_extra
from lite_rag.model_profiles import Prefixes, prefixes_for
from lite_rag.stores.base import VectorCache
from lite_rag.types import model_id

Sleep = Callable[[float], None]
Role = Literal["document", "query"]  # APIs with an input type get it per call


class EmbeddingError(LiteRagError):
    """The embedding service failed after its retries."""

    status = 502


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


def _too_large(exc: BaseException | None) -> bool:
    return isinstance(exc, httpx2.HTTPStatusError) and exc.response.status_code in (400, 413)


class _Prefixing:
    """Adds the profile's prefixes (part of the index identity), then delegates to `_embed`: the
    one place every embedder writes documents and queries the way its model needs."""

    provider: str

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self.model = config.model
        self.prefixes = prefixes_for(config.model, config.query_prefix, config.document_prefix)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.prefixes.document + t for t in texts], "document")

    def embed_query(self, text: str) -> list[float]:
        [vector] = self._embed([self.prefixes.query + text], "query")
        return vector

    def _embed(self, texts: list[str], role: Role) -> list[list[float]]:
        raise NotImplementedError


class _HTTPEmbedder(_Prefixing):
    """Batching, bounded concurrency and retry with exponential backoff, for HTTP embedding APIs.
    Each provider declares its facts — `default_url`, `key_env` (where its key comes from by
    default; None: no key unless configured) and `max_batch` (its limit on texts per call) — and
    implements `_call`."""

    default_url: str
    key_env: str | None = None
    max_batch: int | None = None

    def __init__(
        self,
        config: EmbeddingConfig,
        client: httpx2.Client | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        super().__init__(config)
        # A hosted API without its key fails here, before any call.
        self._key = api_key(f"embedding ({self.provider})", config.api_key_env, self.key_env)
        self._base_url = (config.base_url or self.default_url).rstrip("/")
        self._batch = min(config.batch_size, self.max_batch or config.batch_size)
        self._client = client or httpx2.Client(timeout=config.timeout_s)
        self._sleep = sleep
        self._dim = config.dimensions

    @property
    def dim(self) -> int:
        if self._dim is None:  # learn it from the model once (set `dimensions` to skip this)
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def _embed(self, texts: list[str], role: Role) -> list[list[float]]:
        batches = [list(b) for b in batched(texts, self._batch)]
        if len(batches) <= 1 or self._config.max_concurrency == 1:
            results = [self._fitting(b, role) for b in batches]
        else:
            with ThreadPoolExecutor(self._config.max_concurrency) as pool:
                results = list(pool.map(lambda b: self._fitting(b, role), batches))
        vectors = [v for batch in results for v in batch]
        self._check(vectors, len(texts))
        return vectors

    def _check(self, vectors: list[list[float]], expected: int) -> None:
        if len(vectors) != expected:
            raise EmbeddingError(f"{self.model}: asked for {expected} vectors, got {len(vectors)}")
        if vectors and self._dim is not None and len(vectors[0]) != self._dim:
            raise EmbeddingError(
                f"{self.model}: expected {self._dim}-d vectors, got {len(vectors[0])}"
            )

    def _fitting(self, batch: list[str], role: Role) -> list[list[float]]:
        """One batch. A batch the API refuses as too large (400 / 413 — e.g. Voyage's limit on
        total tokens per request, which depends on the model) is halved and each half sent
        again, down to a single text; a single text that is refused is an error."""
        try:
            return self._with_retry(batch, role)
        except EmbeddingError as exc:
            if len(batch) == 1 or not _too_large(exc.__cause__):
                raise
            half = len(batch) // 2
            return self._fitting(batch[:half], role) + self._fitting(batch[half:], role)

    def _with_retry(self, batch: list[str], role: Role) -> list[list[float]]:
        attempts = self._config.max_retries + 1
        for attempt in range(attempts):
            try:
                return self._call(batch, role)
            except Exception as exc:
                if not _retryable(exc) or attempt == attempts - 1:
                    raise EmbeddingError(
                        f"{model_id(self.provider, self.model)} failed after {attempt + 1} "
                        f"attempt(s): {type(exc).__name__}: {exc}"
                    ) from exc
                self._sleep(min(0.5 * 2**attempt, 8.0))
        raise AssertionError("unreachable")

    def _call(self, batch: list[str], role: Role) -> list[list[float]]:
        raise NotImplementedError

    def _bearer(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"} if self._key else {}

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> Any:
        response = self._client.post(url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()


class OllamaEmbedder(_HTTPEmbedder):
    """Ollama's `/api/embed` (local, free)."""

    provider = "ollama"
    default_url = OLLAMA_URL

    def _call(self, batch: list[str], role: Role) -> list[list[float]]:
        reply = self._post(f"{self._base_url}/api/embed", {"model": self.model, "input": batch}, {})
        embeddings: list[list[float]] = reply["embeddings"]
        return embeddings


class OpenAICompatEmbedder(_HTTPEmbedder):
    """`POST {base_url}/embeddings` — OpenAI, and the many services that copy its API (DashScope,
    SiliconFlow, Jina, vLLM, TEI…). Many of those need no key: none unless `api_key_env` is set."""

    provider = "openai"
    default_url = "https://api.openai.com/v1"
    dimensions_field = "dimensions"

    def _call(self, batch: list[str], role: Role) -> list[list[float]]:
        body: dict[str, Any] = {"model": self.model, "input": batch, **self._role_fields(role)}
        if self._config.dimensions:
            body[self.dimensions_field] = self._config.dimensions
        reply = self._post(f"{self._base_url}/embeddings", body, self._bearer())
        return [d["embedding"] for d in sorted(reply["data"], key=lambda d: d["index"])]

    def _role_fields(self, role: Role) -> dict[str, str]:
        return {}  # OpenAI's models embed queries and documents alike


class VoyageEmbedder(OpenAICompatEmbedder):
    """Voyage AI `POST /v1/embeddings` (paid): OpenAI's shape plus `input_type`. Voyage prepends
    its own retrieval prompt for that role, so no prefix is added here."""

    provider = "voyage"
    default_url = "https://api.voyageai.com/v1"
    key_env = "VOYAGE_API_KEY"
    max_batch = 1000
    dimensions_field = "output_dimension"

    def _role_fields(self, role: Role) -> dict[str, str]:
        return {"input_type": role}


class CohereEmbedder(_HTTPEmbedder):
    """Cohere `POST /v2/embed` (paid): `input_type` search_document / search_query, float
    vectors; at most 96 texts per call."""

    provider = "cohere"
    default_url = "https://api.cohere.com/v2"
    key_env = "CO_API_KEY"
    max_batch = 96
    INPUT_TYPES: ClassVar[dict[Role, str]] = {
        "document": "search_document",
        "query": "search_query",
    }

    def _call(self, batch: list[str], role: Role) -> list[list[float]]:
        body: dict[str, Any] = {
            "model": self.model,
            "texts": batch,
            "input_type": self.INPUT_TYPES[role],
            "embedding_types": ["float"],
        }
        if self._config.dimensions:
            body["output_dimension"] = self._config.dimensions
        reply = self._post(f"{self._base_url}/embed", body, self._bearer())
        embeddings: list[list[float]] = reply["embeddings"]["float"]
        return embeddings


class GeminiEmbedder(_HTTPEmbedder):
    """Gemini `POST /v1beta/models/{model}:batchEmbedContents` (paid, free tier). One Content per
    text: several parts in one Content are merged into a single vector. The role is a text
    prefix where the model profile gives one (gemini-embedding-2, which has no task type) and
    a `taskType` otherwise (gemini-embedding-001 and other models): never neither."""

    provider = "gemini"
    default_url = "https://generativelanguage.googleapis.com/v1beta"
    key_env = "GEMINI_API_KEY"
    max_batch = 100
    TASK_TYPES: ClassVar[dict[Role, str]] = {
        "document": "RETRIEVAL_DOCUMENT",
        "query": "RETRIEVAL_QUERY",
    }

    def _call(self, batch: list[str], role: Role) -> list[list[float]]:
        name = f"models/{self.model}"
        shared: dict[str, Any] = {"model": name}
        if not (self.prefixes.query or self.prefixes.document):
            shared["taskType"] = self.TASK_TYPES[role]
        if self._config.dimensions:
            shared["outputDimensionality"] = self._config.dimensions
        requests = [{**shared, "content": {"parts": [{"text": t}]}} for t in batch]
        reply = self._post(
            f"{self._base_url}/{name}:batchEmbedContents",
            {"requests": requests},
            {"x-goog-api-key": self._key or ""},
        )
        return [e["values"] for e in reply["embeddings"]]


class FastEmbedEmbedder(_Prefixing):
    """In-process ONNX models (extra `[fastembed]`); the model downloads once, then runs offline.
    Uses fastembed's plain `embed` with the profile's prefixes, so the prefixes recorded on the
    index are exactly what was embedded."""

    provider = "fastembed"

    def __init__(self, config: EmbeddingConfig) -> None:
        known = fastembed_models(require_extra("fastembed", "fastembed").TextEmbedding)
        if config.model not in known:
            raise ConfigError(f"fastembed does not support embedding model {config.model!r}")
        super().__init__(config)
        self.dim = int(known[config.model]["dim"])

    @cached_property
    def _impl(self) -> Any:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self.model, cache_dir=self._config.cache_dir)

    def _embed(self, texts: list[str], role: Role) -> list[list[float]]:
        return [v.tolist() for v in self._impl.embed(texts, batch_size=self._config.batch_size)]


class SentenceTransformerEmbedder(_Prefixing):
    """In-process PyTorch models through sentence-transformers (extra `[sentence-transformers]`):
    any Hugging Face model, on CPU / CUDA / Apple MPS. Like fastembed, plain `encode` with the
    profile's prefixes, so the prefixes recorded on the index are exactly what was embedded."""

    provider = "sentence_transformers"

    @cached_property
    def _impl(self) -> Any:
        """Loaded on first use (importing torch alone takes seconds)."""
        st = require_extra("sentence_transformers", "sentence-transformers")
        return st.SentenceTransformer(
            self.model, cache_folder=self._config.cache_dir, truncate_dim=self._config.dimensions
        )

    @cached_property
    def dim(self) -> int:
        return int(self._impl.get_embedding_dimension())

    def _embed(self, texts: list[str], role: Role) -> list[list[float]]:
        vectors = self._impl.encode(
            texts, batch_size=self._config.batch_size, normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]


def fastembed_models(cls: Any) -> dict[str, dict[str, Any]]:
    """The models a fastembed class supports, by name (its own catalogue, no download)."""
    return {m["model"]: m for m in cls.list_supported_models()}


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


EMBEDDERS: dict[str, Callable[[EmbeddingConfig], Embedder]] = {
    cls.provider: cls
    for cls in (
        OllamaEmbedder,
        OpenAICompatEmbedder,
        FastEmbedEmbedder,
        VoyageEmbedder,
        CohereEmbedder,
        GeminiEmbedder,
        SentenceTransformerEmbedder,
    )
}


def build_embedder(config: EmbeddingConfig, cache: VectorCache | None = None) -> Embedder:
    """The configured embedder, wrapped in the vector cache when `cache` is given and enabled."""
    embedder = EMBEDDERS[config.provider](config)
    return CachingEmbedder(embedder, cache) if cache is not None and config.cache else embedder
