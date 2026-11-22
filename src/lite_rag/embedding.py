"""Embedders (design §3.2). Documents and queries are embedded separately: many models want a query
prefix or an input type, so `embed_query` is not assumed to equal `embed_documents([q])[0]`. The
prefixes come from `model_profiles` and are applied here — callers never add them.

Failures raise `EmbeddingError` after bounded retries of the *same* model (429, 5xx, transport
errors) — never an empty or partial result, and never a different model.
"""

import hashlib
from collections.abc import Callable
from functools import cached_property
from typing import Any, ClassVar, Literal, Protocol

import httpx2
import lite_model
from lite_model import ModelsConfig, RolePolicy, RoleSpec

from lite_rag.config import EmbeddingConfig
from lite_rag.errors import ConfigError, LiteRagError
from lite_rag.extras import require_extra
from lite_rag.llm import build_model
from lite_rag.model_profiles import Prefixes, prefixes_for
from lite_rag.stores.base import VectorCache
from lite_rag.types import model_id

Sleep = Callable[[float], None]
Role = Literal["document", "query"]  # APIs with an input type get it per call

ROLE = "embed"  # the lite-model role of the embedders that call through it (S19)
ROLES = RoleSpec({ROLE: RolePolicy(kind="embed")})  # never routable: index identity


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


class _ProbedDim(_Prefixing):
    """The vector size: `dimensions` when configured, else learnt once from the model; every
    reply is checked against it (part of the index identity, rule 3)."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._dim = config.dimensions

    @property
    def dim(self) -> int:
        if self._dim is None:  # learn it from the model once (set `dimensions` to skip this)
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def _check_dim(self, vectors: list[list[float]]) -> None:
        if vectors and self._dim is not None and len(vectors[0]) != self._dim:
            raise EmbeddingError(
                f"{self.model}: expected {self._dim}-d vectors, got {len(vectors[0])}"
            )


class _LiteModelEmbedder(_ProbedDim):
    """An embedder whose calls are lite-model's `embed()` (S19; its design §3.2): batching,
    bounded concurrency, halving a batch refused as too large, retries of the same model on
    429 / 5xx / transport errors, the vectors' count and order. lite-rag keeps what is index
    identity — the provider name, the model, the prefixes (`_Prefixing`) and the dim — and the
    vector cache (`CachingEmbedder`). The configuration stays lite-rag's own."""

    def __init__(self, config: EmbeddingConfig, client: httpx2.Client | None = None) -> None:
        super().__init__(config)
        models = ModelsConfig(
            backend=self.provider,
            model=config.model,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            backends={self.provider: self._backend(config)},
            roles={ROLE: self._role(config)},
        )
        self._model = build_model(models, client, roles=ROLES)

    def _backend(self, config: EmbeddingConfig) -> dict[str, Any]:
        raise NotImplementedError

    # Whether `dimensions` is sent (index identity: Ollama never sent it, S19).
    sends_dimensions: ClassVar[bool] = True

    def _role(self, config: EmbeddingConfig) -> dict[str, Any]:
        role = {"batch_size": config.batch_size, "max_concurrency": config.max_concurrency}
        if self.sends_dimensions and config.dimensions:
            role["dimensions"] = config.dimensions
        return role

    @property
    def endpoint(self) -> str | None:
        """The API root lite-model sends to (its resolved target)."""
        return self._model.target(ROLE).base_url

    def _input_type(self, role: Role) -> Role | None:
        """The role sent as lite-model's `input_type`: none by default (the prefixes carry it);
        the APIs that take one say so (Voyage S20, Cohere and Gemini S23)."""
        return None

    def _embed(self, texts: list[str], role: Role) -> list[list[float]]:
        try:
            kind = self._input_type(role)
            vectors = self._model.embed(ROLE, texts, input_type=kind).vectors
        except lite_model.ModelError as exc:
            raise EmbeddingError(str(exc)) from exc
        self._check_dim(vectors)
        return vectors


class OllamaEmbedder(_LiteModelEmbedder):
    """Ollama (local, free), through lite-model at its OpenAI-compatible `/v1/embeddings` (its
    profile's default endpoint): the same vectors as the `/api/embed` lite-rag called before S19
    (tests/integration/test_ollama.py). `dimensions` is not sent: it was not before, and it is
    part of the index identity."""

    provider = "ollama"
    sends_dimensions = False

    def _backend(self, config: EmbeddingConfig) -> dict[str, Any]:
        where: dict[str, Any] = {"type": "ollama"}
        if config.base_url:  # lite-rag's setting is the server's root
            where["base_url"] = f"{config.base_url.rstrip('/')}/v1"
        return where


class OpenAICompatEmbedder(_LiteModelEmbedder):
    """`POST {base_url}/embeddings` — OpenAI, and the many services that copy its API (DashScope,
    SiliconFlow, Jina, vLLM, TEI…), through lite-model. Many of those need no key: none unless
    `api_key_env` is set."""

    provider = "openai"

    def _backend(self, config: EmbeddingConfig) -> dict[str, Any]:
        where = {"type": "openai", "base_url": config.base_url or "https://api.openai.com/v1"}
        if config.api_key_env:
            return where | {"api_key_env": config.api_key_env}
        return where | {"keyless": True}


def vendor_backend(kind: str, base_url: str | None, api_key_env: str | None) -> dict[str, Any]:
    """lite-model's backend entry for a hosted vendor type (voyage S20, cohere / gemini S23):
    the type's own endpoint and key variable unless configured."""
    where: dict[str, Any] = {"type": kind}
    if base_url:
        where["base_url"] = base_url.rstrip("/")
    if api_key_env:
        where["api_key_env"] = api_key_env
    return where


class VoyageEmbedder(_LiteModelEmbedder):
    """Voyage AI `/v1/embeddings` (paid), through lite-model's `voyage` type (0.7.0, S20): the
    call's role goes as `input_type`, `dimensions` as `output_dimension`. Voyage prepends its own
    retrieval prompt for that role, so no prefix is added here. The vectors come back as base64
    float32 (the SDK's default; Voyage takes no "float"), the same numbers as its JSON floats."""

    provider = "voyage"

    def _input_type(self, role: Role) -> Role | None:
        return role

    def _backend(self, config: EmbeddingConfig) -> dict[str, Any]:
        return vendor_backend("voyage", config.base_url, config.api_key_env)

    def _role(self, config: EmbeddingConfig) -> dict[str, Any]:
        # Voyage takes at most 1000 texts per request
        return super()._role(config) | {"batch_size": min(config.batch_size, 1000)}


class CohereEmbedder(_LiteModelEmbedder):
    """Cohere `POST /v2/embed` (paid), through lite-model's `cohere` type (0.8.0, S23): the role
    as `input_type` (search_document / search_query), float vectors, `output_dimension`; at most
    96 texts per call."""

    provider = "cohere"

    def _input_type(self, role: Role) -> Role | None:
        return role

    def _backend(self, config: EmbeddingConfig) -> dict[str, Any]:
        return vendor_backend("cohere", config.base_url, config.api_key_env)

    def _role(self, config: EmbeddingConfig) -> dict[str, Any]:
        return super()._role(config) | {"batch_size": min(config.batch_size, 96)}


class GeminiEmbedder(_LiteModelEmbedder):
    """Gemini `POST /v1beta/models/{model}:batchEmbedContents` (paid, free tier), through
    lite-model's `gemini` type (0.8.0, S23). One Content per text. The role is a text prefix
    where the model profile gives one (gemini-embedding-2, which has no task type) and a
    `taskType` otherwise (gemini-embedding-001 and other models): never neither."""

    provider = "gemini"

    def _input_type(self, role: Role) -> Role | None:
        return None if self.prefixes.query or self.prefixes.document else role

    def _backend(self, config: EmbeddingConfig) -> dict[str, Any]:
        return vendor_backend("gemini", config.base_url, config.api_key_env)

    def _role(self, config: EmbeddingConfig) -> dict[str, Any]:
        return super()._role(config) | {"batch_size": min(config.batch_size, 100)}


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
