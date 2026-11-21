"""Rerankers (design §5): a cross-encoder reads the query and each candidate together and scores
the pair — slower than the first-stage retrieval, so it only re-scores the top candidates."""

from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol

import lite_model
from lite_model import ModelsConfig, RolePolicy, RoleSpec

from lite_rag.config import RerankConfig
from lite_rag.embedding import fastembed_models, voyage_backend
from lite_rag.errors import ConfigError, LiteRagError
from lite_rag.extras import require_extra
from lite_rag.llm import build_model
from lite_rag.types import model_id

if TYPE_CHECKING:
    import httpx2

ROLE = "rerank"  # the lite-model role of a remote reranker (S20)
ROLES = RoleSpec({ROLE: RolePolicy(kind="rerank")})  # never routable


class RerankError(LiteRagError):
    """A remote reranker's call failed (after retries of the same model)."""

    status = 502


class Reranker(Protocol):
    @property
    def model_id(self) -> str: ...

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text, in the same order (higher is more relevant)."""
        ...


class FastEmbedReranker:
    """Local ONNX cross-encoder (extra `[fastembed]`); the model downloads once."""

    def __init__(self, config: RerankConfig) -> None:
        cross_encoder = require_extra("fastembed.rerank.cross_encoder", "fastembed")
        if config.model not in fastembed_models(cross_encoder.TextCrossEncoder):
            raise ConfigError(f"fastembed has no reranker {config.model!r}")
        self._config = config
        self.model_id = model_id("fastembed", config.model)

    @cached_property
    def _impl(self) -> Any:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name=self._config.model, cache_dir=self._config.cache_dir)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._impl.rerank(query, texts)]


class VoyageReranker:
    """Voyage AI `/v1/rerank` (paid) through lite-model's `rerank()` (0.7.0, S20): every text
    scored, in input order. A failed call raises `RerankError`, never a partial ranking."""

    def __init__(self, config: RerankConfig, client: "httpx2.Client | None" = None) -> None:
        models = ModelsConfig(
            backend="voyage",
            model=config.model,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            backends={"voyage": voyage_backend(config.base_url, config.api_key_env)},
        )
        self._model = build_model(models, client, roles=ROLES)  # a missing key: ConfigError
        self.model_id = model_id("voyage", config.model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        try:
            scores = self._model.rerank(ROLE, query, texts).scores
        except lite_model.ModelError as exc:
            raise RerankError(str(exc)) from exc
        if any(s is None for s in scores):  # no top_k: lite-model checked every text is scored
            raise RerankError(f"{self.model_id}: not every text was scored")
        return [float(s) for s in scores if s is not None]


def build_reranker(config: RerankConfig) -> Reranker:
    if config.provider == "voyage":
        return VoyageReranker(config)
    return FastEmbedReranker(config)
