"""Rerankers (design §5): a cross-encoder reads the query and each candidate together and scores
the pair — slower than the first-stage retrieval, so it only re-scores the top candidates."""

from functools import cached_property
from typing import Any, Protocol

from lite_rag.config import RerankConfig
from lite_rag.embedding import fastembed_models
from lite_rag.errors import ConfigError
from lite_rag.extras import require_extra
from lite_rag.types import model_id


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


def build_reranker(config: RerankConfig) -> Reranker:
    return FastEmbedReranker(config)
