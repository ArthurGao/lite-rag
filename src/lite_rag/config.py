"""Configuration (design §9): environment variables with the `LITERAG_` prefix, optionally on
top of a YAML file. Nested keys use `__`, e.g. `LITERAG_CHUNKING__SIZE=600`.

Precedence, highest first: explicit arguments, environment, `.env`, the YAML file, defaults.
"""

import re
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from lite_rag.loaders import PdfLoader

OLLAMA_URL = "http://127.0.0.1:11434"  # the local Ollama server (embeddings: /api, LLM: /v1)
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


Mode = Literal["vector", "bm25", "hybrid"]


class StoreConfig(BaseModel):
    backend: Literal["postgres"] = "postgres"
    # Where BM25 is scored: "sql" (fast, in PostgreSQL) or "python" (keyword.py, readable).
    # Same formula and statistics; the tests check they agree.
    bm25_impl: Literal["sql", "python"] = "sql"
    schema_name: str = Field(default="lite_rag", alias="schema")

    @field_validator("schema_name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not IDENTIFIER.match(value):
            raise ValueError(f"schema must be a lowercase SQL identifier, got {value!r}")
        return value


class EmbeddingConfig(BaseModel):
    """One embedding model per knowledge base (design §3.2)."""

    provider: Literal[
        "ollama", "openai", "fastembed", "voyage", "cohere", "gemini", "sentence_transformers"
    ] = "ollama"
    model: str = "bge-m3:567m"
    # The API root; None = the provider's own (its embedder's `default_url`). openai: including
    # /v1, so any OpenAI-compatible service works (DashScope, SiliconFlow, TEI…).
    base_url: str | None = None
    # The env var holding the key; None = the provider's convention (VOYAGE_API_KEY, CO_API_KEY,
    # GEMINI_API_KEY; openai-compatible: none, many local servers need no key).
    api_key_env: str | None = None
    dimensions: int | None = Field(default=None, gt=0)  # truncate, when the model supports it
    batch_size: int = Field(default=32, gt=0)
    max_concurrency: int = Field(default=4, gt=0)
    max_retries: int = Field(default=4, ge=0)
    # None = what the model needs (model_profiles.py: e5, nomic, bge, qwen3…); "" = no prefix.
    query_prefix: str | None = None
    document_prefix: str | None = None
    cache: bool = True  # reuse vectors by (model, role, prefix, text) hash
    # Hugging Face tokenizer that sizes chunks in this model's tokens. None = the model profile's
    # (model_profiles.py); "" = estimate.
    tokenizer: str | None = None
    timeout_s: float = Field(default=60.0, gt=0)
    cache_dir: str | None = None  # fastembed / sentence-transformers model cache


class ChunkingConfig(BaseModel):
    strategy: Literal[
        "fixed", "recursive", "sentence", "paragraph", "markdown", "parent_child", "semantic"
    ] = "recursive"
    unit: Literal["tokens", "chars"] = "tokens"
    size: int = Field(default=800, gt=0)
    overlap: int = Field(default=120, ge=0)
    min_size: int = Field(default=50, ge=0)
    # parent_child: `size` / `overlap` cut the indexed children; parents (what a hit returns) are
    # cut at headings, then recursively, up to `parent_size` units without overlap (default: four
    # times `size`).
    parent_size: int | None = Field(default=None, gt=0)
    # semantic: cut between adjacent sentences whose embedding distance is above this percentile
    # of all adjacent distances in the document (higher: fewer, larger chunks).
    semantic_threshold: float = Field(default=95, gt=0, lt=100)
    # semantic through chonkie (extra `chonkie`), for comparison: its own method, whose
    # threshold is a cosine similarity (cut below it), not a percentile.
    semantic_impl: Literal["own", "chonkie"] = "own"
    chonkie_threshold: float = Field(default=0.8, gt=0, lt=1)
    contextual: "ContextualConfig" = Field(default_factory=lambda: ContextualConfig())

    @model_validator(mode="after")
    def _overlap_below_size(self) -> "ChunkingConfig":
        if self.overlap >= self.size:
            raise ValueError(f"chunking.overlap ({self.overlap}) must be below size ({self.size})")
        if self.parent_size is not None and self.parent_size <= self.size:
            raise ValueError(
                f"chunking.parent_size ({self.parent_size}) must be above size ({self.size})"
            )
        return self

    def _semantic_identity(self) -> dict[str, Any]:
        if self.semantic_impl == "chonkie":
            return {"semantic_impl": "chonkie", "chonkie_threshold": self.chonkie_threshold}
        return {"semantic_threshold": self.semantic_threshold}  # own: as before chonkie existed

    @property
    def effective_parent_size(self) -> int:
        return self.parent_size or 4 * self.size

    def identity(self) -> dict[str, Any]:
        """What changes the chunks this config produces — recorded on a kb and compared when it
        is opened. A new field joins only when it changes output (add it here deliberately)."""
        return {
            "strategy": self.strategy,
            "unit": self.unit,
            "size": self.size,
            "overlap": self.overlap,
            "min_size": self.min_size,
            # Only where it applies: kbs built before it existed keep their identity.
            **(
                {"parent_size": self.effective_parent_size}
                if self.strategy == "parent_child"
                else {}
            ),
            **(self._semantic_identity() if self.strategy == "semantic" else {}),
            **({"contextual": self.contextual.identity()} if self.contextual.enabled else {}),
        }


class ContextualConfig(BaseModel):
    """Contextual chunk headers (design §3.1): one LLM call per chunk at ingest. Its own `llm`
    section (default: local Ollama), so answers and contexts can use different models."""

    enabled: bool = False
    llm: "LLMConfig" = Field(default_factory=lambda: LLMConfig())
    max_doc_tokens: int = Field(default=6000, gt=0)  # longer documents: a window per chunk

    def identity(self) -> dict[str, Any]:
        from lite_rag.contextual import prompt_digest  # imported late: it imports this module

        return {
            "llm": f"{self.llm.provider}/{self.llm.model}",
            "max_doc_tokens": self.max_doc_tokens,
            "prompt": prompt_digest(),
        }


class LoaderConfig(BaseModel):
    # pypdf: plain text per page, fast, no models. docling: layout models turn the PDF into
    # Markdown (headings kept), seconds per document; needs the `docling` extra.
    pdf: PdfLoader = "pypdf"


class BM25Config(BaseModel):
    k1: float = Field(default=1.5, ge=0)
    b: float = Field(default=0.75, ge=0, le=1)


class RetrievalConfig(BaseModel):
    """Design §5."""

    mode: Mode = "hybrid"
    top_k: int = Field(default=5, gt=0, le=200)
    candidate_factor: int = Field(default=4, gt=0, le=50)  # each path returns top_k * this
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)
    vector_weight: float = Field(default=0.5, ge=0, le=1)  # weighted fusion; bm25 gets the rest
    bm25: BM25Config = BM25Config()


class RerankConfig(BaseModel):
    """Design §5: a cross-encoder re-scores the fused candidates (off by default)."""

    enabled: bool = False
    provider: Literal["fastembed"] = "fastembed"
    # Apache-2.0, multilingual (bge-m3 family), int8 ONNX ~0.6 GB. Not jina-v2 (CC-BY-NC).
    model: str = "BAAI/bge-reranker-v2-m3-int8"
    top_n: int = Field(default=20, gt=0, le=200)  # candidates re-scored
    cache_dir: str | None = None


class LLMConfig(BaseModel):
    """Design §6. `ollama` and `openai` share the OpenAI-compatible client (Ollama serves /v1,
    so do llm-gateway, DeepSeek, DashScope…); `anthropic` uses Anthropic's SDK (extra). What
    each provider accepts is the `PROVIDERS` table in llm.py; a setting a provider cannot honour
    is refused here rather than silently dropped."""

    provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    model: str = "qwen3:8b"
    base_url: str | None = None  # None: the provider's own (Ollama: local /v1)
    api_key_env: str | None = None  # None: the provider's usual variable, e.g. OPENAI_API_KEY
    temperature: float | None = None  # None: the provider profile's default (Ollama: 0)
    max_output_tokens: int = Field(default=1024, gt=0)
    # The request field carrying max_output_tokens, when an OpenAI-compatible endpoint wants the
    # other one than its provider's default (`max_tokens` for ollama, `max_completion_tokens` for
    # openai).
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] | None = None
    # "none" turns qwen3's thinking off on Ollama (7 vs ~350 tokens for a one-line answer,
    # measured 2026-11-08). None: not sent.
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    timeout_s: float = Field(default=120.0, gt=0)
    connect_timeout_s: float = Field(default=5.0, gt=0)  # a stopped server fails fast
    max_retries: int = Field(default=1, ge=0)  # the same model only; then LLMError

    @model_validator(mode="after")
    def _supported_by_provider(self) -> "LLMConfig":
        if self.provider == "anthropic":
            unsupported = {
                "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort,
                "max_tokens_param": self.max_tokens_param,
            }
            given = [name for name, value in unsupported.items() if value is not None]
            if given:
                raise ValueError(f"llm.{', llm.'.join(given)}: not supported by anthropic")
        return self


ChunkingConfig.model_rebuild()
ContextualConfig.model_rebuild()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LITERAG_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str | None = None
    # Required by the engine: every knowledge base lives inside one, e.g. "tenderted/tenant_42".
    namespace: str | None = None
    store: StoreConfig = StoreConfig()
    loader: LoaderConfig = LoaderConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    rerank: RerankConfig = RerankConfig()
    llm: LLMConfig = LLMConfig()

    yaml_path: ClassVar[Path | None] = None  # set by `load_settings` on a subclass
    yaml_first: ClassVar[bool] = False  # the file outranks the environment (evaluation)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [env_settings, dotenv_settings]
        yaml_path = getattr(settings_cls, "yaml_path", None)
        if yaml_path is not None:
            yaml = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path)
            first = getattr(settings_cls, "yaml_first", False)
            sources = [yaml, *sources] if first else [*sources, yaml]
        return (init_settings, *sources)


def load_settings(
    config_file: str | Path | None = None, *, file_wins: bool = False, **overrides: Any
) -> Settings:
    """Settings from the environment, on top of `config_file` (YAML) when given. With
    `file_wins`, the file outranks the environment and `.env` — so evaluation configs measure
    what they say, whatever a developer's `.env` contains."""
    if config_file is None:
        return Settings(**overrides)
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with_file = type("FileSettings", (Settings,), {"yaml_path": path, "yaml_first": file_wins})
    settings: Settings = with_file(**overrides)
    return settings
