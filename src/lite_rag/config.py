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

    provider: Literal["ollama", "openai", "fastembed"] = "ollama"
    model: str = "bge-m3:567m"
    # ollama: the server root; openai: the API root including /v1 (OpenAI, DashScope, SiliconFlow…)
    base_url: str | None = OLLAMA_URL
    api_key_env: str | None = None  # openai-compatible: the env var holding the key
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
    cache_dir: str | None = None  # fastembed model cache


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "sentence", "paragraph", "markdown"] = "recursive"
    unit: Literal["tokens", "chars"] = "tokens"
    size: int = Field(default=800, gt=0)
    overlap: int = Field(default=120, ge=0)
    min_size: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _overlap_below_size(self) -> "ChunkingConfig":
        if self.overlap >= self.size:
            raise ValueError(f"chunking.overlap ({self.overlap}) must be below size ({self.size})")
        return self

    def identity(self) -> dict[str, Any]:
        """What changes the chunks this config produces — recorded on a kb and compared when it
        is opened. A new field joins only when it changes output (add it here deliberately)."""
        return {
            "strategy": self.strategy,
            "unit": self.unit,
            "size": self.size,
            "overlap": self.overlap,
            "min_size": self.min_size,
        }


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
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    rerank: RerankConfig = RerankConfig()
    llm: LLMConfig = LLMConfig()

    yaml_path: ClassVar[Path | None] = None  # set by `load_settings` on a subclass

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        yaml_path = getattr(settings_cls, "yaml_path", None)
        if yaml_path is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        return tuple(sources)


def load_settings(config_file: str | Path | None = None, **overrides: Any) -> Settings:
    """Settings from the environment, on top of `config_file` (YAML) when given."""
    if config_file is None:
        return Settings(**overrides)
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with_file = type("FileSettings", (Settings,), {"yaml_path": path})
    settings: Settings = with_file(**overrides)
    return settings
