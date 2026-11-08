"""Configuration (design §9): environment variables with the `LITERAG_` prefix, optionally on
top of a YAML file. Nested keys use `__`, e.g. `LITERAG_CHUNKING__SIZE=600`.

Precedence, highest first: explicit arguments, environment, `.env`, the YAML file, defaults.
"""

import re
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class StoreConfig(BaseModel):
    backend: Literal["postgres"] = "postgres"
    schema_name: str = Field(default="lite_rag", alias="schema")

    @field_validator("schema_name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not IDENTIFIER.match(value):
            raise ValueError(f"schema must be a lowercase SQL identifier, got {value!r}")
        return value


class EmbeddingConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    model: str = "bge-m3:567m"
    base_url: str = "http://127.0.0.1:11434"
    batch_size: int = Field(default=32, gt=0)
    max_concurrency: int = Field(default=4, gt=0)
    query_prefix: str = ""
    timeout_s: float = Field(default=60.0, gt=0)


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "sentence", "paragraph", "markdown"] = "recursive"
    unit: Literal["tokens", "chars"] = "tokens"
    size: int = Field(default=800, gt=0)
    overlap: int = Field(default=120, ge=0)
    min_size: int = Field(default=50, ge=0)


class LLMConfig(BaseModel):
    provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    model: str = "qwen3:8b"
    base_url: str | None = "http://127.0.0.1:11434/v1"
    api_key_env: str | None = None
    temperature: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LITERAG_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str | None = None
    store: StoreConfig = StoreConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
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

    @field_validator("chunking")
    @classmethod
    def _overlap_below_size(cls, value: ChunkingConfig) -> ChunkingConfig:
        if value.overlap >= value.size:
            raise ValueError(
                f"chunking.overlap ({value.overlap}) must be below size ({value.size})"
            )
        return value


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
