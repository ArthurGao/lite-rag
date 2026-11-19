"""Configuration (design §9): environment variables with the `LITERAG_` prefix, optionally on
top of a YAML file. Nested keys use `__`, e.g. `LITERAG_CHUNKING__SIZE=600`.

Precedence, highest first: explicit arguments, environment, `.env`, the YAML file, defaults.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from lite_model import ModelsConfig, RoleConfig, Target
from lite_model.config import Name, target
from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from lite_rag.loaders import PdfLoader
from lite_rag.types import model_id

OLLAMA_URL = (
    "http://127.0.0.1:11434"  # the local Ollama server's embeddings (/api); LLMs: lite-model
)
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def sql_identifier(name: str, value: str) -> str:
    """`value` when it is a lowercase SQL identifier; the setting `name` names it otherwise."""
    if not IDENTIFIER.match(value):
        raise ValueError(f"{name} must be a lowercase SQL identifier, got {value!r}")
    return value


Mode = Literal["vector", "bm25", "graph", "hybrid"]
RetrievalPath = Literal["vector", "bm25", "graph"]  # one retrieval path; hybrid fuses several


class StoreConfig(BaseModel):
    backend: Literal["postgres", "chroma"] = "postgres"
    # Where BM25 is scored: "sql" (fast, in PostgreSQL) or "python" (keyword.py, readable).
    # Same formula and statistics; the tests check they agree. Chroma: "python" only.
    bm25_impl: Literal["sql", "python"] = "sql"
    schema_name: str = Field(default="lite_rag", alias="schema")  # postgres
    data_dir: Path = Path("lite_rag_data")  # chroma: the SQLite file and the Chroma directory

    @model_validator(mode="after")
    def _bm25_backend(self) -> Self:
        if self.backend == "chroma":
            if "bm25_impl" not in self.model_fields_set:
                self.bm25_impl = "python"  # chroma's default
            elif self.bm25_impl == "sql":
                raise ValueError("store.bm25_impl sql needs the postgres backend (chroma: python)")
        return self

    @field_validator("schema_name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        return sql_identifier("schema", value)


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

    def identity(self, llm: "LLMConfig") -> dict[str, Any]:
        """What changes the chunks this config produces — recorded on a kb and compared when it
        is opened. A new field joins only when it changes output (add it here deliberately).
        `llm`: the contextual role's model is identity when contextual headers are on."""
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
            **({"contextual": self.contextual.identity(llm)} if self.contextual.enabled else {}),
        }


class ContextualConfig(BaseModel):
    """Contextual chunk headers (design §3.1): one LLM call per chunk at ingest, as the
    `contextual` role of the `llm` section (default: the same local model as answers)."""

    enabled: bool = False

    @model_validator(mode="before")
    @classmethod
    def _llm_moved(cls, value: Any) -> Any:
        return _moved_llm(value, "chunking.contextual", "contextual")

    max_doc_tokens: int = Field(default=6000, gt=0)  # longer documents: a window per chunk
    concurrency: int = Field(default=4, gt=0)  # LLM calls at once while ingesting (not identity)

    def identity(self, llm: "LLMConfig") -> dict[str, Any]:
        from lite_rag.contextual import prompt_digest  # imported late: it imports this module

        return {
            "llm": llm.role_id("contextual"),
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
    # The paths `hybrid` fuses; adding "graph" needs graph.enabled (design §5, S11).
    hybrid_paths: list[RetrievalPath] = ["vector", "bm25"]
    top_k: int = Field(default=5, gt=0, le=200)
    candidate_factor: int = Field(default=4, gt=0, le=50)  # each path returns top_k * this
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)
    vector_weight: float = Field(default=0.5, ge=0, le=1)  # weighted fusion; bm25 gets the rest
    bm25: BM25Config = BM25Config()

    @model_validator(mode="after")
    def _paths(self) -> Self:
        paths = self.hybrid_paths
        if len(paths) < 2 or len(set(paths)) != len(paths):
            raise ValueError(
                f"retrieval.hybrid_paths must name at least two distinct paths: {paths}"
            )
        if self.fusion == "weighted" and set(self.hybrid_paths) != {"vector", "bm25"}:
            raise ValueError("weighted fusion takes exactly the vector and bm25 paths")
        return self


class GraphConfig(BaseModel):
    """Graph retrieval, local mode (design §5, S11): at ingest an LLM lists each chunk's entities
    and relations (the `graph` role of the `llm` section, default local qwen3 without thinking);
    at query time the `graph` path walks them. Only the role's model and the prompt are index
    identity; the rest tunes the query and may change freely."""

    enabled: bool = False

    @model_validator(mode="before")
    @classmethod
    def _llm_moved(cls, value: Any) -> Any:
        return _moved_llm(value, "graph", "graph")

    concurrency: int = Field(default=4, gt=0)  # LLM calls at once while ingesting
    hops: int = Field(default=2, ge=0, le=4)  # 2: the answer two documents from the named one
    # An entity in more chunks than this is a hub (a company name, a running theme): the walk
    # neither passes through it nor reaches chunks by it. Tune to the corpus.
    hub_chunks: int = Field(default=5, gt=0)
    max_neighbors: int = Field(default=20, gt=0, le=1000)  # new entities kept per hop

    def identity(self, llm: "LLMConfig") -> dict[str, Any]:
        from lite_rag.graph import prompt_digest  # imported late: it imports this module

        return {"llm": llm.role_id("graph"), "prompt": prompt_digest()}

    def index_identity(self, llm: "LLMConfig") -> dict[str, Any]:
        """What a graph adds to a kb's identity: nothing when off, so kbs built without one keep
        theirs (the engine's kb record and the evaluation's kb names both use this)."""
        return {"graph": self.identity(llm)} if self.enabled else {}


class RerankConfig(BaseModel):
    """Design §5: a cross-encoder re-scores the fused candidates (off by default)."""

    enabled: bool = False
    provider: Literal["fastembed"] = "fastembed"
    # Apache-2.0, multilingual (bge-m3 family), int8 ONNX ~0.6 GB. Not jina-v2 (CC-BY-NC).
    model: str = "BAAI/bge-reranker-v2-m3-int8"
    top_n: int = Field(default=20, gt=0, le=200)  # candidates re-scored
    cache_dir: str | None = None


def _moved_llm(value: Any, section: str, role: str) -> Any:
    """A role's own `llm` section is gone (S15): refused with where it went, never ignored (a
    graph would silently be built by the default model)."""
    if isinstance(value, dict) and "llm" in value:
        raise ValueError(f"{section}.llm moved to llm.roles.{role} (lite-rag S15, design §6)")
    return value


# Call settings a top-level `llm` value gives the answer role only (S15).
_ANSWER_ONLY = ("temperature", "reasoning_effort", "max_output_tokens")

# lite-rag's LLM roles (design §6). `contextual` and `graph` are index identity: never routed.
Role = Literal["answer", "contextual", "graph"]


class LLMConfig(ModelsConfig):
    """Design §6, S15: lite-model's configuration (its design §4: `backend`, `model`, `roles`,
    `backends`, …) with lite-rag's defaults: every role on the local Ollama — answers on qwen3:8b,
    the index roles on qwen3:1.7b without thinking (S17) —, at most 1024 output tokens, a stopped
    server failing fast (5 s), one retry of the same model.

    The top-level `backend` / `model` move answers only; `contextual` and `graph` keep their own
    (graph: 2048 tokens) until `roles` names them, field by field. Backends are named after their
    type, so a role's id (`ollama/qwen3:8b`) is recorded in index identity and on every answer."""

    # The index roles keep their own model, as their own sections did before lite-model: moving
    # answers (the top-level `backend` / `model`) must not move them — their model is index
    # identity, and ingest would start calling it (S15).
    # S17: qwen3:1.7b builds contexts 3.6x and graphs 5.7x faster than 8b; the right chunk drops
    # from rank 1 to 2-3 on 3-4 of 32 questions, but the 8b answers over its indexes do not change
    # (KB global/llm-evals/evals/2026-11-19-lite-small-model-per-role.md). "none" turns qwen3's
    # thinking off on Ollama (7 vs ~350 tokens for a one-line answer, measured 2026-11-08).
    host_roles: ClassVar[Mapping[str, RoleConfig]] = {
        "contextual": RoleConfig(
            backend="ollama", model="qwen3:1.7b", reasoning_effort="none", max_output_tokens=1024
        ),
        "graph": RoleConfig(
            backend="ollama", model="qwen3:1.7b", reasoning_effort="none", max_output_tokens=2048
        ),
    }
    backend: Name = "ollama"
    model: Name = "qwen3:8b"
    max_output_tokens: int | None = Field(default=1024, gt=0)
    connect_timeout_s: float = Field(default=5.0, gt=0)  # a stopped server fails fast

    @model_validator(mode="before")
    @classmethod
    def _before_s15(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        old = sorted({"provider", "base_url", "api_key_env", "max_tokens_param"} & set(value))
        if old:
            raise ValueError(
                f"llm.{', llm.'.join(old)}: moved in lite-rag S15 — llm.backend names a backend, "
                "and its base_url / api_key_env go under llm.backends.<name> (design §6)"
            )
        # The top level tunes answers, as `llm` did before S15: its call settings go to the answer
        # role (whose own wins), so contexts and graphs stay as their index was built.
        moved = {k: value[k] for k in _ANSWER_ONLY if k in value}
        if not moved:
            return value
        roles = dict(value.get("roles") or {})
        answer = roles.get("answer") or {}
        if isinstance(answer, BaseModel):
            answer = answer.model_dump(exclude_unset=True)
        roles["answer"] = {**moved, **answer}
        return {**{k: v for k, v in value.items() if k not in moved}, "roles": roles}

    def target_of(self, role: Role) -> Target:
        """Where the role's calls go, resolved without checks or a call (lite-model's own
        resolution; the one place lite-rag reaches it)."""
        return target(self, role)

    def role_id(self, role: Role) -> str:
        """`backend/model` the role is configured with (index identity)."""
        t = self.target_of(role)
        return model_id(t.backend, t.model)


ChunkingConfig.model_rebuild()
ContextualConfig.model_rebuild()
GraphConfig.model_rebuild()


class CacheConfig(BaseModel):
    """lite-cache's L1 for lite-rag's model calls (S18; lite-cache design §9). Off by default.
    Entries live in `schema` of lite-rag's own PostgreSQL; every cached call is scoped to
    `literag/<namespace>` (lite-cache §0). Not part of a kb's identity: a hit returns what the
    same request returned before."""

    enabled: bool = False
    roles: list[Role] = ["answer", "contextual", "graph"]
    ttl_s: PositiveInt | None = None  # None: until invalidated
    schema_name: str = Field(default="lite_cache", alias="schema")
    # `ask()` appends {ts, scope, question} per question (JSONL) when set: for a later replay
    # evaluation (lite-cache §8). Off by default: questions may be personal.
    question_log: Path | None = None
    # The variable holding the secret for source-block nonces of cached calls (design §6): with
    # random nonces no prompt would repeat; with this key they repeat and stay unpredictable.
    prompt_secret_env: str = "LITERAG_PROMPT_SECRET"

    @field_validator("schema_name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        # lite-cache's own rule (its design §5): a schema of its own, never a shared or system one
        if value in ("public", "information_schema") or value.startswith("pg_"):
            raise ValueError(f"cache.schema {value!r} is shared or a system schema")
        return sql_identifier("cache.schema", value)


class ApiKeyConfig(BaseModel):
    """One API key of the HTTP service (design §7). Only the key's sha256 is configured;
    `literag key create` prints a new key once, with this entry. Access is the intersection of
    `namespaces` and `kbs` (glob patterns, `*` matches across `/`) and `mode`."""

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    namespaces: list[str] = Field(min_length=1)  # e.g. ["steward/*"]; no default: no access
    kbs: list[str] = ["*"]
    mode: Literal["read", "write"] = "read"


class ServerConfig(BaseModel):
    """`literag serve` (design §7). Keys come from YAML or `LITERAG_SERVER__KEYS` (JSON)."""

    host: str = "127.0.0.1"
    port: int = Field(default=8700, gt=0, lt=65536)
    keys: list[ApiKeyConfig] = []  # none: open access, allowed on a loopback address only
    max_upload_mb: float = Field(default=20, gt=0)
    max_body_mb: float = Field(default=10, gt=0)  # any other request body (JSON)
    max_workers: int = Field(default=8, gt=0)  # requests handled at once (threadpool)


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
    graph: GraphConfig = GraphConfig()
    rerank: RerankConfig = RerankConfig()
    llm: LLMConfig = LLMConfig()
    cache: CacheConfig = CacheConfig()
    server: ServerConfig = ServerConfig()

    @model_validator(mode="after")
    def _cache(self) -> Self:
        if self.cache.schema_name == self.store.schema_name:
            raise ValueError(
                "cache.schema must differ from store.schema (lite-cache owns its schema)"
            )
        if self.cache.enabled:
            if self.store.backend != "postgres":
                raise ValueError(
                    "cache needs the postgres backend (lite-cache stores in PostgreSQL)"
                )
            if not self.database_url:
                raise ValueError("cache needs database_url (lite-cache uses lite-rag's PostgreSQL)")
        return self

    @model_validator(mode="after")
    def _graph(self) -> Self:
        if self.graph.enabled and self.store.backend != "postgres":
            raise ValueError("graph needs the postgres backend (design §5, S11)")
        uses_graph = self.retrieval.mode == "graph" or "graph" in self.retrieval.hybrid_paths
        if uses_graph and not self.graph.enabled:
            raise ValueError(
                "retrieval uses the graph path: set graph.enabled (and rebuild the kb)"
            )
        return self

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
