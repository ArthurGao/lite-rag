from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lite_rag.config import LLMConfig, load_settings
from lite_rag.llm import build_model
from lite_rag.stores.factory import evaluation_store, location


def test_defaults_are_the_local_free_stack() -> None:
    s = load_settings()
    assert (s.store.backend, s.store.schema_name) == ("postgres", "lite_rag")
    assert (s.embedding.provider, s.embedding.model) == ("ollama", "bge-m3:567m")
    assert (s.chunking.strategy, s.chunking.unit, s.chunking.size) == ("recursive", "tokens", 800)
    assert s.llm.role_id("answer") == "ollama/qwen3:8b"


def test_yaml_file_sets_values_and_the_environment_overrides_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "rag.yaml"
    config.write_text(
        "store: {schema: my_rag}\nchunking: {strategy: markdown, size: 400, overlap: 40}\n"
    )
    monkeypatch.setenv("LITERAG_CHUNKING__SIZE", "600")

    s = load_settings(config)

    assert s.store.schema_name == "my_rag"
    assert s.chunking.strategy == "markdown"  # from the file
    assert s.chunking.size == 600  # the environment wins over the file
    assert s.chunking.overlap == 40


def test_dotenv_in_the_working_directory_is_read(tmp_path: Path) -> None:
    (Path.cwd() / ".env").write_text("LITERAG_DATABASE_URL=postgresql://u@h/db\n")
    assert load_settings().database_url == "postgresql://u@h/db"


def test_a_missing_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_settings(tmp_path / "nope.yaml")


@pytest.mark.parametrize("schema", ["Bad", "x;drop", "1abc", ""])
def test_the_schema_must_be_a_plain_identifier(schema: str) -> None:
    with pytest.raises(ValidationError, match="lowercase SQL identifier"):
        load_settings(store={"schema": schema})


def test_overlap_must_be_below_the_chunk_size() -> None:
    with pytest.raises(ValidationError, match="must be below size"):
        load_settings(chunking={"size": 100, "overlap": 100})


def test_with_file_wins_the_file_outranks_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "eval.yaml"
    config.write_text("chunking: {strategy: fixed, size: 120, overlap: 20}\n")
    monkeypatch.setenv("LITERAG_CHUNKING__STRATEGY", "recursive")
    monkeypatch.setenv("LITERAG_CHUNKING__MIN_SIZE", "7")  # not in the file: env still applies
    assert load_settings(config).chunking.strategy == "recursive"  # normal: env wins
    s = load_settings(config, file_wins=True)
    assert (s.chunking.strategy, s.chunking.size, s.chunking.min_size) == ("fixed", 120, 7)


def test_parent_child_needs_parents_larger_than_children() -> None:
    default = load_settings(chunking={"strategy": "parent_child", "size": 200}).chunking
    assert default.identity(LLMConfig())["parent_size"] == 800  # four times size
    with pytest.raises(ValidationError, match="parent_size"):
        load_settings(chunking={"strategy": "parent_child", "size": 200, "parent_size": 200})


def test_parent_size_joins_the_identity_only_for_parent_child() -> None:
    plain = load_settings(chunking={"parent_size": 999}).chunking.identity(LLMConfig())
    assert "parent_size" not in plain  # existing kbs keep the identity they were built with
    pc = load_settings(chunking={"strategy": "parent_child", "size": 200, "parent_size": 900})
    assert pc.chunking.identity(pc.llm)["parent_size"] == 900


def test_semantic_threshold_joins_the_identity_only_for_semantic() -> None:
    assert "semantic_threshold" not in load_settings().chunking.identity(LLMConfig())
    s = load_settings(chunking={"strategy": "semantic", "semantic_threshold": 90})
    assert s.chunking.identity(s.llm)["semantic_threshold"] == 90
    with pytest.raises(ValidationError):
        load_settings(chunking={"strategy": "semantic", "semantic_threshold": 100})


def test_the_chroma_backend_scores_bm25_in_python_by_default() -> None:
    s = load_settings(store={"backend": "chroma"})
    assert (s.store.bm25_impl, s.store.data_dir) == ("python", Path("lite_rag_data"))
    assert load_settings().store.bm25_impl == "sql"  # postgres keeps its default


def test_the_chroma_backend_refuses_sql_bm25() -> None:
    with pytest.raises(ValidationError, match="needs the postgres backend"):
        load_settings(store={"backend": "chroma", "bm25_impl": "sql"})


def test_the_backend_and_data_dir_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITERAG_STORE__BACKEND", "chroma")
    monkeypatch.setenv("LITERAG_STORE__DATA_DIR", "/srv/rag")
    s = load_settings()
    assert (s.store.backend, s.store.bm25_impl, s.store.data_dir) == (
        "chroma",
        "python",
        Path("/srv/rag"),
    )


def test_evaluation_kbs_live_apart_from_application_data() -> None:
    pg = evaluation_store(load_settings().store, "ev")
    chroma = load_settings(store={"backend": "chroma", "data_dir": "/srv/rag"}).store
    assert (pg.schema_name, pg.data_dir) == ("ev", Path("lite_rag_data"))
    assert evaluation_store(chroma, "ev").data_dir == Path("/srv/rag/ev")


def test_a_store_location_names_only_what_decides_it_and_masks_the_password() -> None:
    pg = load_settings(store={"data_dir": "elsewhere"}).store  # data_dir means nothing to pg
    url = "postgresql://u:secret@h/db"
    assert location(pg, url) == location(load_settings().store, url)
    assert "secret" not in location(pg, url) and "schema lite_rag" in location(pg, url)
    chroma = load_settings(store={"backend": "chroma", "data_dir": "/srv/rag"}).store
    assert location(chroma, url) == location(chroma, None) == "chroma /srv/rag"


def test_graph_is_off_by_default_and_hybrid_keeps_two_paths() -> None:
    s = load_settings()
    assert s.graph.enabled is False
    assert s.retrieval.hybrid_paths == ["vector", "bm25"]
    # extraction on local qwen3 without thinking: the reply is JSON, not a monologue
    assert s.llm.role_id("graph") == "ollama/qwen3:8b"
    assert s.llm.target_of("graph").reasoning_effort == "none"


def test_the_graph_needs_the_postgres_backend() -> None:
    with pytest.raises(ValidationError, match="graph needs the postgres backend"):
        load_settings(store={"backend": "chroma"}, graph={"enabled": True})


def test_a_graph_path_or_mode_needs_the_graph_enabled() -> None:
    with pytest.raises(ValidationError, match=r"graph\.enabled"):
        load_settings(retrieval={"hybrid_paths": ["vector", "graph"]})
    with pytest.raises(ValidationError, match=r"graph\.enabled"):
        load_settings(retrieval={"mode": "graph"})
    s = load_settings(
        retrieval={"mode": "graph", "hybrid_paths": ["bm25", "graph"]}, graph={"enabled": True}
    )
    assert s.retrieval.hybrid_paths == ["bm25", "graph"]


def test_hybrid_fuses_at_least_two_distinct_paths() -> None:
    for paths in (["vector"], ["vector", "vector"]):
        with pytest.raises(ValidationError, match="at least two"):
            load_settings(retrieval={"hybrid_paths": paths})


def test_weighted_fusion_takes_vector_and_bm25_only() -> None:
    with pytest.raises(ValidationError, match="weighted fusion"):
        load_settings(
            retrieval={"fusion": "weighted", "hybrid_paths": ["vector", "bm25", "graph"]},
            graph={"enabled": True},
        )


def test_the_graph_identity_names_the_extraction_model_and_prompt() -> None:
    from lite_rag.graph import prompt_digest

    s = load_settings(graph={"enabled": True}, llm={"roles": {"graph": {"model": "qwen3:14b"}}})
    assert s.graph.identity(s.llm) == {"llm": "ollama/qwen3:14b", "prompt": prompt_digest()}


def test_moving_answers_leaves_the_index_roles_on_their_own_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # S15: before lite-model, `llm` configured answers only; contexts and graphs had their own
    # sections on the local model. Moving answers must not move them (index identity, ingest cost).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    s = load_settings(llm={"backend": "anthropic", "model": "claude-x", "max_output_tokens": 700})
    assert s.llm.role_id("answer") == "anthropic/claude-x"
    assert s.llm.role_id("contextual") == s.llm.role_id("graph") == "ollama/qwen3:8b"
    graph = s.llm.target_of("graph")
    assert (graph.reasoning_effort, graph.max_output_tokens) == ("none", 2048)
    assert s.llm.target_of("contextual").max_output_tokens == 1024
    build_model(s.llm)  # a valid config: nothing anthropic cannot take lands on it
    # a role still moves when told to, keeping its other defaults
    moved = load_settings(llm={"roles": {"graph": {"model": "qwen3:14b"}}})
    assert moved.llm.role_id("graph") == "ollama/qwen3:14b"
    assert moved.llm.target_of("graph").reasoning_effort == "none"


@pytest.mark.parametrize(
    ("given", "moved_to"),
    [
        ({"graph": {"enabled": True, "llm": {"model": "qwen3:14b"}}}, "llm.roles.graph"),
        ({"chunking": {"contextual": {"enabled": True, "llm": {"model": "m"}}}},
         "llm.roles.contextual"),
        ({"llm": {"provider": "openai"}}, "llm.backend"),
    ],
)  # fmt: skip
def test_the_llm_settings_from_before_s15_are_refused_with_where_they_went(
    given: dict[str, Any], moved_to: str
) -> None:
    # never silently ignored: a graph built by the default model instead of the configured one
    with pytest.raises(ValidationError, match=moved_to.replace(".", r"\.")):
        load_settings(**given)


def test_top_level_call_settings_tune_answers_only() -> None:
    # Codex S15: before lite-model `llm` configured answers only; a top-level reasoning effort or
    # temperature reaching contexts and graphs would change what new chunks get under an index
    # identity that does not change.
    s = load_settings(
        llm={"reasoning_effort": "none", "temperature": 0.3, "max_output_tokens": 300}
    )
    answer, contextual, graph = (s.llm.target_of(r) for r in ("answer", "contextual", "graph"))
    assert (answer.reasoning_effort, answer.temperature, answer.max_output_tokens) == (
        "none", 0.3, 300
    )  # fmt: skip
    assert (contextual.reasoning_effort, contextual.temperature, contextual.max_output_tokens) == (
        None, 0.0, 1024
    )  # fmt: skip
    assert (graph.reasoning_effort, graph.temperature, graph.max_output_tokens) == (
        "none", 0.0, 2048
    )  # fmt: skip
    # the answer role's own setting wins over the top level
    both = load_settings(llm={"temperature": 0.3, "roles": {"answer": {"temperature": 0.7}}})
    assert both.llm.target_of("answer").temperature == 0.7
