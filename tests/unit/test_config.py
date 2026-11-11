from pathlib import Path

import pytest
from pydantic import ValidationError

from lite_rag.config import load_settings
from lite_rag.stores.factory import evaluation_store, location


def test_defaults_are_the_local_free_stack() -> None:
    s = load_settings()
    assert (s.store.backend, s.store.schema_name) == ("postgres", "lite_rag")
    assert (s.embedding.provider, s.embedding.model) == ("ollama", "bge-m3:567m")
    assert (s.chunking.strategy, s.chunking.unit, s.chunking.size) == ("recursive", "tokens", 800)
    assert (s.llm.provider, s.llm.model) == ("ollama", "qwen3:8b")


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
    assert default.identity()["parent_size"] == 800  # four times size
    with pytest.raises(ValidationError, match="parent_size"):
        load_settings(chunking={"strategy": "parent_child", "size": 200, "parent_size": 200})


def test_parent_size_joins_the_identity_only_for_parent_child() -> None:
    plain = load_settings(chunking={"parent_size": 999}).chunking.identity()
    assert "parent_size" not in plain  # existing kbs keep the identity they were built with
    pc = load_settings(chunking={"strategy": "parent_child", "size": 200, "parent_size": 900})
    assert pc.chunking.identity()["parent_size"] == 900


def test_semantic_threshold_joins_the_identity_only_for_semantic() -> None:
    assert "semantic_threshold" not in load_settings().chunking.identity()
    s = load_settings(chunking={"strategy": "semantic", "semantic_threshold": 90})
    assert s.chunking.identity()["semantic_threshold"] == 90
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
    assert (s.graph.llm.provider, s.graph.llm.model) == ("ollama", "qwen3:8b")
    assert s.graph.llm.reasoning_effort == "none"


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

    s = load_settings(graph={"enabled": True, "llm": {"model": "qwen3:14b"}})
    assert s.graph.identity() == {"llm": "ollama/qwen3:14b", "prompt": prompt_digest()}
