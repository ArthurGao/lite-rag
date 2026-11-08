from pathlib import Path

import pytest
from pydantic import ValidationError

from lite_rag.config import load_settings


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
