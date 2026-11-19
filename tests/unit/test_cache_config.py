"""`cache` settings (S18, lite-cache design §9): off by default, PostgreSQL only."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lite_rag.config import load_settings

URL = "postgresql://u@h/db"


def test_off_by_default_with_every_role() -> None:
    cache = load_settings().cache
    assert cache.enabled is False
    assert cache.roles == ["answer", "contextual", "graph"]
    assert (cache.schema_name, cache.ttl_s, cache.question_log) == ("lite_cache", None, None)


def test_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITERAG_DATABASE_URL", URL)
    monkeypatch.setenv("LITERAG_CACHE__ENABLED", "true")
    monkeypatch.setenv("LITERAG_CACHE__ROLES", '["answer"]')
    monkeypatch.setenv("LITERAG_CACHE__TTL_S", "3600")
    monkeypatch.setenv("LITERAG_CACHE__SCHEMA", "my_cache")
    monkeypatch.setenv("LITERAG_CACHE__QUESTION_LOG", "logs/questions.jsonl")
    cache = load_settings().cache
    assert (cache.enabled, cache.roles, cache.ttl_s) == (True, ["answer"], 3600)
    assert (cache.schema_name, cache.question_log) == ("my_cache", Path("logs/questions.jsonl"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"store": {"backend": "chroma"}, "database_url": URL}, "postgres"),
        ({}, "database_url"),
    ],
    ids=["chroma", "no-database-url"],
)
def test_enabled_needs_postgres(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        load_settings(cache={"enabled": True}, **overrides)


@pytest.mark.parametrize(
    "cache",
    [
        {"roles": ["nope"]},
        {"ttl_s": 0},
        {"schema": "Bad-Name"},
        {"schema": "lite_rag"},  # the store's own schema
        {"schema": "public"},  # lite-cache refuses shared and system schemas
        {"schema": "information_schema"},
        {"schema": "pg_temp"},
    ],
    ids=["role", "ttl", "identifier", "store-schema", "public", "information-schema", "pg"],
)
def test_invalid_cache_settings_are_refused(cache: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        load_settings(cache=cache)
