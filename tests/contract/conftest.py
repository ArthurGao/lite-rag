"""The `store` fixture runs every contract test once per backend, opened the way the engine
opens it (`stores/factory.py`): PostgreSQL in the test schema, Chroma + SQLite in a temporary
data directory."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from lite_rag.config import StoreConfig
from lite_rag.stores.base import CachedStore
from lite_rag.stores.factory import migrate_store, open_store
from tests.conftest import TEST_SCHEMA


@pytest.fixture(params=["postgres", "chroma"], ids=["pg", "chroma"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[CachedStore]:
    config = StoreConfig(backend=request.param, schema=TEST_SCHEMA, data_dir=tmp_path / "data")
    url = None
    if request.param == "postgres":
        request.getfixturevalue("pg")  # an empty test schema, dropped afterwards
        url = request.getfixturevalue("pg_url")
    migrate_store(config, url)
    s = open_store(config, url)
    yield s
    s.close()
