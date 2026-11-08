from collections.abc import Iterator

import psycopg
import pytest

from lite_rag.db import migrate
from lite_rag.stores.pg import PgStore
from tests.conftest import TEST_SCHEMA


@pytest.fixture
def store(pg: psycopg.Connection, pg_url: str) -> Iterator[PgStore]:
    """A PgStore on a freshly migrated `lite_rag_test` schema."""
    migrate(pg, TEST_SCHEMA)
    s = PgStore(pg_url, TEST_SCHEMA)
    yield s
    s.close()
