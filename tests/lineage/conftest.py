"""Ledger tests run once per dialect: PostgreSQL in the test schema (its `search_path`, as a
host's connection would have it) and SQLite in a temporary file. Hosts own transactions, so the
fixtures do too: PostgreSQL migrates in one, the `ledger` works inside one."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from lite_rag.lineage import Ledger, ledger_for, migrate, pending
from tests.conftest import TEST_SCHEMA

Conn = psycopg.Connection | sqlite3.Connection


@pytest.fixture
def pg_conn(pg: psycopg.Connection, pg_url: str) -> Iterator[psycopg.Connection]:
    """A connection to the empty test schema, outside any transaction."""
    pg.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(TEST_SCHEMA)))
    options = f"-c search_path={TEST_SCHEMA}"
    with psycopg.connect(pg_url, autocommit=True, options=options) as conn:
        yield conn


@pytest.fixture
def sqlite_conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(tmp_path / "ledger.sqlite", isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    conn.close()


@pytest.fixture(params=["pg", "sqlite"])
def empty_conn(request: pytest.FixtureRequest) -> Conn:
    conn: Conn = request.getfixturevalue(f"{request.param}_conn")
    return conn


@contextmanager
def transaction(conn: Conn) -> Iterator[None]:
    """A transaction the way a host would hold one, rolled back at the end for SQLite (PG
    commits: its schema is dropped after the test)."""
    if isinstance(conn, sqlite3.Connection):
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        finally:
            conn.execute("ROLLBACK")
    else:
        with conn.transaction():
            yield


def migrated(conn: Conn) -> list[str]:
    if isinstance(conn, sqlite3.Connection):  # applies each file in its own transaction
        return migrate(conn)
    with conn.transaction():
        return migrate(conn)


def pending_of(conn: Conn) -> list[str]:
    if isinstance(conn, sqlite3.Connection):
        return pending(conn)
    with conn.transaction():
        return pending(conn)


@pytest.fixture
def ledger_conn(empty_conn: Conn) -> Conn:
    migrated(empty_conn)
    return empty_conn


class Clock:
    """A clock that moves one second per reading, from a fixed start."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def ledger(ledger_conn: Conn, clock: Clock) -> Iterator[Ledger]:
    with transaction(ledger_conn):
        yield ledger_for(ledger_conn, clock=clock)
