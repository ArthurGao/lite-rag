"""What differs between PostgreSQL (psycopg) and SQLite (sqlite3) for SQL written once with `?`
placeholders: the parameter style, JSON and times. The ledger is built on it; a host can use it
for tables of its own that live next to the ledger's (lite-rag's query traces do)."""

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


class Dialect:
    """SQLite's: JSON as text; times as fixed-width ISO-8601 UTC text, so text order is time
    order. Needs SQLite >= 3.35 (RETURNING) and foreign keys on."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def sql(self, query: str) -> str:
        return query

    def to_json(self, value: Any) -> Any:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def from_json(self, value: Any) -> Any:
        return json.loads(value)

    def to_time(self, value: datetime) -> Any:
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

    def from_time(self, value: Any) -> datetime:
        return datetime.fromisoformat(value)

    def rows(self, query: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        return list(self.conn.execute(self.sql(query), tuple(params)).fetchall())

    def run(self, query: str, params: Sequence[Any] = ()) -> int:
        return int(self.conn.execute(self.sql(query), tuple(params)).rowcount)

    def run_many(self, query: str, params: Sequence[Sequence[Any]]) -> None:
        if params:
            self.conn.executemany(self.sql(query), [tuple(p) for p in params])


class PgDialect(Dialect):
    """psycopg: `%s` parameters (queries must not contain a literal `?`), jsonb, timestamptz."""

    def sql(self, query: str) -> str:
        return query.replace("?", "%s")

    def to_json(self, value: Any) -> Any:
        return Jsonb(value)

    def from_json(self, value: Any) -> Any:
        return value

    def to_time(self, value: datetime) -> Any:
        return value

    def from_time(self, value: Any) -> datetime:
        assert isinstance(value, datetime)
        return value

    def run_many(self, query: str, params: Sequence[Sequence[Any]]) -> None:
        if params:
            with self.conn.cursor() as cur:
                cur.executemany(self.sql(query), [tuple(p) for p in params])


def dialect_for(conn: psycopg.Connection | sqlite3.Connection) -> Dialect:
    return Dialect(conn) if isinstance(conn, sqlite3.Connection) else PgDialect(conn)
