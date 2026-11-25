"""The ledger's own schema migrations, recorded in `lineage_schema_migrations` so they never mix
with the host's. `migrate(conn)` / `pending(conn)` pick the dialect from the connection.
PostgreSQL runs inside the caller's transaction and `search_path` (the host decides the schema
and serialises migrators); SQLite applies each file in its own transaction, since
`executescript` commits whatever is open first (the same pattern as lite-rag's
`stores/chroma.migrate_sqlite`, repeated here because this package imports nothing of lite-rag)."""

import sqlite3
from importlib import resources

import psycopg

PREFIX = "lineage:"
TABLE = "lineage_schema_migrations"


def _files(conn: psycopg.Connection | sqlite3.Connection) -> list[tuple[str, str]]:
    sqlite = isinstance(conn, sqlite3.Connection)
    files = resources.files(f"lite_rag.lineage.migrations{'_sqlite' if sqlite else ''}")
    names = sorted(f.name for f in files.iterdir() if f.name.endswith(".sql"))
    return [(PREFIX + n.removesuffix(".sql"), files.joinpath(n).read_text("utf-8")) for n in names]


def _applied(conn: psycopg.Connection | sqlite3.Connection) -> set[str]:
    if isinstance(conn, sqlite3.Connection):
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (TABLE,)
        ).fetchone()
    else:
        row = conn.execute("SELECT to_regclass(%s)", (TABLE,)).fetchone()
        exists = row if row and row[0] is not None else None
    if exists is None:
        return set()
    return {v for (v,) in conn.execute(f"SELECT version FROM {TABLE}")}


def pending(conn: psycopg.Connection | sqlite3.Connection) -> list[str]:
    """Versions not yet applied (PostgreSQL: in the connection's current schema)."""
    done = _applied(conn)
    return [v for v, _ in _files(conn) if v not in done]


def migrate(conn: psycopg.Connection | sqlite3.Connection) -> list[str]:
    """Apply pending migrations; returns the versions applied."""
    stamp = "timestamptz NOT NULL DEFAULT now()"
    if isinstance(conn, sqlite3.Connection):
        stamp = "text NOT NULL DEFAULT CURRENT_TIMESTAMP"
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} (version text PRIMARY KEY, applied_at {stamp})"
    )
    done = _applied(conn)
    applied = []
    for version, script in _files(conn):
        if version in done:
            continue
        if isinstance(conn, sqlite3.Connection):
            conn.executescript(
                f"BEGIN IMMEDIATE;\n{script}\n"
                f"INSERT INTO {TABLE} (version) VALUES ('{version}');\nCOMMIT;"
            )
        else:
            conn.execute(script.encode())  # trusted SQL shipped in the package
            conn.execute(f"INSERT INTO {TABLE} (version) VALUES (%s)", (version,))
        applied.append(version)
    return applied
