"""PostgreSQL access and migrations (design §4).

Migrations are the `NNNN_*.sql` files in `migrations/`, applied in order, each in its own
transaction, recorded in `<schema>.schema_migrations`. They run with
`search_path = <schema>, public`, so one set of files serves any schema name.
"""

from importlib import resources

import psycopg
from psycopg import sql

from lite_rag.errors import StoreError

MIGRATIONS = "lite_rag.migrations"


def connect(database_url: str | None) -> psycopg.Connection:
    if not database_url:
        raise StoreError("no database configured: set LITERAG_DATABASE_URL")
    try:
        return psycopg.connect(database_url)
    except psycopg.OperationalError as exc:
        raise StoreError(f"cannot connect to PostgreSQL: {exc}") from exc


def _migration_files() -> list[tuple[str, str]]:
    files = resources.files(MIGRATIONS)
    names = sorted(f.name for f in files.iterdir() if f.name.endswith(".sql"))
    return [(name.removesuffix(".sql"), files.joinpath(name).read_text("utf-8")) for name in names]


def use_schema(conn: psycopg.Connection, schema: str) -> None:
    """Point the current transaction at `schema` (`vector` stays reachable through public)."""
    conn.execute(sql.SQL("SET LOCAL search_path TO {}, public").format(sql.Identifier(schema)))


def migrate(conn: psycopg.Connection, schema: str) -> list[str]:
    """Apply pending migrations to `schema`; returns the versions applied (empty: up to date)."""
    with conn.transaction():
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.schema_migrations "
                "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            ).format(sql.Identifier(schema))
        )
    applied: list[str] = []
    for version, script in _migration_files():
        with conn.transaction():
            use_schema(conn, schema)
            # Serialise concurrent migrators on this schema; the check below then sees their work.
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"lite_rag:{schema}",))
            done = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s", (version,)
            ).fetchone()
            if done:
                continue
            conn.execute(script.encode())  # trusted SQL file shipped in the package
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied.append(version)
    return applied


def check_vector(conn: psycopg.Connection) -> None:
    """Run a real vector operation: an extension can look installed while its library is missing
    (KB rag-production-guide §2), so presence alone proves nothing."""
    try:
        conn.execute("SELECT '[1,2]'::vector <=> '[1,2]'").fetchone()
    except psycopg.Error as exc:
        raise StoreError(f"pgvector is not usable: {exc}") from exc
