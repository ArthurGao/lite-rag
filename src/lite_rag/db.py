"""PostgreSQL access and migrations (design §4).

Migrations are the `NNNN_*.sql` files in `migrations/`, applied in order, each in its own
transaction, recorded in `<schema>.schema_migrations`. They run with
`search_path = <schema>, public`, so one set of files serves any schema name.
"""

from importlib import resources

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from lite_rag.errors import StoreError

MIGRATIONS = "lite_rag.migrations"


def require_url(database_url: str | None) -> str:
    if not database_url:
        raise StoreError("no database configured: set LITERAG_DATABASE_URL")
    return database_url


def connection_error(exc: psycopg.Error) -> StoreError:
    return StoreError(f"cannot connect to PostgreSQL: {exc}")


def connect(database_url: str | None) -> psycopg.Connection:
    try:
        return psycopg.connect(require_url(database_url))
    except psycopg.OperationalError as exc:
        raise connection_error(exc) from exc


def redact_conninfo(value: str) -> str:
    """The connection string with its password replaced, in any form libpq accepts (URL or
    `key=value`). Something that does not parse is never echoed back."""
    try:
        params = conninfo_to_dict(value)
    except psycopg.ProgrammingError:
        return "<unparseable connection string>"
    if params.get("password"):
        params["password"] = "***"
    return make_conninfo("", **params)


def _migration_files() -> list[tuple[str, str]]:
    files = resources.files(MIGRATIONS)
    names = sorted(f.name for f in files.iterdir() if f.name.endswith(".sql"))
    return [(name.removesuffix(".sql"), files.joinpath(name).read_text("utf-8")) for name in names]


def use_schema(conn: psycopg.Connection, schema: str) -> None:
    """Point the current transaction at `schema` (`vector` stays reachable through public)."""
    conn.execute(sql.SQL("SET LOCAL search_path TO {}, public").format(sql.Identifier(schema)))


def migrate(conn: psycopg.Connection, schema: str, until: str | None = None) -> list[str]:
    """Apply pending migrations to `schema` — up to and including `until` when given; returns
    the versions applied (empty: up to date)."""
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
        if until is not None and version > until:
            break
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


MIN_PGVECTOR = (0, 8)  # hnsw.iterative_scan, which filtered vector search relies on


def check_vector(conn: psycopg.Connection) -> None:
    """pgvector is new enough, and a real vector operation runs: an extension can look installed
    while its library is missing (KB rag-production-guide §2), so presence alone proves nothing."""
    try:
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        conn.execute("SELECT '[1,2]'::vector <=> '[1,2]'").fetchone()
    except psycopg.Error as exc:
        raise StoreError(f"pgvector is not usable: {exc}") from exc
    version = tuple(int(p) for p in row[0].split(".")[:2]) if row else (0, 0)
    if version < MIN_PGVECTOR:
        found = row[0] if row else "not installed"
        raise StoreError(f"pgvector >= 0.8 is required (found {found})")
