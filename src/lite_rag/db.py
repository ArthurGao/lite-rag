"""PostgreSQL access and migrations (design §4).

Migrations are the `NNNN_*.sql` files in `migrations/`, applied in order, each in its own
transaction, recorded in `<schema>.schema_migrations`. They run with
`search_path = <schema>, public`, so one set of files serves any schema name.
"""

from importlib import resources

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from lite_rag import lineage
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


def migration_files(package: str = MIGRATIONS) -> list[tuple[str, str]]:
    """(version, SQL) of each `NNNN_*.sql` file in `package`, in order."""
    files = resources.files(package)
    names = sorted(f.name for f in files.iterdir() if f.name.endswith(".sql"))
    return [(name.removesuffix(".sql"), files.joinpath(name).read_text("utf-8")) for name in names]


def use_schema(conn: psycopg.Connection, schema: str, *, public: bool = True) -> None:
    """Point the current transaction at `schema`; `public` keeps `vector` reachable. The
    ledger's migrations run without it, so its unqualified tables never resolve to a
    same-named table in public."""
    path = sql.SQL(", public" if public else "")
    conn.execute(sql.SQL("SET LOCAL search_path TO {}{}").format(sql.Identifier(schema), path))


def _lock_schema(conn: psycopg.Connection, schema: str) -> None:
    """Serialise concurrent migrators on this schema (until the transaction ends)."""
    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"lite_rag:{schema}",))


def migrate(conn: psycopg.Connection, schema: str, until: str | None = None) -> list[str]:
    """Apply pending migrations to `schema` — up to and including `until` when given; returns
    the versions applied (empty: up to date)."""
    with conn.transaction():
        # In public: the role and the schema often share a name, and "$user" comes first in
        # search_path, so an unqualified install would land in (and be dropped with) the schema.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.schema_migrations "
                "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            ).format(sql.Identifier(schema))
        )
    applied: list[str] = []
    for version, script in migration_files():
        if until is not None and version > until:
            break
        with conn.transaction():
            use_schema(conn, schema)
            _lock_schema(conn, schema)  # the check below then sees concurrent migrators' work
            done = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s", (version,)
            ).fetchone()
            if done:
                continue
            conn.execute(script.encode())  # trusted SQL file shipped in the package
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied.append(version)
    if until is None:  # the lineage ledger's tables live in the same schema (lineage §5.2)
        with conn.transaction():
            use_schema(conn, schema, public=False)
            _lock_schema(conn, schema)
            applied += lineage.migrate(conn)
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


def pending(conn: psycopg.Connection, schema: str) -> list[str]:
    """The migration versions not yet applied to `schema` (all of them when it has none)."""
    table = sql.Identifier(schema, "schema_migrations")
    applied: set[str] = set()
    with conn.transaction():
        row = conn.execute("SELECT to_regclass(%s)", (table.as_string(conn),)).fetchone()
        if row and row[0] is not None:
            rows = conn.execute(sql.SQL("SELECT version FROM {}").format(table))
            applied = {version for (version,) in rows}
        use_schema(conn, schema, public=False)
        ledger = lineage.pending(conn)
    files = resources.files(MIGRATIONS).iterdir()
    versions = sorted(f.name.removesuffix(".sql") for f in files if f.name.endswith(".sql"))
    return [v for v in versions if v not in applied] + ledger
