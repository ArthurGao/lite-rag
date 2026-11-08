"""The configured backend (design §4): PostgreSQL by default, Chroma + SQLite with
`store.backend: chroma` (extra `[chroma]`, imported only then). Every backend-specific choice —
opening, migrating, where it lives — is made here, not by callers."""

from lite_rag.config import StoreConfig
from lite_rag.db import check_vector, connect, migrate, redact_conninfo
from lite_rag.extras import require_extra
from lite_rag.stores.base import CachedStore
from lite_rag.stores.pg import PgStore


def open_store(
    store: StoreConfig, database_url: str | None, *, max_connections: int = 4
) -> CachedStore:
    if store.backend == "chroma":
        require_extra("chromadb", "chroma")
        from lite_rag.stores.chroma import ChromaStore

        return ChromaStore(store.data_dir)
    return PgStore(
        database_url,
        store.schema_name,
        max_connections=max_connections,
        bm25_impl=store.bm25_impl,
    )


def migrate_store(store: StoreConfig, database_url: str | None) -> list[str]:
    """Create or upgrade the backend's schema; returns the versions applied."""
    if store.backend == "chroma":
        require_extra("chromadb", "chroma")
        from lite_rag.stores.chroma import migrate_sqlite

        return migrate_sqlite(store.data_dir)
    with connect(database_url) as conn:
        applied = migrate(conn, store.schema_name)
        check_vector(conn)
    return applied


def location(store: StoreConfig, database_url: str | None) -> str:
    """Where the data lives, safe to print (the password masked). Two configs with the same
    location share one store."""
    if store.backend == "chroma":
        return f"chroma {store.data_dir}"
    url = redact_conninfo(database_url) if database_url else "<no database>"
    return f"{url} schema {store.schema_name}"


def evaluation_store(store: StoreConfig, name: str) -> StoreConfig:
    """Where `literag eval` builds its kbs, apart from application data: schema `name`
    (postgres), or the subdirectory `name` of `data_dir` (chroma)."""
    if store.backend == "chroma":
        return store.model_copy(update={"data_dir": store.data_dir / name})
    return store.model_copy(update={"schema_name": name})
