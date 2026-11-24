import sqlite3

from tests.lineage.conftest import Conn, migrated, pending_of

ALL = ["lineage:0001_ledger"]


def _tables(conn: Conn) -> set[str]:
    if isinstance(conn, sqlite3.Connection):
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    else:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
        ).fetchall()
    return {name for (name,) in rows}


def test_pending_lists_every_migration_on_an_empty_database(empty_conn: Conn) -> None:
    assert pending_of(empty_conn) == ALL


def test_migrate_creates_the_tables_once(empty_conn: Conn) -> None:
    assert migrated(empty_conn) == ALL
    assert migrated(empty_conn) == []
    assert pending_of(empty_conn) == []
    assert {"lineage_artifacts", "lineage_edges", "lineage_schema_migrations"} <= _tables(
        empty_conn
    )
