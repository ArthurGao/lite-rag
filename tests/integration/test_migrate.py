from unittest.mock import MagicMock

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from typer.testing import CliRunner

from lite_rag.cli import app
from lite_rag.db import check_vector, migrate
from lite_rag.errors import StoreError
from lite_rag.stores.pg import PgStore
from tests.conftest import TEST_SCHEMA

TABLES = {"schema_migrations", "kbs", "documents", "chunks", "postings", "parents"}
TABLES |= {"graph_mentions", "graph_relations"}  # 0006
TABLES |= {"lineage_schema_migrations", "lineage_artifacts", "lineage_edges"}  # S25 ledger
TABLES |= {"rag_query_traces", "rag_query_sources"}  # 0008


def _tables(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (TEST_SCHEMA,)
    ).fetchall()
    return {r[0] for r in rows}


def test_migrate_creates_the_schema_once(pg: psycopg.Connection) -> None:
    assert migrate(pg, TEST_SCHEMA) == [
        "0001_initial",
        "0002_namespace_prefixes",
        "0003_postings_length_doc",
        "0004_pages_parents_context",
        "0005_documents_text",
        "0006_graph",
        "0007_lineage_key",
        "0008_traces",
        "lineage:0001_ledger",
    ]
    assert _tables(pg) == TABLES
    assert migrate(pg, TEST_SCHEMA) == []  # second run: nothing pending
    versions = pg.execute(
        f"SELECT version FROM {TEST_SCHEMA}.schema_migrations ORDER BY version"
    ).fetchall()
    assert versions == [
        ("0001_initial",),
        ("0002_namespace_prefixes",),
        ("0003_postings_length_doc",),
        ("0004_pages_parents_context",),
        ("0005_documents_text",),
        ("0006_graph",),
        ("0007_lineage_key",),
        ("0008_traces",),
    ]


def test_deleting_a_document_cascades_to_its_chunks_and_postings(pg: psycopg.Connection) -> None:
    migrate(pg, TEST_SCHEMA)
    s = TEST_SCHEMA
    pg.execute(
        f"INSERT INTO {s}.kbs (name, namespace, kb, embedding_provider, embedding_model, dim,"
        " chunker) VALUES ('ns:kb', 'ns', 'kb', 'fake', 'm', 4, '{}')"
    )
    pg.execute(
        f"INSERT INTO {s}.documents (kb, doc_id, title, content_hash, chunk_count) "
        "VALUES ('ns:kb', 'd1', 't', 'h', 1)"
    )
    pg.execute(
        f"INSERT INTO {s}.chunks"
        " (kb, id, doc_id, n, text, start_offset, end_offset, tokens, length)"
        " VALUES ('ns:kb', 'd1#0', 'd1', 0, 'x', 0, 1, 1, 1)"
    )
    pg.execute(
        f"INSERT INTO {s}.postings (kb, term, chunk_id, tf, length, doc_id)"
        " VALUES ('ns:kb', 'x', 'd1#0', 1, 1, 'd1')"
    )

    pg.execute(f"DELETE FROM {s}.documents WHERE kb = 'ns:kb' AND doc_id = 'd1'")

    assert pg.execute(f"SELECT count(*) FROM {s}.chunks").fetchone() == (0,)
    assert pg.execute(f"SELECT count(*) FROM {s}.postings").fetchone() == (0,)


def test_pgvector_actually_works(pg: psycopg.Connection) -> None:
    check_vector(pg)


def test_cli_migrate_reports_then_is_up_to_date(pg: psycopg.Connection, pg_url: str) -> None:
    env = {"LITERAG_DATABASE_URL": pg_url, "LITERAG_STORE__SCHEMA": TEST_SCHEMA}
    runner = CliRunner()
    first = runner.invoke(app, ["migrate"], env=env)
    assert first.exit_code == 0, first.output
    assert "applied 0001_initial" in first.output
    password = conninfo_to_dict(pg_url).get("password")
    if password:  # the value may equal the user name (CI: postgres/postgres): check the field
        assert f"password={password}" not in first.output and "password=***" in first.output
    second = runner.invoke(app, ["migrate"], env=env)
    assert second.exit_code == 0 and "up to date" in second.output


def test_cli_migrate_without_a_database_fails_clearly() -> None:
    result = CliRunner().invoke(app, ["migrate"])
    assert result.exit_code == 1
    assert "LITERAG_DATABASE_URL" in result.output


def test_0003_backfills_existing_postings(pg: psycopg.Connection) -> None:
    s = TEST_SCHEMA
    migrate(pg, s, until="0002_namespace_prefixes")
    pg.execute(
        f"INSERT INTO {s}.kbs (name, namespace, kb, embedding_provider, embedding_model, dim,"
        " chunker) VALUES ('ns:kb', 'ns', 'kb', 'fake', 'm', 4, '{}')"
    )
    pg.execute(
        f"INSERT INTO {s}.documents (kb, doc_id, title, content_hash, chunk_count)"
        " VALUES ('ns:kb', 'd1', 't', 'h', 1)"
    )
    pg.execute(
        f"INSERT INTO {s}.chunks (kb, id, doc_id, n, text, start_offset, end_offset, tokens,"
        " length) VALUES ('ns:kb', 'd1#0', 'd1', 0, 'x y', 0, 3, 2, 2)"
    )
    pg.execute(
        f"INSERT INTO {s}.postings (kb, term, chunk_id, tf) VALUES ('ns:kb', 'x', 'd1#0', 1)"
    )

    assert migrate(pg, s, until="0003_postings_length_doc") == ["0003_postings_length_doc"]

    row = pg.execute(f"SELECT length, doc_id FROM {s}.postings").fetchone()
    assert row == (2, "d1")


def test_the_vector_extension_is_created_in_public(pg: psycopg.Connection) -> None:
    # pgvector already exists here, so check the statement sent, not where a fresh install lands.
    spy = MagicMock(wraps=pg)
    migrate(spy, TEST_SCHEMA)
    sent = [str(c.args[0]) for c in spy.execute.call_args_list]
    created = [q for q in sent if "CREATE EXTENSION" in q]
    assert created == ["CREATE EXTENSION IF NOT EXISTS vector SCHEMA public"]


def test_opening_a_store_on_an_outdated_schema_says_to_migrate(
    pg: psycopg.Connection, pg_url: str
) -> None:
    migrate(pg, TEST_SCHEMA, until="0004_pages_parents_context")
    with pytest.raises(StoreError, match=r"0005_documents_text, 0006_graph.*literag migrate"):
        PgStore(pg_url, TEST_SCHEMA)
    migrate(pg, TEST_SCHEMA)
    PgStore(pg_url, TEST_SCHEMA).close()  # up to date: opens


def test_the_store_refuses_a_schema_whose_ledger_is_behind(
    pg: psycopg.Connection, pg_url: str
) -> None:
    """Hosts such as llm-steward call `lite_rag.db.migrate` and open the store themselves: the
    ledger's migrations come with it, and its pending ones block opening like lite-rag's."""
    migrate(pg, TEST_SCHEMA)
    pg.execute(f"DELETE FROM {TEST_SCHEMA}.lineage_schema_migrations")
    with pytest.raises(StoreError, match="lineage:0001_ledger"):
        PgStore(pg_url, TEST_SCHEMA)


def test_backfill_fills_keys_of_chunks_written_before_s25(
    pg: psycopg.Connection, pg_url: str
) -> None:
    """Chunks from before migration 0007 have no lineage key, and old documents no text."""
    from lite_rag.provenance import chunk_keys
    from tests.contract.test_lineage import assert_consistent, write
    from tests.contract.test_store import KB, info

    migrate(pg, TEST_SCHEMA)
    store = PgStore(pg_url, TEST_SCHEMA, lineage=False)
    store.create_kb(info())
    write(store, "d1", ["a", "b"], h="h1")
    store.close()
    pg.execute(f"UPDATE {TEST_SCHEMA}.chunks SET lineage_key = NULL")
    pg.execute(f"UPDATE {TEST_SCHEMA}.documents SET text = NULL")
    store = PgStore(pg_url, TEST_SCHEMA)
    try:
        assert store.backfill_lineage(KB) == 1
        got = store.chunks(KB, ["d1#0", "d1#1"])
        assert [got[c].lineage_key for c in ("d1#0", "d1#1")] == chunk_keys("d1", ["a", "b"])
        assert_consistent(store)
    finally:
        store.close()


def test_a_chunk_without_a_lineage_key_is_traced_as_unrecorded(
    pg: psycopg.Connection, pg_url: str
) -> None:
    """A kb from before S25, not backfilled: tracing still works, its sources unrecorded."""
    from tests.integration.support import make_engine

    migrate(pg, TEST_SCHEMA)
    store = PgStore(pg_url, TEST_SCHEMA)
    try:
        engine = make_engine(store, size=200, trace={"enabled": True})
        engine.upsert_text("kb", "d1", "D1", "refunds take seven days")
        pg.execute(f"UPDATE {TEST_SCHEMA}.chunks SET lineage_key = NULL")
        traced = engine.search_traced("kb", "refunds", mode="bm25")
        with store.traces() as log:
            found = log.get(engine.namespace, traced.trace_id)  # type: ignore[arg-type]
        assert found is not None
        assert {s.artifact_key for s in found[1]} == {None}
        assert {s.chunk_id for s in found[1]} == {"d1#0"}
    finally:
        store.close()


def test_a_report_shows_an_unrecorded_source(pg: psycopg.Connection, pg_url: str) -> None:
    from tests.integration.support import make_engine

    migrate(pg, TEST_SCHEMA)
    store = PgStore(pg_url, TEST_SCHEMA)
    try:
        engine = make_engine(store, size=200, trace={"enabled": True})
        engine.upsert_text("kb", "d1", "D1", "refunds take seven days")
        pg.execute(f"UPDATE {TEST_SCHEMA}.chunks SET lineage_key = NULL")
        traced = engine.search_traced("kb", "refunds", mode="bm25")
        sources = engine.trace_report(traced.trace_id).sources  # type: ignore[arg-type]
        assert {(s.source.stage, s.state, s.artifact) for s in sources} == {
            ("retrieved", "unrecorded", None),
            ("in_context", "unrecorded", None),
        }
    finally:
        store.close()
