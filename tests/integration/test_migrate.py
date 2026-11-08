import psycopg
from psycopg.conninfo import conninfo_to_dict
from typer.testing import CliRunner

from lite_rag.cli import app
from lite_rag.db import check_vector, migrate
from tests.conftest import TEST_SCHEMA

TABLES = {"schema_migrations", "kbs", "documents", "chunks", "postings", "parents"}


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
