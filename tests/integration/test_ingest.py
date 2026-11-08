from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from lite_rag.config import load_settings
from lite_rag.db import migrate
from lite_rag.embedding import CachingEmbedder, Embedder
from lite_rag.engine import RagEngine
from lite_rag.errors import ConfigError
from lite_rag.model_profiles import Prefixes
from lite_rag.stores.base import KbRef
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeEmbedder
from tests.conftest import TEST_SCHEMA

NS = "proj/tenant_1"
KB = KbRef(NS, "kb")
CHUNKING = {"strategy": "paragraph", "unit": "chars", "size": 40, "overlap": 0, "min_size": 0}

DOC_V1 = "Refunds take seven days.\n\nShipping is free above fifty dollars."
DOC_V2 = (
    "Refunds now take three days.\n\nShipping is free above fifty dollars.\n\nReturns: 30 days."
)


@pytest.fixture
def store(pg: psycopg.Connection, pg_url: str) -> Iterator[PgStore]:
    migrate(pg, TEST_SCHEMA)
    s = PgStore(pg_url, TEST_SCHEMA)
    yield s
    s.close()


def _engine(
    store: PgStore,
    embedder: FakeEmbedder | None = None,
    *,
    namespace: str = NS,
    cache: bool = True,
    **chunking: Any,
) -> RagEngine:
    """An engine as `from_settings` builds one, with a fake embedder (cached unless cache=False)."""
    settings = load_settings(chunking={**CHUNKING, **chunking}, embedding={"cache": cache})
    inner = embedder or FakeEmbedder(dim=8)
    wrapped: Embedder = CachingEmbedder(inner, store) if cache else inner
    return RagEngine(settings, store, wrapped, namespace=namespace)


def _count(pg: psycopg.Connection, table: str, kb: KbRef = KB, column: str = "kb") -> int:
    row = pg.execute(
        f"SELECT count(*) FROM {TEST_SCHEMA}.{table} WHERE {column} = %s", (kb.key,)
    ).fetchone()
    assert row is not None
    return int(row[0])


def _chunk_ids(pg: psycopg.Connection) -> list[str]:
    rows = pg.execute(
        f"SELECT id FROM {TEST_SCHEMA}.chunks WHERE kb = %s ORDER BY n", (KB.key,)
    ).fetchall()
    return [r[0] for r in rows]


def test_upsert_writes_chunks_vectors_postings_and_stats(
    store: PgStore, pg: psycopg.Connection
) -> None:
    result = _engine(store).upsert_text("kb", "d1", "Policy", DOC_V1)

    assert (result.chunks, result.skipped) == (2, False)
    assert _chunk_ids(pg) == ["d1#0", "d1#1"]
    assert _count(pg, "embeddings_8") == 2
    terms = pg.execute(
        f"SELECT term, tf FROM {TEST_SCHEMA}.postings WHERE chunk_id = 'd1#0' ORDER BY term"
    ).fetchall()
    assert terms == [("days", 1), ("refunds", 1), ("seven", 1), ("take", 1)]
    stats = store.stats(KB)
    assert stats is not None
    assert (stats.documents, stats.chunks, stats.dim) == (1, 2, 8)
    assert stats.avg_length == (4 + 5) / 2  # BM25 terms per chunk ("is" is a stopword)


def test_the_kb_row_records_namespace_name_and_key(store: PgStore, pg: psycopg.Connection) -> None:
    _engine(store).upsert_text("kb", "d1", "Policy", DOC_V1)
    row = pg.execute(f"SELECT name, namespace, kb FROM {TEST_SCHEMA}.kbs").fetchone()
    assert row == ("proj/tenant_1:kb", "proj/tenant_1", "kb")


def test_the_database_refuses_a_key_that_disagrees_with_its_parts(
    store: PgStore, pg: psycopg.Connection
) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        pg.execute(
            f"INSERT INTO {TEST_SCHEMA}.kbs (name, namespace, kb, embedding_provider,"
            " embedding_model, dim, chunker) VALUES ('a:x', 'b', 'x', 'p', 'm', 4, '{}')"
        )


def test_unchanged_content_is_not_re_embedded(store: PgStore, pg: psycopg.Connection) -> None:
    embedder = FakeEmbedder(dim=8)
    engine = _engine(store, embedder)
    engine.upsert_text("kb", "d1", "Policy", DOC_V1)
    calls = len(embedder.calls)

    again = engine.upsert_text("kb", "d1", "Policy", DOC_V1)

    assert again.skipped and len(embedder.calls) == calls
    assert _chunk_ids(pg) == ["d1#0", "d1#1"]


def test_a_changed_document_replaces_every_old_chunk(
    store: PgStore, pg: psycopg.Connection
) -> None:
    engine = _engine(store)
    engine.upsert_text("kb", "d1", "Policy", DOC_V1)
    engine.upsert_text("kb", "d2", "Other", "Unrelated text here.")

    engine.upsert_text("kb", "d1", "Policy", DOC_V2)

    texts = pg.execute(
        f"SELECT text FROM {TEST_SCHEMA}.chunks WHERE doc_id = 'd1' ORDER BY n"
    ).fetchall()
    assert [t[0] for t in texts] == [
        "Refunds now take three days.",
        "Shipping is free above fifty dollars.",
        "Returns: 30 days.",
    ]
    assert _count(pg, "embeddings_8") == 4  # 3 for d1 + 1 for d2, nothing left over
    stats = store.stats(KB)
    assert stats is not None and (stats.documents, stats.chunks) == (2, 4)
    assert stats.avg_length == (5 + 5 + 3 + 3) / 4  # the old d1 chunks left the statistics
    stale = pg.execute(
        f"SELECT count(*) FROM {TEST_SCHEMA}.postings WHERE term = 'seven'"
    ).fetchone()
    assert stale == (0,)


def test_delete_leaves_nothing_behind(store: PgStore, pg: psycopg.Connection) -> None:
    engine = _engine(store)
    engine.upsert_text("kb", "d1", "Policy", DOC_V1)

    assert engine.delete("kb", "d1") is True
    assert engine.delete("kb", "d1") is False

    for table in ("documents", "chunks", "postings", "embeddings_8"):
        assert _count(pg, table) == 0, table
    stats = store.stats(KB)
    assert stats is not None and (stats.chunks, stats.avg_length) == (0, 0.0)


def test_a_failure_mid_write_keeps_the_old_version(
    store: PgStore, pg: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(store)
    engine.upsert_text("kb", "d1", "Policy", DOC_V1)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "_copy_chunks", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        engine.upsert_text("kb", "d1", "Policy", DOC_V2)

    assert _chunk_ids(pg) == ["d1#0", "d1#1"]  # the old version, untouched
    assert _count(pg, "embeddings_8") == 2
    stats = store.stats(KB)
    assert stats is not None and stats.chunks == 2


def test_a_kb_refuses_another_embedding_model(store: PgStore) -> None:
    _engine(store, FakeEmbedder(dim=8)).upsert_text("kb", "d1", "t", DOC_V1)
    other = _engine(store, FakeEmbedder(dim=8, model="another-model"))
    with pytest.raises(ConfigError, match="embedding_model: built with 'fake-embedder'"):
        other.upsert_text("kb", "d2", "t", "x")


def test_a_kb_refuses_another_chunker(store: PgStore) -> None:
    _engine(store).upsert_text("kb", "d1", "t", DOC_V1)
    with pytest.raises(ConfigError, match="chunker: built with"):
        _engine(store, size=80).upsert_text("kb", "d2", "t", "x")


def test_a_kb_refuses_another_document_prefix(store: PgStore) -> None:
    _engine(store, FakeEmbedder(dim=8)).upsert_text("kb", "d1", "t", DOC_V1)
    prefixed = FakeEmbedder(dim=8, prefixes=Prefixes(document="passage: "))
    with pytest.raises(ConfigError, match="document_prefix: built with ''"):
        _engine(store, prefixed).upsert_text("kb", "d2", "t", "x")


def test_kbs_with_different_dimensions_live_side_by_side(
    store: PgStore, pg: psycopg.Connection
) -> None:
    _engine(store, FakeEmbedder(dim=8)).upsert_text("small", "d", "t", DOC_V1)
    _engine(store, FakeEmbedder(dim=16)).upsert_text("large", "d", "t", DOC_V1)
    assert _count(pg, "embeddings_8", KbRef(NS, "small")) == 2
    assert _count(pg, "embeddings_16", KbRef(NS, "large")) == 2


def test_delete_kb_removes_everything(store: PgStore, pg: psycopg.Connection) -> None:
    engine = _engine(store)
    engine.upsert_text("kb", "d1", "Policy", DOC_V1)
    assert engine.delete_kb("kb") is True
    assert _count(pg, "kbs", column="name") == 0
    for table in ("documents", "chunks", "postings", "embeddings_8"):
        assert _count(pg, table) == 0, table


def test_ingest_a_markdown_file_by_path(
    store: PgStore, pg: psycopg.Connection, tmp_path: Path
) -> None:
    f = tmp_path / "policy.md"
    f.write_text("# Refund policy\n\nRefunds take seven days.\n", encoding="utf-8")
    engine = _engine(store)

    first = engine.ingest("kb", f)
    second = engine.ingest("kb", f)

    assert first.doc_id == second.doc_id and second.skipped
    row = pg.execute(f"SELECT title, source FROM {TEST_SCHEMA}.documents").fetchone()
    assert row == ("Refund policy", str(f))


def test_namespaces_never_see_each_other(store: PgStore) -> None:
    a = _engine(store, namespace="proj/a")
    b = _engine(store, namespace="proj/b")
    a.upsert_text("kb", "d1", "t", DOC_V1)

    assert b.stats("kb") is None  # same kb name, other namespace: nothing there
    assert b.delete("kb", "d1") is False
    stats = a.stats("kb")
    assert stats is not None and (stats.namespace, stats.kb, stats.documents) == ("proj/a", "kb", 1)


@pytest.mark.parametrize("bad", ["", "Proj", "a b", "../x", "a:b", "a/b/c/d/e"])
def test_an_invalid_namespace_is_refused(store: PgStore, bad: str) -> None:
    with pytest.raises(ConfigError, match="namespace"):
        _engine(store, namespace=bad)


def test_a_kb_name_cannot_smuggle_a_namespace(store: PgStore) -> None:
    with pytest.raises(ConfigError, match="invalid knowledge base name"):
        _engine(store).upsert_text("other:kb", "d", "t", "x")


def test_cached_vectors_are_reused_across_kbs_and_namespaces(store: PgStore) -> None:
    first, second = FakeEmbedder(dim=8), FakeEmbedder(dim=8)
    _engine(store, first).upsert_text("kb", "d1", "Policy", DOC_V1)

    _engine(store, second, namespace="another").upsert_text("copy", "d1", "Policy", DOC_V1)

    assert first.calls and second.calls == []  # same model, same inputs: nothing re-embedded


def test_the_cache_is_per_model_and_never_mixes_queries_with_documents(store: PgStore) -> None:
    e = FakeEmbedder(dim=8)
    _engine(store, e).upsert_text("kb", "d1", "Policy", "Refunds take seven days.")
    other_model = FakeEmbedder(dim=8, model="other-model")
    _engine(store, other_model).upsert_text("kb2", "d1", "Policy", "Refunds take seven days.")
    assert other_model.calls == [["Refunds take seven days."]]  # another model: a miss

    cached = CachingEmbedder(e, store)
    cached.embed_query("Refunds take seven days.")
    cached.embed_query("Refunds take seven days.")
    assert e.queries == ["Refunds take seven days."]  # query: own key, embedded once, then reused


def test_disabling_the_cache_embeds_every_time(store: PgStore) -> None:
    e1, e2 = FakeEmbedder(dim=8), FakeEmbedder(dim=8)
    _engine(store, e1, namespace="n1", cache=False).upsert_text("kb", "d", "t", DOC_V1)
    _engine(store, e2, namespace="n2", cache=False).upsert_text("kb", "d", "t", DOC_V1)
    assert e1.calls and e2.calls


def test_a_chars_kb_does_not_record_a_tokenizer(store: PgStore, pg: psycopg.Connection) -> None:
    _engine(store).upsert_text("kb", "d1", "t", DOC_V1)
    row = pg.execute(f"SELECT chunker FROM {TEST_SCHEMA}.kbs").fetchone()
    assert row is not None and row[0]["counter"] == "chars"


@pytest.mark.parametrize("schema", ["a,public", "x; drop", "Public", ""])
def test_the_store_refuses_an_unsafe_schema_name(pg_url: str, schema: str) -> None:
    with pytest.raises(ConfigError, match="schema"):
        PgStore(pg_url, schema)
