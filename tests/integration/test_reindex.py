import psycopg
import pytest

from lite_rag.errors import InvalidRequest, KbNotFound
from lite_rag.stores.pg import PgStore
from tests.conftest import TEST_SCHEMA
from tests.integration.support import make_engine

DOCS = {
    "refund": ("Refunds", "Refunds take seven days.\n\nLate refunds go to support."),
    "ship": ("Shipping", "Shipping is free above fifty dollars.\n\nOrders ship on weekdays."),
}


def _fill(store: PgStore, **chunking: object):  # type: ignore[no-untyped-def]
    engine = make_engine(store, **chunking)
    for doc_id, (title, text) in DOCS.items():
        engine.upsert_text("kb", doc_id, title, text, metadata={"lang": "en"})
    return engine


def test_reindex_rebuilds_a_kb_under_the_current_settings(
    store: PgStore, pg: psycopg.Connection
) -> None:
    _fill(store, size=30)
    bigger = make_engine(store, size=80)  # another chunker: the old kb cannot be opened with it
    with pytest.raises(InvalidRequest, match="built differently"):
        bigger.search("kb", "refunds")

    assert bigger.reindex("kb", "kb-80") == 2

    s = TEST_SCHEMA
    counts = dict(
        pg.execute(
            f"SELECT k.kb, count(*) FROM {s}.chunks c JOIN {s}.kbs k ON k.name = c.kb GROUP BY k.kb"
        ).fetchall()
    )
    assert counts["kb-80"] < counts["kb"]  # bigger chunks, fewer of them; the source untouched
    hit = bigger.search("kb-80", "refunds", mode="bm25")[0]
    assert hit.doc_id == "refund" and "seven days" in hit.text
    meta = pg.execute(
        f"SELECT metadata, title FROM {s}.documents WHERE kb LIKE '%:kb-80' AND doc_id = 'ship'"
    ).fetchone()
    assert meta == ({"lang": "en"}, "Shipping")  # metadata and title travel too


def test_reindex_refuses_documents_stored_without_their_text(
    store: PgStore, pg: psycopg.Connection
) -> None:
    _fill(store, size=30)
    pg.execute(f"UPDATE {TEST_SCHEMA}.documents SET text = NULL WHERE doc_id = 'ship'")  # pre-0005
    engine = make_engine(store, size=80)
    with pytest.raises(InvalidRequest, match=r"1 document.*re-ingest"):
        engine.reindex("kb", "kb-80")
    with pytest.raises(KbNotFound):  # nothing half-built is left behind
        engine.search("kb-80", "refunds")


def test_reindex_never_writes_into_an_existing_kb(store: PgStore) -> None:
    engine = _fill(store, size=30)
    engine.upsert_text("other", "x", "X", "Something else.")
    with pytest.raises(InvalidRequest, match="already exists"):
        engine.reindex("kb", "other")
    with pytest.raises(KbNotFound):
        engine.reindex("missing", "new")


def test_a_namespace_lists_only_its_own_kbs(store: PgStore) -> None:
    engine = _fill(store, size=30)
    engine.upsert_text("faq", "q", "Q", "A question.")
    make_engine(store, namespace="other/ns").upsert_text("secret", "s", "S", "Hidden.")
    assert engine.list_kbs() == ["faq", "kb"]


def test_a_reindex_that_fails_halfway_leaves_no_target(store: PgStore) -> None:
    from lite_rag.embedding import EmbeddingError
    from lite_rag.testing import FakeEmbedder

    class FailsOnSecondDocument(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if self.calls:
                raise EmbeddingError("model went away")
            return super().embed_documents(texts)

    _fill(store, size=30)
    engine = make_engine(store, FailsOnSecondDocument(dim=8), cache=False, size=80)
    with pytest.raises(EmbeddingError):
        engine.reindex("kb", "kb-80")
    with pytest.raises(KbNotFound):  # the half-built kb (one document) was dropped
        engine.search("kb-80", "refunds")


def test_reindex_never_takes_over_a_target_created_meanwhile(
    store: PgStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _fill(store, size=30)
    engine.upsert_text("kb-80", "theirs", "Theirs", "Someone else's document.")
    real_get = store.get_kb
    # the race: the "does the target exist?" check runs before the other writer created it
    monkeypatch.setattr(store, "get_kb", lambda ref: None if ref.name == "kb-80" else real_get(ref))
    with pytest.raises(InvalidRequest, match="already exists"):
        engine.reindex("kb", "kb-80")
    monkeypatch.undo()
    hits = engine.search("kb-80", "someone else", mode="bm25")
    assert [h.doc_id for h in hits] == ["theirs"]  # neither written into nor deleted
