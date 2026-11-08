import psycopg
import pytest

from lite_rag.errors import InvalidRequest
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeEmbedder
from tests.conftest import TEST_SCHEMA
from tests.integration.support import make_engine

PC = {"strategy": "recursive", "size": 30, "overlap": 0}  # children; parents below
TEXT = (
    "Refunds are paid within seven days. Late refunds go to support.\n\n"
    "Shipping is free above fifty dollars. Orders ship on weekdays.\n\n"
    "Returns must be unused. Keep the original packaging for returns."
)


def _engine(store: PgStore, embedder: FakeEmbedder | None = None, parent_size: int = 70):
    chunking = {**PC, "strategy": "parent_child", "parent_size": parent_size}
    return make_engine(store, embedder, **chunking)


def test_only_children_are_indexed_and_parents_are_stored(
    store: PgStore, pg: psycopg.Connection
) -> None:
    embedder = FakeEmbedder(dim=8)
    _engine(store, embedder).upsert_text("kb", "d", "Policy", TEXT)
    s = TEST_SCHEMA
    (n_children,) = pg.execute(f"SELECT count(*) FROM {s}.chunks").fetchone()  # type: ignore[misc]
    (n_parents,) = pg.execute(f"SELECT count(*) FROM {s}.parents").fetchone()  # type: ignore[misc]
    assert 1 < n_parents < n_children
    assert sum(len(batch) for batch in embedder.calls) == n_children  # parents are not embedded
    orphans = pg.execute(f"SELECT count(*) FROM {s}.chunks WHERE parent_n IS NULL").fetchone()
    assert orphans == (0,)


def test_a_hit_returns_its_parent_and_the_span_that_matched(store: PgStore) -> None:
    engine = _engine(store)
    engine.upsert_text("kb", "d", "Policy", TEXT)
    hit = engine.search("kb", "packaging", top_k=1, mode="bm25")[0]
    assert "Returns must be unused." in hit.text  # the parent: more than the matching child
    assert TEXT[hit.start : hit.end] == hit.text
    assert hit.matched is not None
    a, b = hit.matched
    assert hit.start <= a < b <= hit.end and "packaging" in TEXT[a:b]
    assert len(TEXT[a:b]) <= 30 < len(hit.text)


def test_top_k_means_k_distinct_parents(store: PgStore) -> None:
    engine = _engine(store)
    engine.upsert_text("kb", "d", "Policy", TEXT)
    hits = engine.search("kb", "refunds shipping returns orders", top_k=3, mode="bm25")
    spans = [(h.start, h.end) for h in hits]
    assert len(spans) == len(set(spans)) == 3


def test_deleting_a_document_removes_its_parents(store: PgStore, pg: psycopg.Connection) -> None:
    engine = _engine(store)
    engine.upsert_text("kb", "d", "Policy", TEXT)
    engine.upsert_text("kb", "d", "Policy", TEXT + "\n\nNew rule.")  # replaced, not added to
    engine.upsert_text("fresh", "d", "Policy", TEXT + "\n\nNew rule.")  # the same, built once
    s = TEST_SCHEMA
    counts = dict(pg.execute(f"SELECT kb, count(*) FROM {s}.parents GROUP BY kb").fetchall())
    assert len(counts) == 2 and len(set(counts.values())) == 1  # no parent of the old version left
    assert engine.delete("kb", "d")
    engine.delete("fresh", "d")
    assert pg.execute(f"SELECT count(*) FROM {s}.parents").fetchone() == (0,)


def test_a_kb_refuses_another_parent_size(store: PgStore) -> None:
    _engine(store).upsert_text("kb", "d", "Policy", TEXT)
    with pytest.raises(InvalidRequest, match="parent_size"):
        _engine(store, parent_size=90).search("kb", "refunds")


def test_rerank_scores_one_child_per_parent(store: PgStore) -> None:
    from lite_rag.testing import FakeReranker

    reranker = FakeReranker()
    chunking = {**PC, "strategy": "parent_child", "parent_size": 70}
    engine = make_engine(store, reranker=reranker, rerank={"enabled": True, "top_n": 3}, **chunking)
    engine.upsert_text("kb", "d", "Policy", TEXT)
    hits = engine.search("kb", "refunds shipping returns orders", top_k=3, mode="bm25")
    ((query, texts),) = reranker.calls
    assert len(texts) == 3  # top_n distinct parents, not top_n children of fewer parents
    children = [t.split("\n", 1)[1] for t in texts]  # after the "title > headings" line
    assert all(len(c) <= 30 for c in children)  # the reranker reads children: the precise text
    assert len({(h.start, h.end) for h in hits}) == len(hits) == 3
    # the hits come in the reranker's order, each standing for the child it scored
    scores = dict(zip(children, reranker.score(query, texts), strict=True))
    matched = [TEXT[h.matched[0] : h.matched[1]] for h in hits if h.matched]
    assert matched == sorted(children, key=scores.__getitem__, reverse=True)


def test_search_looks_deeper_when_children_outnumber_the_estimate(store: PgStore) -> None:
    engine = _engine(store)
    engine.upsert_text("kb", "d", "Policy", TEXT)
    engine._chunker.children_per_parent = 1  # an estimate that is too low  # type: ignore[misc]
    hits = engine.search("kb", "refunds shipping returns orders", top_k=3, mode="bm25")
    assert len({(h.start, h.end) for h in hits}) == 3  # still three distinct parents


def test_top_k_below_one_is_an_error(store: PgStore) -> None:
    engine = _engine(store)
    engine.upsert_text("kb", "d", "Policy", TEXT)
    with pytest.raises(InvalidRequest, match="top_k"):
        engine.search("kb", "refunds", top_k=0)
