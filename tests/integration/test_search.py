from typing import Any

import pytest
from psycopg import sql

from lite_rag.engine import RagEngine
from lite_rag.errors import ConfigError, KbNotFound
from lite_rag.stores.base import KbRef
from lite_rag.stores.pg import PgStore, _table, _vector_text, vector_sql
from lite_rag.testing import FakeEmbedder, SpreadEmbedder
from lite_rag.text import terms
from tests.conftest import TEST_SCHEMA
from tests.integration.support import NS, make_engine

DOCS = {
    "refunds": "Refunds are processed within seven days of the return.",
    "shipping": "Shipping is free for orders above fifty dollars.",
    "returns": "Returns must be unused and in the original packaging.",
    "zh-refund": "退款会在七天内处理完成，请耐心等待。",
    "zh-ship": "满五十元包邮，订单在工作日发货。",
}


def _filled(store: PgStore, **kwargs: Any) -> RagEngine:
    engine = make_engine(store, size=200, **kwargs)
    for doc_id, text in DOCS.items():
        meta = {"lang": "zh" if doc_id.startswith("zh") else "en"}
        engine.upsert_text("kb", doc_id, doc_id.title(), text, metadata=meta)
    return engine


def test_bm25_mode_finds_the_exact_term(store: PgStore) -> None:
    engine = _filled(store)
    hits = engine.search("kb", "packaging", mode="bm25")
    assert [h.doc_id for h in hits] == ["returns"]
    assert hits[0].ranks == {"bm25": 1} and "vector" not in hits[0].scores


def test_bm25_segments_chinese_queries(store: PgStore) -> None:
    hits = _filled(store).search("kb", "退款要多久", mode="bm25")
    assert hits[0].doc_id == "zh-refund"


def test_vector_mode_ranks_by_similarity(store: PgStore) -> None:
    hits = _filled(store).search("kb", "free shipping orders", mode="vector")
    assert hits[0].doc_id == "shipping"
    sims = [h.scores["vector"] for h in hits]
    assert sims == sorted(sims, reverse=True) and all(-1.0 <= s <= 1.0 for s in sims)


def test_hybrid_fuses_both_lists_with_rrf(store: PgStore) -> None:
    hits = _filled(store).search("kb", "refunds seven days", top_k=3)
    top = hits[0]
    assert top.doc_id == "refunds"
    assert set(top.ranks) == {"vector", "bm25", "fused"}
    expected = sum(1 / (60 + top.ranks[path]) for path in ("vector", "bm25"))
    assert top.score == pytest.approx(expected)
    assert set(top.scores) == {"vector", "bm25", "fused"}  # every stage, including fusion
    assert top.score == top.scores["fused"]
    assert top.title == "Refunds" and top.text == DOCS["refunds"]


def test_weighted_fusion_is_configurable(store: PgStore) -> None:
    engine = _filled(store, retrieval={"fusion": "weighted", "vector_weight": 0.0})
    bm25_only = engine.search("kb", "packaging")
    assert bm25_only[0].doc_id == "returns" and bm25_only[0].score == pytest.approx(1.0)


@pytest.mark.parametrize("where", [None, {"lang": "en"}])
def test_sql_and_python_bm25_agree(
    store: PgStore, pg_url: str, where: dict[str, str] | None
) -> None:
    engine = _filled(store)
    in_python = PgStore(pg_url, TEST_SCHEMA, bm25_impl="python")
    try:
        ref = KbRef(NS, "kb")
        for query in ["refunds seven days", "orders above fifty", "七天 退款", "shipping 包邮"]:
            q = terms(query)
            sql_hits = store.bm25_query(ref, q, 10, 1.5, 0.75, where)
            py_hits = in_python.bm25_query(ref, q, 10, 1.5, 0.75, where)
            assert [c for c, _ in sql_hits] == [c for c, _ in py_hits], query
            assert [s for _, s in sql_hits] == pytest.approx([s for _, s in py_hits]), query
    finally:
        in_python.close()
    assert engine.search("kb", "orders", mode="bm25")


@pytest.mark.parametrize("impl", ["sql", "python"])
def test_a_filter_narrows_hits_but_never_changes_scores(
    store: PgStore, pg_url: str, impl: str
) -> None:
    """N, n(t) and avgdl are the whole kb's: a chunk scores the same with or without a filter."""
    _filled(store)
    s = PgStore(pg_url, TEST_SCHEMA, bm25_impl=impl)  # type: ignore[arg-type]
    try:
        ref, q = KbRef(NS, "kb"), terms("refunds 退款 seven 七天 days")
        unfiltered = dict(s.bm25_query(ref, q, 10, 1.5, 0.75))
        filtered = dict(s.bm25_query(ref, q, 10, 1.5, 0.75, {"lang": "zh"}))
    finally:
        s.close()
    assert filtered and set(filtered) < set(unfiltered)
    assert all(filtered[c] == pytest.approx(unfiltered[c]) for c in filtered)


def test_metadata_filter_applies_to_both_paths(store: PgStore) -> None:
    engine = _filled(store)
    for mode in ("vector", "bm25", "hybrid"):
        hits = engine.search("kb", "refunds 退款", mode=mode, where={"lang": "zh"})
        assert hits and all(h.doc_id.startswith("zh") for h in hits), mode


def test_a_deleted_document_is_never_returned(store: PgStore) -> None:
    engine = _filled(store)
    engine.delete("kb", "returns")
    hits = engine.search("kb", "packaging unused returns", top_k=10)
    assert "returns" not in {h.doc_id for h in hits}


def test_search_never_crosses_namespaces(store: PgStore) -> None:
    _filled(store)
    other = make_engine(store, namespace="someone/else", size=200)
    with pytest.raises(KbNotFound):
        other.search("kb", "refunds seven days")


def test_the_store_queries_only_the_given_namespace(store: PgStore, pg_url: str) -> None:
    _filled(store)
    theirs_engine = make_engine(store, namespace="someone/else", size=200)
    theirs_engine.upsert_text("kb", "x", "t", "weather refunds")  # shares a term with mine
    mine, theirs = KbRef(NS, "kb"), KbRef("someone/else", "kb")
    q = FakeEmbedder(dim=8).embed_query("refunds seven days")
    assert {c for c, _ in store.vector_query(theirs, q, 50)} == {"x#0"}
    assert len(store.vector_query(mine, q, 50)) == len(DOCS)
    assert [c for c, _ in store.bm25_query(theirs, ["refunds"], 50, 1.5, 0.75)] == ["x#0"]
    in_python = PgStore(pg_url, TEST_SCHEMA, bm25_impl="python")
    try:
        assert [c for c, _ in in_python.bm25_query(theirs, ["refunds"], 50, 1.5, 0.75)] == ["x#0"]
    finally:
        in_python.close()


def test_searching_a_missing_kb_is_an_error_and_creates_nothing(store: PgStore) -> None:
    engine = make_engine(store, size=200)
    with pytest.raises(KbNotFound, match="'nope'"):
        engine.search("nope", "anything")
    assert engine.stats("nope") is None
    assert engine.delete("nope", "doc") is False  # deleting stays idempotent


def test_search_refuses_a_kb_built_with_another_model(store: PgStore) -> None:
    _filled(store)
    other = make_engine(store, FakeEmbedder(dim=8, model="other"), size=200)
    with pytest.raises(ConfigError, match="embedding_model"):
        other.search("kb", "refunds")


def test_a_query_of_only_stopwords_has_no_bm25_hits(store: PgStore) -> None:
    assert _filled(store).search("kb", "the of and", mode="bm25") == []


def test_top_k_and_a_small_kb(store: PgStore) -> None:
    engine = _filled(store)
    assert len(engine.search("kb", "orders", top_k=2)) == 2
    assert len(engine.search("kb", "orders", top_k=50, mode="vector")) == len(DOCS)


def test_a_small_kb_still_fills_top_k_through_the_hnsw_index(pg_url: str, store: PgStore) -> None:
    """Both kbs share one HNSW index and the kb filter applies after the index scan: its
    candidates (ef_search = 40) all belong to the big kb, so without iterative scanning the small
    kb comes back short. `enable_sort=off` makes the planner take the ordered HNSW scan even for
    a small kb (normally it filters by kb first and sorts exactly, which is also correct)."""
    engine = make_engine(store, SpreadEmbedder(dim=8), cache=False, size=60)
    paragraphs = "\n\n".join(f"refund policy seven days {i}" for i in range(2000))
    engine.upsert_text("big", "b", "t", paragraphs)  # 2000 chunks near the query
    for i in range(5):
        engine.upsert_text("small", f"s{i}", "t", f"weather report {i}")

    sep = "&" if "?" in pg_url else "?"
    forced = PgStore(f"{pg_url}{sep}options=-c%20enable_sort%3Doff", TEST_SCHEMA)
    try:
        q = SpreadEmbedder(dim=8).embed_query("refund policy seven days")
        with forced._tx() as conn:  # the plan the store's own query gets on this connection
            plan = conn.execute(
                sql.SQL(
                    "EXPLAIN SELECT chunk_id FROM {} WHERE kb = %s"
                    " ORDER BY embedding <=> %s::{} LIMIT 5"
                ).format(_table("embeddings", 8), vector_sql(8)),
                (KbRef(NS, "small").key, _vector_text(q)),
            ).fetchall()
        assert "ix_embeddings_8_hnsw" in str(plan)  # the HNSW path is really taken
        assert len(forced.vector_query(KbRef(NS, "small"), q, 5)) == 5
    finally:
        forced.close()


def test_the_python_bm25_really_scores_in_python(
    store: PgStore, pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _filled(store)
    import lite_rag.stores.pg as pg_module

    calls: list[int] = []
    real = pg_module.bm25

    def spy(*args: Any, **kwargs: Any) -> list[tuple[str, float]]:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pg_module, "bm25", spy)
    in_python = PgStore(pg_url, TEST_SCHEMA, bm25_impl="python")
    try:
        assert in_python.bm25_query(KbRef(NS, "kb"), ["refunds"], 5, 1.5, 0.75)
        assert store.bm25_query(KbRef(NS, "kb"), ["refunds"], 5, 1.5, 0.75)  # sql: no call
    finally:
        in_python.close()
    assert calls == [1]


def test_hybrid_asks_each_path_for_candidates_single_modes_for_top_k(
    store: PgStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _filled(store)  # candidate_factor defaults to 4
    asked: list[tuple[str, int]] = []
    vector, bm25_query = store.vector_query, store.bm25_query
    monkeypatch.setattr(
        store,
        "vector_query",
        lambda ref, v, n, w=None: asked.append(("vector", n)) or vector(ref, v, n, w),
    )
    monkeypatch.setattr(
        store,
        "bm25_query",
        lambda ref, t, n, k1, b, w=None: (
            asked.append(("bm25", n)) or bm25_query(ref, t, n, k1, b, w)
        ),
    )
    engine.search("kb", "refunds", top_k=3)
    engine.search("kb", "refunds", top_k=3, mode="vector")
    assert asked == [("vector", 12), ("bm25", 12), ("vector", 3)]


def test_a_kb_deleted_elsewhere_is_not_found_on_the_next_read(store: PgStore) -> None:
    reader = _filled(store)
    assert reader.search("kb", "refunds")  # opened once
    make_engine(store, size=200).delete_kb("kb")  # another engine / process drops it
    with pytest.raises(KbNotFound):
        reader.search("kb", "refunds")


def test_sql_and_python_bm25_order_tied_chunks_the_same(store: PgStore, pg_url: str) -> None:
    engine = make_engine(store, size=200)
    for i, words in enumerate(["alpha beta gamma", "gamma beta alpha", "beta gamma alpha"]):
        engine.upsert_text("ties", f"d{i}", "t", words)  # identical statistics: an exact tie
    q = terms("alpha beta gamma")
    in_python = PgStore(pg_url, TEST_SCHEMA, bm25_impl="python")
    try:
        ref = KbRef(NS, "ties")
        sql_hits = store.bm25_query(ref, q, 3, 1.5, 0.75)
        py_hits = in_python.bm25_query(ref, q, 3, 1.5, 0.75)
    finally:
        in_python.close()
    assert [c for c, _ in sql_hits] == [c for c, _ in py_hits] == ["d0#0", "d1#0", "d2#0"]


def test_hybrid_hits_record_the_fused_stage(store: PgStore) -> None:
    top = _filled(store).search("kb", "refunds seven days", top_k=3)[0]
    assert set(top.ranks) == {"vector", "bm25", "fused"} and top.ranks["fused"] == 1
    assert top.score == top.scores["fused"]


def test_chunks_deleted_mid_search_are_replaced_by_the_next_candidates(
    store: PgStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _filled(store)
    first = [h.doc_id for h in engine.search("kb", "refunds seven days", top_k=2)]
    real = store.chunks

    def without_first_two(ref: KbRef, ids: list[str]) -> dict[str, Any]:  # as if just deleted
        found = real(ref, ids)
        return {c: v for c, v in found.items() if v.doc_id not in first}

    monkeypatch.setattr(store, "chunks", without_first_two)
    hits = engine.search("kb", "refunds seven days", top_k=2)
    assert len(hits) == 2 and not {h.doc_id for h in hits} & set(first)
