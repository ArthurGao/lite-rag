"""The `Store` + `VectorCache` contract (design §3, §4): every backend must pass these, through
the Protocol only — no SQL, no Chroma calls."""

from typing import Any

import pytest

from lite_rag.errors import KbNotFound, StoreError
from lite_rag.keyword import Bm25Inputs, Posting, bm25
from lite_rag.stores.base import (
    CachedStore,
    IndexedChunk,
    KbInfo,
    KbRef,
    KbStats,
    StoredChunk,
    StoredDocument,
    StoredParent,
)
from lite_rag.text import terms
from lite_rag.types import Chunk, Document

NS = "proj/t1"
KB = KbRef(NS, "kb")
CHUNKER = {"strategy": "paragraph", "size": 40, "tokenizer": {"name": "x", "nested": [1, 2]}}


def info(ref: KbRef = KB, model: str = "m1", dim: int = 4) -> KbInfo:
    return KbInfo(ref, "fake", model, dim, "q: ", "d: ", CHUNKER)


def indexed(
    doc_id: str,
    n: int,
    text: str,
    vector: list[float],
    *,
    heading: tuple[str, ...] = (),
    page: int | None = None,
    parent: int | None = None,
    context: str | None = None,
) -> IndexedChunk:
    start = n * 100
    chunk = Chunk(
        f"{doc_id}#{n}", doc_id, n, text, start, start + len(text), 3, heading, page, parent
    )
    return IndexedChunk(chunk, vector, terms(text), context)


def put(
    store: CachedStore,
    doc_id: str,
    chunks: list[tuple[str, list[float]]],
    *,
    kb: KbRef = KB,
    metadata: dict[str, Any] | None = None,
    content_hash: str = "h1",
) -> None:
    doc = Document(doc_id, doc_id.title(), " ".join(t for t, _ in chunks), None, metadata or {})
    store.upsert_document(
        kb, doc, content_hash, [indexed(doc_id, n, t, v) for n, (t, v) in enumerate(chunks)]
    )


def ids(ranking: list[tuple[str, float]]) -> list[str]:
    return [chunk for chunk, _ in ranking]


# --- knowledge bases ---------------------------------------------------------------------------


def test_get_or_create_returns_the_kb_as_stored(store: CachedStore) -> None:
    assert store.get_kb(KB) is None
    assert store.get_or_create_kb(info()) == info()
    assert store.get_or_create_kb(info(model="m2")) == info()  # the stored one: caller compares
    assert store.get_kb(KB) == info()


def test_create_kb_creates_only_if_absent(store: CachedStore) -> None:
    assert store.create_kb(info()) is True
    assert store.create_kb(info(model="m2")) is False
    assert store.get_kb(KB) == info()


def test_list_kbs_is_per_namespace_and_sorted(store: CachedStore) -> None:
    for ref in [KbRef(NS, "b"), KbRef(NS, "a"), KbRef("other", "c")]:
        store.create_kb(info(ref))
    assert store.list_kbs(NS) == ["a", "b"]
    assert store.list_kbs("other") == ["c"]
    assert store.list_kbs("nobody") == []


def test_delete_kb_removes_everything_and_a_new_kb_of_that_name_starts_empty(
    store: CachedStore,
) -> None:
    store.create_kb(info())
    put(store, "d1", [("refunds take seven days", [1, 0, 0, 0])])

    assert store.delete_kb(KB) is True
    assert store.delete_kb(KB) is False
    assert (store.get_kb(KB), store.stats(KB), store.list_kbs(NS)) == (None, None, [])

    store.create_kb(info())
    assert store.vector_query(KB, [1, 0, 0, 0], 5) == []
    assert store.bm25_query(KB, ["refunds"], 5, 1.2, 0.75) == []
    assert store.document_hash(KB, "d1") is None
    assert store.stats(KB) == KbStats(NS, "kb", 0, 0, 0.0, "fake/m1", 4)


# --- documents ---------------------------------------------------------------------------------


def test_upsert_stores_the_document_its_chunks_and_parents(store: CachedStore) -> None:
    store.create_kb(info())
    doc = Document("d1", "Policy", "the full text", "d1.md", {"lang": "en", "n": 3}, (0, 40))
    parent = Chunk("d1#0", "d1", 0, "the parent", 0, 50, 10, ("H",), 2)
    first = indexed(
        "d1", 0, "refunds take seven days", [1, 0, 0, 0],
        heading=("Policy", "Refunds"), page=1, parent=0, context="about refunds",
    )  # fmt: skip
    second = indexed("d1", 1, "shipping is free", [0, 1, 0, 0], page=2)
    store.upsert_document(KB, doc, "h1", [first, second], [parent])

    assert store.document_hash(KB, "d1") == "h1"
    got = store.chunks(KB, ["d1#0", "d1#1", "missing#0"])
    assert got == {
        "d1#0": StoredChunk(
            "d1#0", "d1", "Policy", "refunds take seven days", 0, 23, ("Policy", "Refunds"), 1,
            StoredParent(0, "the parent", 0, 50, ("H",), 2),
        ),
        "d1#1": StoredChunk("d1#1", "d1", "Policy", "shipping is free", 100, 116, (), 2, None),
    }  # fmt: skip
    assert list(store.documents(KB)) == [
        StoredDocument("d1", "Policy", "the full text", {"lang": "en", "n": 3}, "d1.md", (0, 40))
    ]
    assert store.count_missing_text(KB) == 0
    avg = (len(first.terms) + len(second.terms)) / 2
    assert store.stats(KB) == KbStats(NS, "kb", 1, 2, avg, "fake/m1", 4)


def test_an_upsert_replaces_every_old_chunk(store: CachedStore) -> None:
    store.create_kb(info())
    put(store, "d1", [("alpha one", [1, 0, 0, 0]), ("beta two", [0, 1, 0, 0])])
    put(store, "d1", [("gamma three", [0, 0, 1, 0])], content_hash="h2")

    assert store.document_hash(KB, "d1") == "h2"
    assert set(store.chunks(KB, ["d1#0", "d1#1"])) == {"d1#0"}
    assert store.chunks(KB, ["d1#0"])["d1#0"].text == "gamma three"
    assert ids(store.vector_query(KB, [0, 1, 0, 0], 10)) == ["d1#0"]
    [(_, similarity)] = store.vector_query(KB, [0, 0, 1, 0], 1)
    assert similarity == pytest.approx(1.0)  # the new vector, not the old one under that id
    assert store.bm25_query(KB, ["beta"], 10, 1.2, 0.75) == []
    assert ids(store.bm25_query(KB, ["gamma"], 10, 1.2, 0.75)) == ["d1#0"]
    stats = store.stats(KB)
    assert stats is not None and (stats.documents, stats.chunks) == (1, 1)


def test_delete_document_removes_it_from_every_read(store: CachedStore) -> None:
    store.create_kb(info())
    put(store, "d1", [("alpha one", [1, 0, 0, 0])])
    put(store, "d2", [("beta two", [0, 1, 0, 0])])

    assert store.delete_document(KB, "d1") is True
    assert store.delete_document(KB, "d1") is False
    assert store.delete_document(KbRef(NS, "missing"), "d1") is False

    assert store.document_hash(KB, "d1") is None
    assert store.chunks(KB, ["d1#0"]) == {}
    assert ids(store.vector_query(KB, [1, 0, 0, 0], 10)) == ["d2#0"]
    assert store.bm25_query(KB, ["alpha"], 10, 1.2, 0.75) == []
    assert [d.doc_id for d in store.documents(KB)] == ["d2"]
    stats = store.stats(KB)
    assert stats is not None and (stats.documents, stats.chunks, stats.avg_length) == (1, 1, 2.0)


def test_writing_to_a_missing_kb_is_an_error(store: CachedStore) -> None:
    with pytest.raises(KbNotFound):
        put(store, "d1", [("alpha", [1, 0, 0, 0])])


def test_a_vector_of_the_wrong_dimension_is_refused_and_keeps_the_old_version(
    store: CachedStore,
) -> None:
    store.create_kb(info())
    put(store, "d1", [("alpha one", [1, 0, 0, 0])])
    with pytest.raises(StoreError, match="3-d vector in a 4-d kb"):
        put(store, "d1", [("beta two", [1, 0, 0])], content_hash="h2")

    assert store.document_hash(KB, "d1") == "h1"
    assert ids(store.vector_query(KB, [1, 0, 0, 0], 10)) == ["d1#0"]
    assert store.chunks(KB, ["d1#0"])["d1#0"].text == "alpha one"


def test_documents_are_read_page_by_page_in_doc_id_order(store: CachedStore) -> None:
    store.create_kb(info())
    store.create_kb(info(KbRef(NS, "other")))
    for doc_id in ["d3", "d1", "d5", "d2", "d4"]:
        put(store, doc_id, [(f"text of {doc_id}", [1, 0, 0, 0])])
    put(store, "d0", [("elsewhere", [1, 0, 0, 0])], kb=KbRef(NS, "other"))

    assert [d.doc_id for d in store.documents(KB, page=2)] == ["d1", "d2", "d3", "d4", "d5"]


# --- retrieval ---------------------------------------------------------------------------------


def _three_docs(store: CachedStore) -> None:
    store.create_kb(info())
    put(store, "a", [("refunds take seven days", [1, 0, 0, 0])], metadata={"lang": "en"})
    put(store, "b", [("refunds 退款 seven", [1, 1, 0, 0])], metadata={"lang": "zh"})
    put(store, "c", [("shipping is free, refunds too", [0, 0, 1, 0])], metadata={"lang": "en"})


def test_vector_query_ranks_by_cosine_similarity(store: CachedStore) -> None:
    _three_docs(store)
    got = store.vector_query(KB, [1, 0, 0, 0], 3)
    assert ids(got) == ["a#0", "b#0", "c#0"]
    assert [s for _, s in got] == pytest.approx([1.0, 0.5**0.5, 0.0], abs=1e-5)
    assert ids(store.vector_query(KB, [1, 0, 0, 0], 1)) == ["a#0"]


def test_vector_query_applies_the_metadata_filter_before_cutting_to_n(store: CachedStore) -> None:
    _three_docs(store)
    # b is the nearest, but filtered out: n=2 still returns two English chunks
    assert ids(store.vector_query(KB, [1, 1, 0, 0], 2, {"lang": "en"})) == ["a#0", "c#0"]
    assert store.vector_query(KB, [1, 1, 0, 0], 2, {"lang": "fr"}) == []


def test_queries_never_leave_the_kb(store: CachedStore) -> None:
    _three_docs(store)
    for ref in [KbRef(NS, "other"), KbRef("elsewhere", "kb")]:
        store.create_kb(info(ref))
        put(store, "x", [("refunds take seven days", [1, 0, 0, 0])], kb=ref)

    assert ids(store.vector_query(KB, [1, 0, 0, 0], 10)) == ["a#0", "b#0", "c#0"]
    assert set(ids(store.bm25_query(KB, ["refunds"], 10, 1.2, 0.75))) == {"a#0", "b#0", "c#0"}
    assert store.chunks(KB, ["x#0"]) == {}


@pytest.mark.parametrize("where", [None, {"lang": "en"}])
def test_bm25_query_equals_keyword_bm25_over_the_whole_kb_statistics(
    store: CachedStore, where: dict[str, Any] | None
) -> None:
    _three_docs(store)
    query = ["refunds", "seven", "free", "退款"]
    chunk_terms = {
        "a#0": terms("refunds take seven days"),
        "b#0": terms("refunds 退款 seven"),
        "c#0": terms("shipping is free, refunds too"),
    }
    allowed = {"a#0", "c#0"} if where else set(chunk_terms)
    postings = [
        Posting(t, c, ts.count(t), len(ts))
        for c, ts in chunk_terms.items()
        for t in set(ts) & set(query)
        if c in allowed
    ]
    inputs = Bm25Inputs(
        n_chunks=3,
        avg_length=sum(map(len, chunk_terms.values())) / 3,
        doc_freq={t: sum(t in ts for ts in chunk_terms.values()) for t in query},
        postings=postings,
    )
    expected = bm25(inputs, 10, 1.2, 0.75)

    got = store.bm25_query(KB, query, 10, 1.2, 0.75, where)
    assert ids(got) == ids(expected) and len(got) == len(allowed)
    assert [s for _, s in got] == pytest.approx([s for _, s in expected])
    assert store.bm25_query(KB, [], 10, 1.2, 0.75) == []


# --- vector cache ------------------------------------------------------------------------------


def test_the_vector_cache_is_per_model_and_dimension_and_never_overwrites(
    store: CachedStore,
) -> None:
    store.put("m1", 2, {"k1": [0.5, 0.25], "k2": [1.0, -2.0]})
    store.put("m1", 2, {"k1": [9.0, 9.0]})

    assert store.get("m1", 2, ["k1", "k2", "k3"]) == {"k1": [0.5, 0.25], "k2": [1.0, -2.0]}
    assert store.get("m2", 2, ["k1"]) == {}
    assert store.get("m1", 3, ["k1"]) == {}
    assert store.get("m1", 2, []) == {}
