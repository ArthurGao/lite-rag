"""Lineage through the store (lineage design §6.2): every write records the document version and
its chunks in the ledger, in the write's own transaction — on every backend."""

from collections.abc import Iterator, Sequence
from datetime import timedelta

import pytest

from lite_rag.errors import InvalidRequest, KbNotFound
from lite_rag.lineage import Artifact, utcnow
from lite_rag.maintenance import collect
from lite_rag.provenance import chunk_keys, text_hash
from lite_rag.stores.base import CachedStore, IndexedChunk, KbRef
from lite_rag.stores.factory import open_store
from lite_rag.types import Document
from tests.contract.conftest import Migrated
from tests.contract.test_store import KB, NS, indexed, info
from tests.integration.support import NS as ENGINE_NS
from tests.integration.support import make_engine

V = [1.0, 0.0, 0.0, 0.0]


def write(
    store: CachedStore,
    doc_id: str,
    texts: Sequence[str],
    *,
    h: str,
    kb: KbRef = KB,
    title: str = "T",
    contexts: Sequence[str | None] | None = None,
) -> None:
    ctx = contexts or [None] * len(texts)
    chunks: list[IndexedChunk] = [
        indexed(doc_id, n, t, V, context=c) for n, (t, c) in enumerate(zip(texts, ctx, strict=True))
    ]
    store.upsert_document(kb, Document(doc_id, title, " ".join(texts)), h, chunks)


def active_doc(store: CachedStore, doc_id: str, kb: KbRef = KB) -> Artifact | None:
    with store.ledger() as ledger:
        return ledger.active(kb.key, "document", doc_id)


def chunk(
    store: CachedStore, doc_id: str, text: str, k: int = 0, kb: KbRef = KB
) -> Artifact | None:
    key = f"{doc_id}/{text_hash(text)[:16]}/{k}"
    with store.ledger() as ledger:
        return ledger.active(kb.key, "chunk", key)


def assert_consistent(store: CachedStore, kb: KbRef = KB) -> None:
    """Every stored chunk is reached from its document's active version by exactly one edge, to
    an active chunk artifact with the chunk's key; and no other chunk artifact is active."""
    total = 0
    with store.ledger() as ledger:
        for d in store.documents(kb):
            doc = ledger.active(kb.key, "document", d.doc_id)
            assert doc is not None
            edges = {r.via.metadata["chunk_id"]: r.artifact for r in ledger.children(doc.id)}
            stored = store.chunks(kb, list(edges))
            assert set(stored) == set(edges)
            for chunk_id, artifact in edges.items():
                assert artifact.status == "active"
                assert artifact.key == stored[chunk_id].lineage_key
            total += len(edges)
        assert ledger.count_active(kb.key, "chunk") == total
    stats = store.stats(kb)
    assert (stats.chunks if stats else 0) == total


def test_upsert_records_the_document_and_its_chunks(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["alpha", "beta"], h="h1", title="Policy")
    doc = active_doc(store, "d1")
    assert doc is not None and (doc.version, doc.content_hash) == (1, "h1")
    assert doc.metadata == {"title": "Policy", "source": None, "embedding": "fake/m1", "dim": 4}
    a = chunk(store, "d1", "alpha")
    assert a is not None
    assert (a.content_hash, a.payload, a.metadata) == (text_hash("alpha"), "alpha", {})
    assert a.producer is not None and a.producer.startswith("chunker:paragraph@")
    with store.ledger() as ledger:
        [edge] = [r.via for r in ledger.children(doc.id) if r.artifact.id == a.id]
    assert edge.metadata == {
        "chunk_id": "d1#0", "n": 0, "start": 0, "end": 5, "page": None, "parent_n": None,
    }  # fmt: skip
    assert_consistent(store)


def test_an_edit_versions_only_the_chunks_whose_text_changed(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["one", "two", "three", "four"], h="h1")
    before = {t: chunk(store, "d1", t) for t in ("one", "two", "three", "four")}
    # a new paragraph before "three" renumbers it and "four"; "four" changes
    write(store, "d1", ["one", "two", "new", "three", "four!"], h="h2")
    for t in ("one", "two", "three"):
        now = chunk(store, "d1", t)
        assert now is not None and now.id == before[t].id  # type: ignore[union-attr]
    assert chunk(store, "d1", "four") is None
    with store.ledger() as ledger:
        old = ledger.get([before["four"].id])[before["four"].id]  # type: ignore[union-attr]
        assert old.status == "superseded"
        three = before["three"]
        assert three is not None
        up = {r.artifact.version: r.via.metadata["chunk_id"] for r in ledger.upstream(three.id)}
    assert up == {1: "d1#2", 2: "d1#3"}  # one edge per document version, each its position
    assert_consistent(store)


def test_a_context_change_alone_reuses_every_chunk(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["a", "b"], h="h1", contexts=["c1", "c1"])
    first = chunk(store, "d1", "a")
    write(store, "d1", ["a", "b"], h="h2", contexts=["c2", "c2"])
    assert chunk(store, "d1", "a") == first
    doc = active_doc(store, "d1")
    assert doc is not None
    with store.ledger() as ledger:  # LLM-written text about the content: never in the ledger
        assert all("context" not in r.via.metadata for r in ledger.children(doc.id))
    assert_consistent(store)


def test_repeated_texts_are_separate_artifacts(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["same", "other", "same"], h="h1")
    a, b = chunk(store, "d1", "same", 0), chunk(store, "d1", "same", 1)
    assert a is not None and b is not None and a.id != b.id
    assert_consistent(store)


def test_a_title_only_change_reuses_every_chunk(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["a", "b"], h="h1", title="Old")
    first = chunk(store, "d1", "a")
    write(store, "d1", ["a", "b"], h="h2", title="New")
    doc = active_doc(store, "d1")
    assert doc is not None and doc.version == 2 and doc.metadata["title"] == "New"
    assert chunk(store, "d1", "a") == first
    assert_consistent(store)


def test_an_empty_new_version_supersedes_every_chunk(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["a", "b"], h="h1")
    old = chunk(store, "d1", "a")
    write(store, "d1", [], h="h2")
    assert chunk(store, "d1", "a") is None
    with store.ledger() as ledger:
        assert old is not None and ledger.get([old.id])[old.id].status == "superseded"
    assert_consistent(store)


def test_delete_marks_the_document_and_its_chunks_deleted(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["a"], h="h1")
    doc, a = active_doc(store, "d1"), chunk(store, "d1", "a")
    assert store.delete_document(KB, "d1")
    with store.ledger() as ledger:
        assert doc is not None and a is not None
        found = ledger.get([doc.id, a.id])
    assert {x.status for x in found.values()} == {"deleted"}
    assert_consistent(store)


def test_deleting_a_missing_document_changes_nothing(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["a"], h="h1")
    assert not store.delete_document(KB, "nope")
    assert chunk(store, "d1", "a") is not None
    assert_consistent(store)


def test_the_same_document_in_two_kbs_is_recorded_apart(store: CachedStore) -> None:
    other = KbRef(NS, "copy")  # what `reindex` writes into
    store.create_kb(info())
    store.create_kb(info(other))
    write(store, "d1", ["a"], h="h1")
    write(store, "d1", ["a"], h="h1", kb=other)
    assert store.delete_document(KB, "d1")
    assert chunk(store, "d1", "a") is None
    assert chunk(store, "d1", "a", kb=other) is not None
    assert_consistent(store, other)


def test_delete_kb_leaves_nothing_active(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["a", "b"], h="h1")
    assert store.delete_kb(KB)
    with store.ledger() as ledger:
        assert ledger.count_active(KB.key, "chunk") == 0
        assert ledger.active(KB.key, "document", "d1") is None


def test_a_sequence_of_writes_keeps_the_ledger_consistent(store: CachedStore) -> None:
    store.create_kb(info())
    steps = [
        ("d1", ["a", "b", "c"]), ("d2", ["x", "y"]), ("d1", ["a", "c", "b"]),
        ("d2", ["x", "x", "z"]), ("d1", []), ("d3", ["a"]), ("d1", ["b"]),
    ]  # fmt: skip
    for i, (doc_id, texts) in enumerate(steps):
        write(store, doc_id, texts, h=f"h{i}")
        assert_consistent(store)
    store.delete_document(KB, "d2")
    assert_consistent(store)


@pytest.fixture
def unrecorded(migrated: Migrated) -> Iterator[CachedStore]:
    s = open_store(*migrated, lineage=False)
    yield s
    s.close()


def test_lineage_off_records_nothing(unrecorded: CachedStore) -> None:
    unrecorded.create_kb(info())
    write(unrecorded, "d1", ["a"], h="h1")
    got = unrecorded.chunks(KB, ["d1#0"])["d1#0"]
    assert got.lineage_key == chunk_keys("d1", ["a"])[0]  # the key is stored regardless
    with pytest.raises(Exception, match="lineage is disabled"):
        with unrecorded.ledger():
            pass


def test_backfill_records_what_was_written_without_lineage(migrated: Migrated) -> None:
    off = open_store(*migrated, lineage=False)
    off.create_kb(info())
    write(off, "d1", ["a", "b", "a"], h="h1", contexts=["x", "y", "z"])
    write(off, "d2", ["c"], h="h2")
    off.close()
    store = open_store(*migrated)
    try:
        assert store.backfill_lineage(KB) == 2
        assert_consistent(store)
        doc = active_doc(store, "d1")
        assert doc is not None and doc.content_hash == "h1"
        with store.ledger() as ledger:
            assert len(ledger.children(doc.id)) == 3
        assert store.backfill_lineage(KB) == 0  # recorded already: nothing to do
    finally:
        store.close()


def test_backfill_of_a_missing_kb_raises(store: CachedStore) -> None:
    with pytest.raises(KbNotFound):
        store.backfill_lineage(KB)


# --- gc and inspection (S25d) ---------------------------------------------------------------------


def test_gc_clears_old_text_keeps_rows_and_recent_text(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["old"], h="h1")
    old = chunk(store, "d1", "old")
    write(store, "d1", ["new"], h="h2")  # "old" superseded now
    days = {"keep_text_days": 90, "retention_days": 30}
    assert collect(store, **days, now=utcnow() + timedelta(days=89)) == (0, 0)
    assert collect(store, **days, now=utcnow() + timedelta(days=91), batch=1) == (1, 0)
    with store.ledger() as ledger:
        assert old is not None
        kept = ledger.get([old.id])[old.id]
    assert (kept.status, kept.payload) == ("superseded", None)  # the row stays
    assert chunk(store, "d1", "new").payload == "new"  # type: ignore[union-attr]


def test_gc_with_lineage_off_leaves_the_ledger_alone(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["old"], h="h1")
    write(store, "d1", ["new"], h="h2")
    later = utcnow() + timedelta(days=365)
    assert collect(store, keep_text_days=None, retention_days=30, now=later) == (0, 0)
    assert chunk_text_of_old(store) == "old"


def chunk_text_of_old(store: CachedStore) -> str | None:
    with store.ledger() as ledger:
        [old] = ledger.history(KB.key, "chunk", chunk_keys("d1", ["old"])[0])
    return old.payload


def test_gc_deletes_traces_past_retention(store: CachedStore) -> None:
    engine = make_engine(store, size=200, trace={"enabled": True})
    engine.upsert_text("kb", "d1", "D1", "refunds take seven days")
    first = engine.search_traced("kb", "refunds", mode="bm25").trace_id
    days = {"keep_text_days": 90, "retention_days": 30}
    assert collect(store, **days, now=utcnow() + timedelta(days=29)) == (0, 0)
    assert collect(store, **days, now=utcnow() + timedelta(days=31), batch=1) == (0, 1)
    with store.traces() as log:
        assert log.get(ENGINE_NS, first) is None  # type: ignore[arg-type]
        assert log.ids(ENGINE_NS) == []


def test_document_history_lists_each_version_with_its_chunks(store: CachedStore) -> None:
    engine = make_engine(store, size=200)
    engine.upsert_text("kb", "d1", "D1", "one\n\ntwo")
    engine.upsert_text("kb", "d1", "D1", "one\n\nthree")
    history = engine.lineage_document("kb", "d1")
    assert [(v.version, v.status) for v, _ in history] == [(2, "active"), (1, "superseded")]
    newest_chunks = {r.artifact.payload for r in history[0][1]}
    assert any("three" in (p or "") for p in newest_chunks)


def test_upstream_of_another_namespaces_artifact_is_refused(store: CachedStore) -> None:
    engine = make_engine(store, size=200)
    engine.upsert_text("kb", "d1", "D1", "refunds take seven days")
    [(doc, chunks)] = engine.lineage_document("kb", "d1")
    assert [r.artifact.id for r in engine.lineage_upstream(chunks[0].artifact.id)] == [doc.id]
    assert [r.artifact.id for r in engine.lineage_downstream(doc.id)] == [chunks[0].artifact.id]
    stranger = make_engine(store, size=200, namespace="other/tenant")
    with pytest.raises(InvalidRequest):
        stranger.lineage_upstream(chunks[0].artifact.id)
    with pytest.raises(InvalidRequest):
        stranger.lineage_downstream(doc.id)


@pytest.mark.parametrize("batch", [0, -1])
def test_gc_refuses_a_batch_that_would_never_finish(store: CachedStore, batch: int) -> None:
    with pytest.raises(InvalidRequest, match="batch"):
        collect(store, keep_text_days=90, retention_days=30, batch=batch)


def test_delete_leaves_no_title_or_source_behind(store: CachedStore) -> None:
    store.create_kb(info())
    write(store, "d1", ["a"], h="h1", title="Confidential HR title")
    doc = active_doc(store, "d1")
    store.delete_document(KB, "d1")
    with store.ledger() as ledger:
        assert doc is not None and ledger.get([doc.id])[doc.id].metadata == {}


def test_backfill_ends_documents_deleted_while_lineage_was_off(migrated: Migrated) -> None:
    on = open_store(*migrated)
    on.create_kb(info())
    write(on, "d1", ["a"], h="h1")
    write(on, "d2", ["b"], h="h2")
    on.close()
    off = open_store(*migrated, lineage=False)
    off.delete_document(KB, "d1")  # the ledger does not hear of it
    off.close()
    store = open_store(*migrated)
    try:
        assert active_doc(store, "d1") is not None  # out of step until backfill
        assert store.backfill_lineage(KB) == 0
        assert active_doc(store, "d1") is None and chunk(store, "d1", "a") is None
        assert active_doc(store, "d2") is not None
        assert_consistent(store)
    finally:
        store.close()
