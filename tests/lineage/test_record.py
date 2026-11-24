import pytest

from lite_rag.lineage import Item, Ledger
from tests.lineage.conftest import Clock

NS = "ns"


def _doc(ledger: Ledger, h: str, key: str = "a") -> str:
    return ledger.record(NS, "document", key, h).artifact.id


def _chunk(ledger: Ledger, key: str, parent: str, **edge: object) -> str:
    return ledger.record(NS, "chunk", key, key, parents=[parent], edge_metadata=edge).artifact.id


def test_new_key_starts_at_version_one(ledger: Ledger) -> None:
    rec = ledger.record(NS, "document", "a", "h1", producer="p", payload="t", metadata={"x": 1})
    a = rec.artifact
    assert (a.version, a.status, a.ended_at, rec.replaced, rec.reused) == (
        1,
        "active",
        None,
        None,
        False,
    )
    assert (a.namespace, a.kind, a.key, a.content_hash) == (NS, "document", "a", "h1")
    assert (a.producer, a.payload, a.metadata) == ("p", "t", {"x": 1})


def test_same_hash_reuses_the_artifact_and_adds_an_edge(ledger: Ledger) -> None:
    d1 = _doc(ledger, "h1")
    c = _chunk(ledger, "k", d1, chunk_id="a#0")
    d2 = ledger.record(NS, "document", "a", "h2")
    again = ledger.record(
        NS, "chunk", "k", "k", parents=[d2.artifact.id], edge_metadata={"chunk_id": "a#1"}
    )
    assert again.reused and again.artifact.id == c and again.replaced is None
    assert d2.artifact.version == 2 and d2.replaced is not None and d2.replaced.id == d1
    assert d2.replaced.status == "superseded"
    # record never cascades: the chunk stays active until the old version is retired
    assert ledger.active(NS, "chunk", "k") is not None


def test_same_hash_on_an_existing_edge_replaces_its_metadata(ledger: Ledger) -> None:
    d1 = _doc(ledger, "h1")
    c = _chunk(ledger, "k", d1, context="old")
    _chunk(ledger, "k", d1, context="new")
    [edge] = [r.via for r in ledger.children(d1)]
    assert (edge.child_id, edge.metadata) == (c, {"context": "new"})


def test_retire_ends_only_the_children_left_without_an_active_parent(
    ledger: Ledger, clock: Clock
) -> None:
    d1 = _doc(ledger, "h1")
    keep, gone = _chunk(ledger, "keep", d1), _chunk(ledger, "gone", d1)
    d2 = ledger.record(NS, "document", "a", "h2")
    _chunk(ledger, "keep", d2.artifact.id)
    ended = ledger.retire(d1)  # already superseded by record: only its orphans end now
    assert [a.id for a in ended] == [gone]
    assert ended[0].status == "superseded" and ended[0].ended_at == clock.now
    assert ledger.active(NS, "chunk", "keep") is not None
    assert ledger.get([keep])[keep].status == "active"


def test_retire_deleted_cascades_down_every_level(ledger: Ledger) -> None:
    d = _doc(ledger, "h1")
    c = _chunk(ledger, "c", d)
    s = ledger.record(NS, "summary", "s", "s", parents=[c]).artifact.id
    ended = ledger.retire(d, status="deleted")
    assert [a.id for a in ended] == [d, c, s]
    assert {a.status for a in ended} == {"deleted"}


def test_retire_keeps_a_child_with_another_active_parent(ledger: Ledger) -> None:
    a, b = _doc(ledger, "h", "a"), _doc(ledger, "h", "b")
    shared = ledger.record(NS, "summary", "s", "s", parents=[a, b]).artifact.id
    assert [x.id for x in ledger.retire(a, status="deleted")] == [a]
    assert ledger.get([shared])[shared].status == "active"


def test_retire_twice_ends_nothing_more(ledger: Ledger) -> None:
    d = _doc(ledger, "h1")
    ledger.retire(d, status="deleted")
    assert ledger.retire(d, status="deleted") == []


def test_record_after_delete_starts_the_next_version(ledger: Ledger) -> None:
    ledger.retire(_doc(ledger, "h1"), status="deleted")
    rec = ledger.record(NS, "document", "a", "h1")
    assert (rec.artifact.version, rec.replaced, rec.reused) == (2, None, False)


def test_retire_namespace_ends_only_that_namespace(ledger: Ledger) -> None:
    d = _doc(ledger, "h1")
    _chunk(ledger, "c", d)
    other = ledger.record("other", "document", "a", "h1").artifact.id
    assert ledger.retire_namespace(NS) == 2
    assert ledger.active(NS, "document", "a") is None
    assert ledger.get([other])[other].status == "active"
    assert ledger.get([d])[d].status == "deleted"


def test_record_with_an_unknown_parent_raises(ledger: Ledger) -> None:
    with pytest.raises(ValueError, match="nope"):
        ledger.record(NS, "chunk", "k", "k", parents=["nope"])


def test_retire_namespace_cascades_to_orphans_in_other_namespaces(ledger: Ledger) -> None:
    d = _doc(ledger, "h1")
    orphan = ledger.record("other", "summary", "s", "s", parents=[d]).artifact.id
    other_doc = ledger.record("other", "document", "b", "h").artifact.id
    shared = ledger.record("other", "summary", "t", "t", parents=[d, other_doc]).artifact.id
    assert ledger.retire_namespace(NS) == 2  # d, and the orphan it left
    found = ledger.get([orphan, shared])
    assert (found[orphan].status, found[shared].status) == ("deleted", "active")


def test_cascade_works_across_parameter_batches(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lite_rag.lineage.ledger as ledger_module

    monkeypatch.setattr(ledger_module, "BATCH", 2)  # a large kb exceeds the driver's limit
    docs = [_doc(ledger, "h", f"d{i}") for i in range(5)]
    # in another namespace: only the cascade (not the namespace update) reaches them
    chunks = [
        ledger.record("other", "chunk", f"c{i}", "c", parents=[d]).artifact.id
        for i, d in enumerate(docs)
    ]
    assert ledger.retire_namespace(NS) == 10
    assert {a.status for a in ledger.get(chunks).values()} == {"deleted"}


def test_record_many_records_each_item_under_the_same_parents(ledger: Ledger) -> None:
    d1 = _doc(ledger, "h1")
    first = ledger.record_many(
        NS, "chunk", [Item("a", "ha", payload="A", edge_metadata={"n": 0}), Item("b", "hb")],
        parents=[d1], producer="p", metadata={"m": 1},
    )  # fmt: skip
    assert [r.artifact.key for r in first] == ["a", "b"]
    assert (first[0].artifact.payload, first[0].artifact.producer) == ("A", "p")
    assert first[0].artifact.metadata == {"m": 1}
    d2 = ledger.record(NS, "document", "a", "h2").artifact.id
    again = ledger.record_many(NS, "chunk", [Item("a", "ha"), Item("b", "hb2")], parents=[d2])
    assert (again[0].reused, again[1].reused) == (True, False)
    assert again[1].artifact.version == 2 and again[1].replaced is not None
    assert {r.via.metadata.get("n") for r in ledger.children(d1)} == {0, None}


def test_record_many_with_an_unknown_parent_writes_nothing(ledger: Ledger) -> None:
    with pytest.raises(ValueError, match="nope"):
        ledger.record_many(NS, "chunk", [Item("a", "ha")], parents=["nope"])
    assert ledger.active(NS, "chunk", "a") is None


def test_deleting_clears_the_text_at_once_superseding_keeps_it(ledger: Ledger) -> None:
    d = _doc(ledger, "h1")
    kept = ledger.record(NS, "chunk", "kept", "k", parents=[d], payload="old text").artifact.id
    ledger.record(NS, "document", "a", "h2")
    ledger.retire(d)  # superseded: the text stays until gc
    assert ledger.get([kept])[kept].payload == "old text"
    d3 = ledger.record(NS, "document", "b", "h").artifact.id
    gone = ledger.record(NS, "chunk", "gone", "g", parents=[d3], payload="secret").artifact.id
    ledger.retire(d3, status="deleted")
    assert ledger.get([gone])[gone].payload is None
    other = ledger.record("x", "chunk", "c", "c", payload="x text").artifact.id
    ledger.retire_namespace("x")
    assert ledger.get([other])[other].payload is None
