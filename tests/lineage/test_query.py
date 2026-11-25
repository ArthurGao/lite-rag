from datetime import timedelta

from lite_rag.lineage import Ledger
from tests.lineage.conftest import Clock

NS = "ns"


def test_as_of_picks_the_version_valid_at_that_time(ledger: Ledger, clock: Clock) -> None:
    before = clock.now
    v1 = ledger.record(NS, "document", "a", "h1").artifact
    v2 = ledger.record(NS, "document", "a", "h2").artifact
    assert v2.valid_from > v1.valid_from
    assert ledger.as_of(NS, "document", "a", before) is None
    assert ledger.as_of(NS, "document", "a", v1.valid_from) == ledger.get([v1.id])[v1.id]
    between = v1.valid_from + (v2.valid_from - v1.valid_from) / 2
    assert ledger.as_of(NS, "document", "a", between).id == v1.id  # type: ignore[union-attr]
    assert ledger.as_of(NS, "document", "a", v2.valid_from).id == v2.id  # type: ignore[union-attr]
    later = v2.valid_from + timedelta(days=1)
    assert ledger.as_of(NS, "document", "a", later).id == v2.id  # type: ignore[union-attr]


def test_as_of_after_delete_is_none(ledger: Ledger, clock: Clock) -> None:
    d = ledger.record(NS, "document", "a", "h1").artifact
    [gone] = ledger.retire(d.id, status="deleted")
    assert gone.ended_at is not None
    assert ledger.as_of(NS, "document", "a", gone.ended_at) is None
    assert ledger.as_of(NS, "document", "a", gone.ended_at - timedelta(microseconds=1)) is not None


def test_upstream_reaches_every_document_version(ledger: Ledger) -> None:
    d1 = ledger.record(NS, "document", "a", "h1").artifact.id
    c = ledger.record(NS, "chunk", "k", "k", parents=[d1], edge_metadata={"chunk_id": "a#0"})
    d2 = ledger.record(NS, "document", "a", "h2").artifact.id
    ledger.record(NS, "chunk", "k", "k", parents=[d2], edge_metadata={"chunk_id": "a#1"})
    up = ledger.upstream(c.artifact.id)
    assert {r.artifact.id: r.via.metadata["chunk_id"] for r in up} == {d1: "a#0", d2: "a#1"}


def test_upstream_and_downstream_walk_several_levels(ledger: Ledger) -> None:
    d = ledger.record(NS, "document", "a", "h").artifact.id
    c = ledger.record(NS, "chunk", "c", "c", parents=[d]).artifact.id
    s = ledger.record(NS, "summary", "s", "s", parents=[c]).artifact.id
    assert [r.artifact.id for r in ledger.downstream(d)] == [c, s]
    assert [r.artifact.id for r in ledger.upstream(s)] == [c, d]
    assert ledger.downstream(s) == [] and ledger.upstream(d) == []


def test_downstream_lists_a_shared_child_once(ledger: Ledger) -> None:
    d = ledger.record(NS, "document", "a", "h").artifact.id
    c1 = ledger.record(NS, "chunk", "c1", "c1", parents=[d]).artifact.id
    c2 = ledger.record(NS, "chunk", "c2", "c2", parents=[d]).artifact.id
    s = ledger.record(NS, "summary", "s", "s", parents=[c1, c2]).artifact.id
    assert sorted(r.artifact.id for r in ledger.downstream(d)) == sorted([c1, c2, s])


def test_gc_clears_the_text_of_ended_artifacts_only(ledger: Ledger, clock: Clock) -> None:
    old = ledger.record(NS, "chunk", "old", "o", payload="old text").artifact.id
    ledger.retire(old)
    cutoff = clock.now + timedelta(microseconds=1)
    late = ledger.record(NS, "chunk", "late", "l", payload="late text").artifact.id
    ledger.retire(late)  # ended after the cutoff: kept
    live = ledger.record(NS, "chunk", "live", "v", payload="live text").artifact.id
    assert ledger.gc(cutoff) == 1
    found = ledger.get([old, late, live])
    assert [found[i].payload for i in (old, late, live)] == [None, "late text", "live text"]
    assert found[old].status == "superseded"  # the row stays
    assert ledger.gc(cutoff) == 0


def test_count_active_counts_one_kind_in_one_namespace(ledger: Ledger) -> None:
    d = ledger.record(NS, "document", "a", "h").artifact.id
    ledger.record(NS, "chunk", "c1", "c1", parents=[d])
    ledger.record(NS, "chunk", "c2", "c2", parents=[d])
    ledger.record("other", "chunk", "c3", "c3")
    assert ledger.count_active(NS, "chunk") == 2
    ledger.retire(d, status="deleted")
    assert ledger.count_active(NS, "chunk") == 0


def test_started_by_is_the_latest_version_begun_by_then_even_if_ended(
    ledger: Ledger, clock: Clock
) -> None:
    before = clock.now
    v1 = ledger.record(NS, "chunk", "k", "h").artifact
    [ended] = ledger.retire(v1.id)
    assert ledger.started_by(NS, "chunk", "k", before) is None
    later = ended.ended_at + timedelta(days=1)  # type: ignore[operator]
    assert ledger.as_of(NS, "chunk", "k", later) is None  # not current then
    found = ledger.started_by(NS, "chunk", "k", later)
    assert found is not None and (found.id, found.status) == (v1.id, "superseded")
    v2 = ledger.record(NS, "chunk", "k", "h").artifact
    at_end = ended.ended_at
    assert at_end is not None and at_end < v2.valid_from
    assert ledger.started_by(NS, "chunk", "k", at_end).id == v1.id  # type: ignore[union-attr]
    assert ledger.started_by(NS, "chunk", "k", v2.valid_from).id == v2.id  # type: ignore[union-attr]


def test_gc_with_a_limit_clears_at_most_that_many(ledger: Ledger, clock: Clock) -> None:
    ids = [ledger.record(NS, "chunk", f"c{i}", "h", payload="t").artifact.id for i in range(3)]
    for i in ids:
        ledger.retire(i)
    cutoff = clock.now + timedelta(seconds=1)
    assert ledger.gc(cutoff, limit=2) == 2
    assert ledger.gc(cutoff, limit=2) == 1
    assert ledger.gc(cutoff, limit=2) == 0
