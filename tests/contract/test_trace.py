"""Query traces (lineage design §6.3): which chunk versions a search used, at which stage —
on every backend."""

import hashlib
from typing import Any

import pytest

from lite_rag.engine import RagEngine
from lite_rag.errors import InvalidRequest
from lite_rag.stores.base import CachedStore
from lite_rag.testing import FakeLLM, FakeReranker
from lite_rag.tracing import TracedSource, TraceRecord, TraceSource
from tests.integration.support import NS, make_engine

DOCS = {
    "refunds": "Refunds are processed within seven days of the return.",
    "shipping": "Shipping is free for orders above fifty dollars.",
    "returns": "Returns must be unused and in the original packaging.",
}
ON = {"enabled": True}


def _engine(store: CachedStore, trace: dict[str, Any] | None = None, **kw: Any) -> RagEngine:
    engine = make_engine(store, size=200, trace=trace or ON, **kw)
    for doc_id, text in DOCS.items():
        engine.upsert_text("kb", doc_id, doc_id.title(), text)
    return engine


def _trace(store: CachedStore, trace_id: str | None) -> tuple[TraceRecord, list[TraceSource]]:
    """A trace of the test namespace, which must exist."""
    assert trace_id is not None
    with store.traces() as log:
        found = log.get(NS, trace_id)
    assert found is not None
    return found


def _ids(store: CachedStore) -> list[str]:
    with store.traces() as log:
        return log.ids(NS)


def _sources(store: CachedStore, trace_id: str | None) -> dict[str, list[tuple[str, int | None]]]:
    _, sources = _trace(store, trace_id)
    out: dict[str, list[tuple[str, int | None]]] = {}
    for s in sorted(sources, key=lambda s: (s.stage, s.rank or 0)):
        out.setdefault(s.stage, []).append((s.chunk_id, s.rank))
    return out


def test_search_traced_records_each_stage(store: CachedStore) -> None:
    engine = _engine(store, rerank={"enabled": True, "top_n": 3}, reranker=FakeReranker())
    traced = engine.search_traced("kb", "refunds seven days", top_k=2, mode="bm25")
    assert traced.trace_id is not None
    got = _sources(store, traced.trace_id)
    assert set(got) == {"retrieved", "reranked", "in_context"}
    assert [r for _, r in got["retrieved"]] == list(range(1, len(got["retrieved"]) + 1))
    assert [c for c, _ in got["in_context"]] == [h.chunk_id for h in traced.hits]
    assert [c for c, _ in got["reranked"]][:2] == [h.chunk_id for h in traced.hits]
    record, sources = _trace(store, traced.trace_id)
    assert (record.operation, record.namespace, record.kbs) == ("search", engine.namespace, ["kb"])
    assert record.query == "refunds seven days"
    assert record.retrieval["mode"] == "bm25" and record.retrieval["top_k"] == 2
    assert record.retrieval["rerank"] is True
    assert all(s.artifact_ns == f"{engine.namespace}:kb" and s.artifact_key for s in sources)


def test_without_rerank_there_is_no_reranked_stage(store: CachedStore) -> None:
    engine = _engine(store)
    traced = engine.search_traced("kb", "refunds", top_k=1, mode="bm25")
    assert traced.trace_id is not None
    assert set(_sources(store, traced.trace_id)) == {"retrieved", "in_context"}


def test_search_kbs_traced_names_each_kb(store: CachedStore) -> None:
    engine = _engine(store)
    engine.upsert_text("other", "refunds", "Refunds", DOCS["refunds"])
    traced = engine.search_kbs_traced(["kb", "other"], "refunds seven days", mode="bm25")
    assert traced.trace_id is not None
    record, sources = _trace(store, traced.trace_id)
    assert record.kbs == ["kb", "other"]
    in_context = {(s.artifact_ns, s.chunk_id) for s in sources if s.stage == "in_context"}
    ns = engine.namespace
    assert {(f"{ns}:kb", "refunds#0"), (f"{ns}:other", "refunds#0")} <= in_context


def test_plain_search_writes_no_trace(store: CachedStore) -> None:
    engine = _engine(store)
    assert engine.search("kb", "refunds", mode="bm25")  # unchanged: a list of hits
    traced = engine.search_traced("kb", "refunds", mode="bm25")
    assert traced.trace_id is not None
    assert _ids(store) == [traced.trace_id]  # the plain search left none


def test_trace_off_returns_no_id_and_writes_nothing(store: CachedStore) -> None:
    engine = _engine(store, trace={"enabled": False})
    traced = engine.search_traced("kb", "refunds", mode="bm25")
    assert traced.trace_id is None and traced.hits
    assert _ids(store) == []


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("hash", hashlib.sha256(b"refunds").hexdigest()), ("none", None), ("text", "refunds")],
)
def test_the_query_is_kept_as_configured(
    store: CachedStore, mode: str, expected: str | None
) -> None:
    engine = _engine(store, trace={"enabled": True, "query": mode})
    traced = engine.search_traced("kb", "refunds", mode="bm25")
    record, _ = _trace(store, traced.trace_id)
    assert record.query == expected


def test_ask_traces_four_stages_and_returns_the_id(store: CachedStore) -> None:
    llm = FakeLLM(reply="Seven days [2].")
    engine = _engine(store, llm=llm, rerank={"enabled": True, "top_n": 3}, reranker=FakeReranker())
    answer = engine.ask("kb", "refunds shipping orders", top_k=2, mode="bm25")
    assert answer.trace_id is not None and len(answer.sources) == 2
    got = _sources(store, answer.trace_id)
    assert set(got) == {"retrieved", "reranked", "in_context", "cited"}
    assert got["cited"] == [(answer.sources[1].chunk_id, 1)]
    record, _ = _trace(store, answer.trace_id)
    assert (record.operation, record.status, record.model_id) == (
        "ask",
        "answered",
        "fake/fake-model",
    )
    assert record.cached is False and record.usage == answer.usage


def test_ask_without_sources_is_traced_too(store: CachedStore) -> None:
    engine = _engine(store, llm=FakeLLM(reply="unused"))
    answer = engine.ask("kb", "zzz", mode="bm25")
    assert answer.status == "no_sources" and answer.trace_id is not None
    record, sources = _trace(store, answer.trace_id)
    assert (record.status, record.model_id, sources) == ("no_sources", None, [])


def test_ask_with_tracing_off_has_no_id(store: CachedStore) -> None:
    engine = _engine(store, trace={"enabled": False}, llm=FakeLLM(reply="Seven days [1]."))
    assert engine.ask("kb", "refunds", mode="bm25").trace_id is None
    assert _ids(store) == []


def test_record_cited_after_a_traced_search(store: CachedStore) -> None:
    engine = _engine(store)
    engine.upsert_text("other", "refunds", "Refunds", DOCS["refunds"])
    traced = engine.search_kbs_traced(["kb", "other"], "refunds seven days", mode="bm25")
    assert traced.trace_id is not None
    cited = [h for h in traced.hits if h.kb == "other"][:1] + traced.hits[:1]
    engine.record_cited(traced.trace_id, cited)
    _, sources = _trace(store, traced.trace_id)
    rows = sorted((s.rank, s.artifact_ns, s.chunk_id) for s in sources if s.stage == "cited")
    ns = engine.namespace
    assert rows == [
        (1, f"{ns}:other", cited[0].chunk_id),
        (2, f"{ns}:{cited[1].kb}", cited[1].chunk_id),
    ]


def test_record_cited_refuses_a_hit_the_trace_did_not_use(store: CachedStore) -> None:
    engine = _engine(store)
    traced = engine.search_traced("kb", "refunds seven days", top_k=1, mode="bm25")
    other = engine.search("kb", "shipping free", top_k=1, mode="bm25")
    assert traced.trace_id is not None and other[0].chunk_id != traced.hits[0].chunk_id
    with pytest.raises(InvalidRequest):
        engine.record_cited(traced.trace_id, [traced.hits[0], other[0]])
    _, sources = _trace(store, traced.trace_id)
    assert [s for s in sources if s.stage == "cited"] == []  # nothing written


def test_record_cited_refuses_another_namespaces_trace(store: CachedStore) -> None:
    engine = _engine(store)
    traced = engine.search_traced("kb", "refunds", mode="bm25")
    stranger = make_engine(store, size=200, namespace="other/tenant", trace=ON)
    with pytest.raises(InvalidRequest):
        stranger.record_cited(traced.trace_id, traced.hits)  # type: ignore[arg-type]
    with pytest.raises(InvalidRequest):
        stranger.record_cited("no-such-trace", [])


# --- trace_report: was the data current then, is it now ----------------------------------------


def _visa(store: CachedStore, **kw: Any) -> RagEngine:
    engine = make_engine(store, size=200, trace=ON, **kw)
    engine.upsert_text("kb", "visa", "Visa", "The visa fee is 100 dollars.")
    engine.upsert_text("kb", "hours", "Hours", "The office opens at nine.")
    return engine


def _cited(engine: RagEngine, trace_id: str) -> TracedSource:
    [cited] = [s for s in engine.trace_report(trace_id).sources if s.source.stage == "cited"]
    return cited


def test_an_edit_after_the_answer_marks_the_cited_chunk_stale_with_its_old_text(
    store: CachedStore,
) -> None:
    engine = _visa(store, llm=FakeLLM(reply="It is 100 dollars [1]."))
    answer = engine.ask("kb", "visa fee", top_k=1, mode="bm25")
    assert answer.trace_id is not None
    assert _cited(engine, answer.trace_id).state == "current"
    engine.upsert_text("kb", "visa", "Visa", "The visa fee is 120 dollars.")
    cited = _cited(engine, answer.trace_id)
    assert cited.state == "stale" and cited.artifact is not None
    assert cited.artifact.payload == "The visa fee is 100 dollars."
    assert (
        cited.artifact.ended_at is not None
        and cited.artifact.ended_at > engine.trace_report(answer.trace_id).record.created_at
    )


def test_an_edit_during_the_answer_still_names_what_was_retrieved(store: CachedStore) -> None:
    holder: list[RagEngine] = []

    def reply_after_an_edit(messages: list[Any]) -> str:  # the document changes mid-answer
        holder[0].upsert_text("kb", "visa", "Visa", "The visa fee is 120 dollars.")
        return "It is 100 dollars [1]."

    engine = _visa(store, llm=FakeLLM(reply=reply_after_an_edit))
    holder.append(engine)
    answer = engine.ask("kb", "visa fee", top_k=1, mode="bm25")
    cited = _cited(engine, answer.trace_id)  # type: ignore[arg-type]
    assert cited.artifact is not None
    assert cited.artifact.payload == "The visa fee is 100 dollars."  # not the 120 written later
    assert cited.state == "stale"


def test_an_untouched_source_stays_current(store: CachedStore) -> None:
    engine = _visa(store, llm=FakeLLM(reply="Nine [1]."))
    answer = engine.ask("kb", "office opens", top_k=1, mode="bm25")
    engine.upsert_text("kb", "visa", "Visa", "The visa fee is 120 dollars.")
    assert _cited(engine, answer.trace_id).state == "current"  # type: ignore[arg-type]


def test_a_deleted_source_is_stale_without_its_text(store: CachedStore) -> None:
    engine = _visa(store, llm=FakeLLM(reply="It is 100 dollars [1]."))
    answer = engine.ask("kb", "visa fee", top_k=1, mode="bm25")
    engine.delete("kb", "visa")
    cited = _cited(engine, answer.trace_id)  # type: ignore[arg-type]
    assert cited.state == "stale" and cited.artifact is not None
    assert (cited.artifact.status, cited.artifact.payload) == ("deleted", None)


def test_a_report_of_another_namespaces_trace_is_refused(store: CachedStore) -> None:
    engine = _visa(store)
    traced = engine.search_traced("kb", "visa", mode="bm25")
    stranger = make_engine(store, size=200, namespace="other/tenant", trace=ON)
    with pytest.raises(InvalidRequest):
        stranger.trace_report(traced.trace_id)  # type: ignore[arg-type]
    with pytest.raises(InvalidRequest):
        engine.trace_report("no-such-trace")


def test_an_edit_between_the_read_and_the_trace_time_is_still_stale(
    store: CachedStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex S25c: the document changes after the chunks are read, before the trace time is
    taken — the source is the old version, stale, not unrecorded."""
    import lite_rag.engine as engine_module
    from lite_rag.lineage import utcnow

    engine = _visa(store)

    def edit_then_now() -> Any:
        monkeypatch.setattr(engine_module, "utcnow", utcnow)  # once
        engine.upsert_text("kb", "visa", "Visa", "The visa fee is 120 dollars.")
        return utcnow()

    monkeypatch.setattr(engine_module, "utcnow", edit_then_now)
    traced = engine.search_traced("kb", "visa fee", top_k=1, mode="bm25")
    [used] = [
        s for s in engine.trace_report(traced.trace_id).sources if s.source.stage == "in_context"
    ]  # type: ignore[arg-type]
    assert used.state == "stale" and used.artifact is not None
    assert used.artifact.payload == "The visa fee is 100 dollars."
