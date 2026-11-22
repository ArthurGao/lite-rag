"""Each kb with its own embedder (S22): chosen when the kb is created, found again from the
identity the kb recorded, and searched together across embedders by rank (RRF)."""

import pytest

from lite_rag.engine import RagEngine
from lite_rag.errors import InvalidRequest, KbMismatch
from lite_rag.fusion import rrf
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeEmbedder
from tests.integration.support import NS, make_engine, make_runtime

DOCS = {
    "faq": {"refunds": "Refunds are processed within seven days of the return."},
    "ship": {"shipping": "Shipping is free for orders above fifty dollars of refunds."},
}


def _two(store: PgStore) -> tuple[RagEngine, FakeEmbedder, FakeEmbedder]:
    default, alt = FakeEmbedder(dim=8), FakeEmbedder(dim=16, model="alt-embedder")
    engine = make_runtime(store, default, size=200, embedders={"alt": alt}).engine(NS)
    engine.create_kb("ship", embedding="alt")
    for kb, docs in DOCS.items():
        for doc_id, text in docs.items():
            engine.upsert_text(kb, doc_id, doc_id.title(), text)
    return engine, default, alt


def test_a_kb_created_with_a_named_embedder_keeps_it(store: PgStore) -> None:
    engine, default, alt = _two(store)
    assert engine.create_kb("faq").embedding == "default"
    info = engine.create_kb("ship")  # an existing kb: found again from what it recorded
    assert (info.embedding, info.embedding_model, info.dim) == ("alt", "alt-embedder", 16)
    # each kb's chunks were embedded by its own embedder only
    assert [len(c) for c in default.calls] == [1] and [len(c) for c in alt.calls] == [1]
    assert "Shipping" in alt.calls[0][0] and "Refunds" in default.calls[0][0]


def test_asking_an_existing_kb_for_another_embedder_is_a_mismatch(store: PgStore) -> None:
    engine, _, _ = _two(store)
    with pytest.raises(KbMismatch, match="alt"):
        engine.create_kb("faq", embedding="alt")


def test_a_kb_no_configured_embedder_built_is_a_mismatch_naming_them(store: PgStore) -> None:
    _two(store)
    other = make_engine(store, FakeEmbedder(dim=8, model="another"), size=200)
    with pytest.raises(KbMismatch, match="default"):
        other.search("faq", "refunds")


def test_an_unknown_embedder_name_is_refused(store: PgStore) -> None:
    engine, _, _ = _two(store)
    with pytest.raises(InvalidRequest, match="no embedder 'nope'"):
        engine.create_kb("new", embedding="nope")


def test_search_kbs_embeds_once_per_embedder_and_fuses_their_lists_by_rank(
    store: PgStore,
) -> None:
    engine, default, alt = _two(store)
    default.queries.clear(), alt.queries.clear()
    hits = engine.search_kbs(["faq", "ship"], "refunds", mode="vector", top_k=5)
    assert default.queries == ["refunds"] and alt.queries == ["refunds"]
    assert {h.kb for h in hits} == {"faq", "ship"}
    # each kb's own similarity stays on the hit; the order is the RRF of the two lists
    one = {h.kb: h for h in engine.search_kbs(["faq"], "refunds", mode="vector")}
    two = {h.kb: h for h in engine.search_kbs(["ship"], "refunds", mode="vector")}
    assert {h.kb: h.scores["vector"] for h in hits} == {
        "faq": one["faq"].scores["vector"], "ship": two["ship"].scores["vector"],
    }  # fmt: skip
    fused = rrf({"a": [((0, "f"), 1.0)], "b": [((1, "s"), 1.0)]}, 60)
    assert [h.score for h in hits] == [s for _, s in fused]  # two first places, equal fused score


def test_one_embedder_searches_exactly_as_before(store: PgStore) -> None:
    """With every kb on one embedder nothing changes: raw-score merge, no rank fusion."""
    engine, default, _ = _two(store)
    engine.upsert_text("faq2", "returns", "Returns", "Returns must be unused, refunds follow.")
    hits = engine.search_kbs(["faq", "faq2"], "refunds", mode="vector")
    query = default.embed_query("refunds")
    texts = {h.chunk_id: default.embed_documents([h.text])[0] for h in hits}
    cosine = {c: sum(a * b for a, b in zip(query, v, strict=True)) for c, v in texts.items()}
    assert [h.score for h in hits] == pytest.approx([cosine[h.chunk_id] for h in hits])
    assert [h.score for h in hits] == [h.scores["vector"] for h in hits]  # raw, not RRF


def test_reindex_into_another_embedder(store: PgStore) -> None:
    engine, _, _ = _two(store)
    assert engine.reindex("faq", "faq-alt", embedding="alt") == 1
    info = engine.create_kb("faq-alt")
    assert (info.embedding, info.dim) == ("alt", 16)


def test_a_search_across_embedders_without_rerank_warns_once(
    store: PgStore, caplog: pytest.LogCaptureFixture
) -> None:
    """S22 eval: across embedders each kb's best ties in RRF (hit@1 0.46 vs 1.00 for bge-m3
    alone on golden-xlang); a cross-encoder rerank makes the scores comparable again (1.00)."""
    engine, _, _ = _two(store)
    with caplog.at_level("WARNING", logger="lite_rag.engine"):
        engine.search_kbs(["faq", "ship"], "refunds", mode="vector")
        engine.search_kbs(["faq", "ship"], "refunds", mode="vector")
        engine.search_kbs(["faq"], "refunds", mode="vector")  # one embedder: no warning
    warned = [r for r in caplog.records if "rerank" in r.getMessage()]
    assert len(warned) == 1 and "default, alt" in warned[0].getMessage()
