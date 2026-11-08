"""The engine on every backend: what users do — ingest, search in each mode, parent_child,
filters, reindex — must behave the same whichever store holds the kb."""

import pytest

from lite_rag.stores.base import CachedStore
from lite_rag.testing import FakeEmbedder
from tests.integration.support import make_engine

DOCS = {
    "refunds": "Refunds are processed within seven days of the return.",
    "shipping": "Shipping is free for orders above fifty dollars.",
    "returns": "Returns must be unused and in the original packaging.",
    "zh-refund": "退款会在七天内处理完成，请耐心等待。",
}
TEXT = (
    "Refunds are paid within seven days. Late refunds go to support.\n\n"
    "Shipping is free above fifty dollars. Orders ship on weekdays.\n\n"
    "Returns must be unused. Keep the original packaging for returns."
)


def _filled(store: CachedStore, embedder: FakeEmbedder | None = None):  # type: ignore[no-untyped-def]
    engine = make_engine(store, embedder, size=200)
    for doc_id, text in DOCS.items():
        meta = {"lang": "zh" if doc_id.startswith("zh") else "en"}
        engine.upsert_text("kb", doc_id, doc_id.title(), text, metadata=meta)
    return engine


@pytest.mark.parametrize(
    ("mode", "query", "top"),
    [
        ("bm25", "packaging", "returns"),
        ("bm25", "退款要多久", "zh-refund"),
        ("vector", "free shipping orders", "shipping"),
        ("hybrid", "refunds seven days", "refunds"),
    ],
)
def test_each_search_mode_finds_the_document(
    store: CachedStore, mode: str, query: str, top: str
) -> None:
    hits = _filled(store).search("kb", query, mode=mode)  # type: ignore[arg-type]
    assert hits[0].doc_id == top
    assert hits[0].text == DOCS[top] and hits[0].title == top.title()


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_a_filter_narrows_the_hits_of_every_mode(store: CachedStore, mode: str) -> None:
    hits = _filled(store).search("kb", "refunds 退款", where={"lang": "zh"}, mode=mode)  # type: ignore[arg-type]
    assert [h.doc_id for h in hits] == ["zh-refund"]


def test_unchanged_content_is_skipped_and_changed_content_replaced(store: CachedStore) -> None:
    embedder = FakeEmbedder(dim=8)
    engine = _filled(store, embedder)
    calls = len(embedder.calls)
    en = {"lang": "en"}  # metadata is part of the content hash
    assert engine.upsert_text("kb", "refunds", "Refunds", DOCS["refunds"], metadata=en).skipped
    assert len(embedder.calls) == calls
    engine.upsert_text("kb", "refunds", "Refunds", "Refunds now take three days.")
    assert engine.search("kb", "seven", mode="bm25") == []
    assert engine.search("kb", "three", mode="bm25")[0].doc_id == "refunds"


def test_parent_child_returns_the_parent_of_the_matching_child(store: CachedStore) -> None:
    engine = make_engine(store, strategy="parent_child", size=30, overlap=0, parent_size=70)
    engine.upsert_text("kb", "d", "Policy", TEXT)
    hit = engine.search("kb", "packaging", top_k=1, mode="bm25")[0]
    assert "Returns must be unused." in hit.text and TEXT[hit.start : hit.end] == hit.text
    assert hit.matched is not None and "packaging" in TEXT[hit.matched[0] : hit.matched[1]]


def test_reindex_rebuilds_a_kb_under_new_settings(store: CachedStore) -> None:
    _filled(store)
    bigger = make_engine(store, size=400)
    assert bigger.reindex("kb", "kb-400") == len(DOCS)
    hit = bigger.search("kb-400", "packaging", mode="bm25")[0]
    assert hit.doc_id == "returns" and hit.text == DOCS["returns"]
    zh = bigger.search("kb-400", "退款", where={"lang": "zh"}, mode="bm25")
    assert [h.doc_id for h in zh] == ["zh-refund"]  # metadata travelled too
    assert sorted(bigger.list_kbs()) == ["kb", "kb-400"]
