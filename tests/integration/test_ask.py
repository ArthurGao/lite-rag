import re
from typing import Any

import pytest

from lite_rag.engine import RagEngine
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeLLM, FakeReranker
from tests.integration.support import make_engine

DOCS = {
    "refunds": "Refunds are processed within seven days of the return.",
    "shipping": "Shipping is free for orders above fifty dollars.",
    "returns": "Returns must be unused and in the original packaging.",
    "warranty": "Electronics have a one year warranty against defects.",
}


def _engine(store: PgStore, **kwargs: Any) -> RagEngine:
    engine = make_engine(store, size=200, **kwargs)
    for doc_id, text in DOCS.items():
        engine.upsert_text("kb", doc_id, doc_id.title(), text)
    return engine


def test_ask_answers_from_the_hits_and_maps_citations_to_chunks(store: PgStore) -> None:
    llm = FakeLLM("Refunds take seven days [1].")
    engine = _engine(store, llm=llm)

    answer = engine.ask("kb", "How long do refunds take?", top_k=3)

    assert (answer.status, answer.text) == ("answered", "Refunds take seven days [1].")
    assert [c.chunk_id for c in answer.citations] == [answer.sources[0].chunk_id]
    assert answer.sources[0].doc_id == "refunds"
    assert (answer.model_id, answer.dropped) == ("fake/fake-llm", [])
    [messages] = llm.calls
    assert re.search(
        r'<source-[0-9a-f]{12} id="1" from="Refunds">\nRefunds are processed within seven days',
        messages[1].content,
    )
    assert (
        len(answer.sources) == 3
        and ' id="3" from=' in messages[1].content
        and ' id="4" from=' not in messages[1].content
    )


def test_a_citation_to_a_source_that_was_not_given_is_dropped(store: PgStore) -> None:
    engine = _engine(store, llm=FakeLLM("Seven days [1][5]."))
    answer = engine.ask("kb", "refunds", top_k=2)
    assert answer.text == "Seven days [1]." and answer.dropped == [5]


def test_no_hits_means_no_llm_call(store: PgStore) -> None:
    llm = FakeLLM("should not be used")
    answer = _engine(store, llm=llm).ask("kb", "the of and", mode="bm25")
    assert llm.calls == [] and answer.model_id is None and answer.sources == []
    assert (answer.status, answer.text) == ("no_sources", "")


def test_rerank_rescores_the_fused_candidates(store: PgStore) -> None:
    reranker = FakeReranker()
    engine = _engine(store, reranker=reranker)
    hits = engine.search("kb", "warranty defects electronics", top_k=2, rerank=True)
    assert hits[0].doc_id == "warranty"
    assert set(hits[0].ranks) == {"vector", "bm25", "fused", "rerank"}
    assert hits[0].ranks["rerank"] == 1
    assert hits[0].score == hits[0].scores["rerank"]
    [(query, texts)] = reranker.calls
    assert query == "warranty defects electronics"
    assert len(texts) == len(DOCS)  # every candidate up to rerank.top_n (20) was re-scored
    assert any(t.startswith("Warranty\n") for t in texts)  # title given as context


def test_rerank_is_off_unless_asked_and_its_pool_follows_top_n(store: PgStore) -> None:
    reranker = FakeReranker()
    engine = _engine(store, reranker=reranker)
    engine.search("kb", "refunds", top_k=2)
    assert reranker.calls == []  # off by default
    small_pool = make_engine(store, size=200, reranker=reranker, rerank={"top_n": 2})
    small_pool.search("kb", "refunds seven days returns", top_k=1, rerank=True)
    assert len(reranker.calls[-1][1]) == 2  # only rerank.top_n candidates are re-scored


def test_the_prompt_carries_every_source_and_the_question(store: PgStore) -> None:
    llm = FakeLLM("ok")
    _engine(store, llm=llm).ask("kb", "warranty?", top_k=4)
    user = llm.calls[0][1].content
    assert all(f' id="{i}" from=' in user for i in range(1, 5))
    assert user.rstrip().endswith("Question: warranty?")


@pytest.mark.parametrize("bad", ["[0]", "[99]"])
def test_only_given_numbers_survive(store: PgStore, bad: str) -> None:
    answer = _engine(store, llm=FakeLLM(f"Claim {bad}.")).ask("kb", "refunds", top_k=2)
    assert answer.citations == [] and bad not in answer.text
