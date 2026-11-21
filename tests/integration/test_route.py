"""`search(mode="route")` / `ask(mode="route")` (S21): the router's choice is what runs, and every
hit says what was chosen, by which model and why."""

import json

import pytest

from lite_rag.llm import LLMError
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeLLM
from tests.integration.support import make_engine

DOCS = {
    "refunds": "Refunds are processed within seven days of the return.",
    "returns": "Returns must be unused and in the original packaging.",
    "shipping": "Shipping is free for orders above fifty dollars.",
}
ROUTER = {"router": {"modes": ["vector", "bm25"]}}


def _engine(store: PgStore, mode: str, **kwargs: object):  # type: ignore[no-untyped-def]
    router = FakeLLM(json.dumps({"mode": mode, "reason": "a keyword question"}))
    engine = make_engine(store, size=200, retrieval=ROUTER, route_llm=router, **kwargs)
    for doc_id, text in DOCS.items():
        engine.upsert_text("kb", doc_id, doc_id.title(), text)
    return engine, router


@pytest.mark.parametrize("mode", ["vector", "bm25"])
def test_search_runs_the_mode_the_router_chose(store: PgStore, mode: str) -> None:
    engine, router = _engine(store, mode)
    routed = engine.search("kb", "packaging", mode="route")
    direct = engine.search("kb", "packaging", mode=mode)  # type: ignore[arg-type]
    assert [h.chunk_id for h in routed] == [h.chunk_id for h in direct]
    assert [h.scores for h in routed] == [h.scores for h in direct]
    assert {(h.route.mode, h.route.model_id) for h in routed if h.route} == {
        (mode, "fake/fake-model")
    }
    assert all(h.route is not None for h in routed) and all(h.route is None for h in direct)
    assert len(router.calls) == 1  # one routing call per search, none for a fixed mode


def test_the_configured_mode_route_routes_every_search(store: PgStore) -> None:
    _, router = _engine(store, "bm25")
    engine_routed = make_engine(
        store, size=200, retrieval={**ROUTER, "mode": "route"}, route_llm=router
    )
    [hit, *_] = engine_routed.search("kb", "packaging")
    assert hit.route is not None and hit.route.mode == "bm25"


def test_a_router_that_fails_fails_the_search(store: PgStore) -> None:
    _engine(store, "vector")

    class Down(FakeLLM):
        def structured_reply(self, messages, schema):  # type: ignore[no-untyped-def, override]
            raise LLMError("role route (ollama:qwen3:1.7b) failed: APIConnectionError")

    broken = make_engine(store, size=200, retrieval=ROUTER, route_llm=Down())
    with pytest.raises(LLMError):
        broken.search("kb", "packaging", mode="route")


def test_ask_answers_from_the_routed_hits(store: PgStore) -> None:
    engine, _ = _engine(store, "bm25", llm=FakeLLM("Unused, in the packaging [1]."))
    answer = engine.ask("kb", "packaging", mode="route")
    assert answer.status == "answered"
    assert answer.sources[0].route is not None and answer.sources[0].route.mode == "bm25"


def test_no_existing_kb_returns_nothing_without_routing(store: PgStore) -> None:
    router = FakeLLM(json.dumps({"mode": "vector", "reason": "x"}))
    engine = make_engine(store, size=200, retrieval=ROUTER, route_llm=router)
    assert engine.search_kbs(["nothing-here", "nor-here"], "packaging", mode="route") == []
    assert router.calls == []  # Codex S21: no model call for a search over no kb
