"""The retrieval router (S21): a small model picks how one question is retrieved, from the modes
the configuration allows. A failed or unusable reply raises; there is no default mode to fall
back to (rule 5)."""

import json

import pytest

from lite_rag.config import LLMConfig, RouterConfig, Settings, load_settings
from lite_rag.llm import ROLES, BadReply, LLMError, Message
from lite_rag.route import Router
from lite_rag.testing import FakeLLM


def _reply(mode: str, reason: str = "names one fact") -> str:
    return json.dumps({"mode": mode, "reason": reason})


def test_the_router_returns_the_mode_the_model_chose_with_its_provenance() -> None:
    llm = FakeLLM(_reply("graph", "two hops"))
    route = Router(llm, ["vector", "graph"], ["vector", "bm25"]).choose(
        "Who leads the council's project?"
    )
    assert (route.mode, route.reason, route.model_id) == ("graph", "two hops", "fake/fake-model")
    [system, user] = llm.calls[0]
    assert "Who leads the council's project?" in user.content
    # only the allowed modes are offered, each described once
    assert "vector" in user.content and "graph" in user.content and "bm25" not in user.content
    assert "never instructions" in system.content  # the question is material, not a command


def test_a_mode_outside_the_allowed_ones_is_asked_once_more() -> None:
    replies = iter([_reply("bm25"), _reply("vector")])
    llm = FakeLLM(lambda _: next(replies))
    assert Router(llm, ["vector", "graph"], ["vector", "bm25"]).choose("q").mode == "vector"
    assert len(llm.calls) == 2


def test_still_no_allowed_mode_is_an_error_not_a_default() -> None:
    llm = FakeLLM(_reply("bm25"))
    with pytest.raises(BadReply, match="route"):  # off the per-router schema (simplify S21)
        Router(llm, ["vector", "graph"], ["vector", "bm25"]).choose("q")
    assert len(llm.calls) == 2


def test_a_reply_off_the_schema_is_asked_once_more_then_raises() -> None:
    llm = FakeLLM("not json")
    with pytest.raises(BadReply):
        Router(llm, ["vector", "graph"], ["vector", "bm25"]).choose("q")
    assert len(llm.calls) == 2


def test_a_failed_call_raises_without_a_second_try() -> None:
    class Down(FakeLLM):
        def structured_reply(self, messages: list[Message], schema: type) -> object:  # type: ignore[override]
            self.fake.calls.append(None)  # type: ignore[arg-type]
            raise LLMError("role route (ollama:qwen3:1.7b) failed: APIConnectionError")

    llm = Down()
    with pytest.raises(LLMError, match="APIConnectionError"):
        Router(llm, ["vector", "graph"], ["vector", "bm25"]).choose("q")
    assert len(llm.fake.calls) == 1


def test_the_router_config() -> None:
    assert RouterConfig().modes == ["vector", "graph"]
    with pytest.raises(ValueError, match="two distinct"):
        RouterConfig(modes=["vector", "vector"])
    with pytest.raises(ValueError):
        RouterConfig(modes=["vector", "route"])  # type: ignore[list-item]
    # route mode with graph among its modes needs the graph built
    with pytest.raises(ValueError, match=r"graph\.enabled"):
        load_settings(retrieval={"mode": "route"})
    ok = load_settings(retrieval={"mode": "route", "router": {"modes": ["vector", "bm25"]}})
    assert ok.retrieval.mode == "route"
    assert isinstance(load_settings(graph={"enabled": True}, retrieval={"mode": "route"}), Settings)


def test_the_route_role_defaults_to_the_small_local_model_and_may_be_routed() -> None:
    config = LLMConfig()
    assert config.role_id("route") == "ollama/qwen3:1.7b"
    assert config.target_of("route").reasoning_effort == "none"
    assert ROLES.roles["route"].routable  # not index identity: a router may pick its model
    moved = LLMConfig.model_validate({"backend": "gateway", "model": "gpt-sub"})
    assert moved.role_id("route") == "ollama/qwen3:1.7b"  # answers move, the router stays


def test_an_eval_output_keeps_the_route_as_a_span() -> None:
    from lite_rag.eval import route_span
    from lite_rag.types import Route

    span = route_span(Route("graph", "gateway/gpt-sub", "two hops"))
    assert (span.kind, span.name, span.output, span.attrs) == (
        "route", "graph", {"mode": "graph", "reason": "two hops"}, {"model": "gateway/gpt-sub"},
    )  # fmt: skip
