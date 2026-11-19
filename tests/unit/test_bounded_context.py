"""`BoundedLLM.map` runs each item in the caller's context (S18): lite-model 0.2.1's
`Bounded.map` uses plain worker threads, which do not inherit contextvars, so `call_context()`
(lite-cache's scope and policy) would not reach the hooks of calls made in them."""

from typing import Any

import httpx2
import pytest
from lite_model import call_context
from lite_model.hooks import Call

from lite_rag.config import LLMConfig
from lite_rag.llm import Message, RoleLLM, build_model
from tests.chat import chat_completion


def _ok(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=chat_completion("qwen3:1.7b", "ctx", usage=(1, 1)))


class Seen:
    name = "seen"

    def __init__(self) -> None:
        self.scopes: list[Any] = []

    def before(self, call: Call) -> None:
        self.scopes.append(call.context.get("scope"))

    def after(self, call: Call, reply: Any) -> None: ...


@pytest.mark.parametrize("concurrency", [1, 4])
def test_call_context_reaches_hooks_in_map_workers(concurrency: int) -> None:
    model = build_model(LLMConfig(), http_client=httpx2.Client(transport=httpx2.MockTransport(_ok)))
    seen = Seen()
    model.hooks.register(seen)
    bounded = RoleLLM(model, "contextual").bounded(concurrency)

    def one(n: int) -> str:
        return bounded.complete([Message("user", f"chunk {n}")]).text

    with call_context(scope="literag/ns", cache=True):
        out = bounded.map(one, list(range(8)))
    assert out == ["ctx"] * 8
    assert seen.scopes == ["literag/ns"] * 8
