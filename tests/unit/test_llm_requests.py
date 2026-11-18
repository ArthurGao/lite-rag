"""The request bodies lite-rag sends for each role (lite-model design §12.2 step 1, lite-rag S15).

`tests/fixtures/llm_requests.json` was captured from the client lite-rag had before it moved onto
lite-model (lite-rag d007040); the client it has now must send the same JSON. Only `_llm` knows
how a role's client is built."""

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from lite_model import strict_schema

from lite_rag.config import Settings
from lite_rag.graph import GraphReply
from lite_rag.llm import LLM, Message, RoleLLM, build_model

RECORDED = Path(__file__).parents[1] / "fixtures" / "llm_requests.json"

MESSAGES = [
    Message("system", "只根据下面的资料回答，并标出处 [n]。"),
    Message("user", "资料：\n[1] 咖啡粉和水的比例是 1:16。\n\n问题：手冲咖啡的粉水比是多少？"),
]

# name -> (role, the settings that configure it). The vendor keeps the backend name `openai` (only
# its endpoint and key change), so its id stays `openai/<model>` as before S15.
CASES: dict[str, tuple[str, dict[str, Any]]] = {
    # the defaults: every role on the local Ollama
    "answer": ("answer", {}),
    "contextual": ("contextual", {}),
    "graph": ("graph", {}),
    # an OpenAI-compatible vendor with every parameter set
    "vendor": ("answer", {"llm": {
        "backend": "openai", "model": "vendor-model", "temperature": 0.3,
        "reasoning_effort": "low", "max_output_tokens": 512,
        "backends": {"openai": {"base_url": "http://vendor.test/v1",
                                "api_key_env": "LITERAG_TEST_VENDOR_KEY"}},
    }}),
    "anthropic": ("answer", {"llm": {
        "backend": "anthropic", "model": "claude-x", "max_output_tokens": 700,
        "backends": {"anthropic": {"base_url": "http://anthropic.test",
                                   "api_key_env": "LITERAG_TEST_ANTHROPIC_KEY"}},
    }}),
}  # fmt: skip


def _reply(request: httpx2.Request) -> httpx2.Response:
    if request.url.path.endswith("/messages"):
        return httpx2.Response(200, json={
            "id": "m", "type": "message", "role": "assistant", "model": "claude-x",
            "stop_reason": "end_turn", "stop_sequence": None,
            "content": [{"type": "text", "text": "1:16 [1]"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })  # fmt: skip
    structured = "response_format" in json.loads(request.content)
    content = '{"entities": [], "relations": []}' if structured else "1:16 [1]"
    return httpx2.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 0, "model": "qwen3:8b",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })  # fmt: skip


def _llm(case: str, http: httpx2.Client) -> LLM:
    """The client a role gets, built the way lite-rag builds it, over a recording transport."""
    role, data = CASES[case]
    settings = Settings.model_validate(data)
    return RoleLLM(build_model(settings.llm, http_client=http), role)  # type: ignore[arg-type]


def record(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Every case: what reached the wire."""
    monkeypatch.setenv("LITERAG_TEST_VENDOR_KEY", "vendor-key")
    monkeypatch.setenv("LITERAG_TEST_ANTHROPIC_KEY", "anthropic-key")
    seen: dict[str, Any] = {}
    for case in CASES:
        sent: list[httpx2.Request] = []

        def handler(request: httpx2.Request, sent: list[httpx2.Request] = sent) -> httpx2.Response:
            sent.append(request)
            return _reply(request)

        llm = _llm(case, httpx2.Client(transport=httpx2.MockTransport(handler)))
        if case == "graph":  # the graph's reply is constrained to its schema (S16)
            assert llm.structured(MESSAGES, GraphReply) == GraphReply(entities=[], relations=[])
        else:
            assert llm.complete(MESSAGES).text == "1:16 [1]"
        [request] = sent  # one request per call: no retry, no second call
        seen[case] = {
            "method": request.method,
            "url": str(request.url),
            "auth": request.headers.get("authorization") or request.headers.get("x-api-key"),
            "body": json.loads(request.content),
        }
    return seen


def test_every_role_sends_the_recorded_request(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))
    # S16 asks the graph role for a reply held to a JSON schema: the one change since the recording
    schema = {"name": "GraphReply", "schema": strict_schema(GraphReply), "strict": True}
    recorded["graph"]["body"]["response_format"] = {"type": "json_schema", "json_schema": schema}
    seen = record(monkeypatch)
    assert sorted(seen) == sorted(recorded)
    for case in recorded:
        assert seen[case] == recorded[case], case
