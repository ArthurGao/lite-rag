"""lite-rag's side of the LLM (S15): `RoleLLM` over lite-model. What reaches the wire is pinned
in test_llm_requests.py; here, what comes back and what is refused. Requests go to a recording
transport, never a network."""

from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from lite_rag.config import LLMConfig, load_settings
from lite_rag.errors import ConfigError
from lite_rag.llm import LLMError, Message, RoleLLM, build_model

MESSAGES = [Message("system", "Be brief."), Message("user", "Hi")]


def _reply(model: str, content: str | None, finish: str = "stop") -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    })  # fmt: skip


def _role(handler: Any, role: str = "answer", **llm: Any) -> RoleLLM:
    config = LLMConfig.model_validate({"max_retries": 0, **llm})
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return RoleLLM(build_model(config, http_client=http), role)  # type: ignore[arg-type]


def test_the_answer_names_the_model_that_answered_and_its_tokens() -> None:
    out = _role(lambda r: _reply("qwen3:8b-q4", "Hello [1]")).complete(MESSAGES)
    # read from the response, not the configured qwen3:8b
    assert (out.text, out.model_id, out.usage) == (
        "Hello [1]", "ollama/qwen3:8b-q4", {"input": 12, "output": 3}
    )  # fmt: skip


@pytest.mark.parametrize(
    "reply",
    [
        _reply("m", "cut off mid", finish="length"),  # S15: a cut answer is not an answer
        _reply("m", ""),  # e.g. the whole budget went to thinking
        _reply("m", None),
        httpx2.Response(500, json={"error": "boom"}),
    ],
)
def test_an_unusable_reply_is_an_llm_error(reply: httpx2.Response) -> None:
    with pytest.raises(LLMError) as caught:
        _role(lambda r: reply).complete(MESSAGES)
    assert caught.value.status == 502 and "role answer (ollama:qwen3:8b)" in str(caught.value)


def test_an_llm_error_never_carries_upstream_text() -> None:
    leaky = "401 for https://u:pw@proxy/v1 with Authorization: Bearer sk-secret-123"

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(leaky, request=request)

    with pytest.raises(LLMError) as caught:
        _role(handler).complete(MESSAGES)
    assert "sk-secret" not in str(caught.value) and "pw@" not in str(caught.value)
    assert caught.value.__cause__ is not None  # kept for debugging


def test_only_answers_may_use_a_routing_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    # contexts and graphs are index identity: a router's pick could not be rebuilt the same
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    routed = {"backend": "litellm", "model": "smart-router"}
    build_model(LLMConfig.model_validate({"roles": {"answer": routed}}))
    for role in ("contextual", "graph"):
        with pytest.raises(ConfigError, match="routing alias and the role is not routable"):
            build_model(LLMConfig.model_validate({"roles": {role: routed}}))


@pytest.mark.parametrize(
    ("llm", "problem"),
    [
        ({"backend": "openai", "model": "gpt-x"}, "set OPENAI_API_KEY"),
        # anthropic takes no temperature: refused, never silently dropped
        (
            {"backend": "anthropic", "model": "c", "temperature": 0.2},
            "temperature is not supported",
        ),
        ({"backend": "nope"}, "unknown backend"),
    ],
)
def test_a_config_a_backend_cannot_honour_fails_before_any_call(
    monkeypatch: pytest.MonkeyPatch, llm: dict[str, Any], problem: str
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(ConfigError, match=problem):
        build_model(load_settings(llm=llm).llm)


def test_a_reasoning_effort_typo_fails_at_config_load() -> None:
    with pytest.raises(ValidationError, match="reasoning_effort"):
        load_settings(llm={"reasoning_effort": "hihg"})
