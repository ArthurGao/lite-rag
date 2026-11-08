"""The LLM clients against fake SDK clients: what each provider is sent, what comes back."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from lite_rag.config import LLMConfig
from lite_rag.errors import ConfigError
from lite_rag.llm import AnthropicLLM, LLMError, Message, OpenAICompatLLM, build_llm

MESSAGES = [Message("system", "Be brief."), Message("user", "Hi")]


class FakeOpenAI:
    def __init__(self, fail: bool = False, content: str | None = "Hello [1]") -> None:
        self.sent: dict[str, Any] = {}
        self.fail = fail
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        if self.fail:
            raise TimeoutError("read timed out")
        self.sent = kwargs
        message = SimpleNamespace(content=self.content)
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeAnthropic:
    def __init__(self) -> None:
        self.sent: dict[str, Any] = {}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.sent = kwargs
        blocks = [
            SimpleNamespace(type="thinking", text="(hidden)"),
            SimpleNamespace(type="text", text="Hello "),
            SimpleNamespace(type="text", text="[1]"),
        ]
        return SimpleNamespace(
            content=blocks, usage=SimpleNamespace(input_tokens=9, output_tokens=2)
        )


def test_ollama_sends_max_tokens_and_records_provider_and_model() -> None:
    client = FakeOpenAI()
    llm = OpenAICompatLLM(LLMConfig(provider="ollama", model="qwen3:8b"), client)
    out = llm.complete(MESSAGES)
    assert client.sent["max_tokens"] == 1024 and "max_completion_tokens" not in client.sent
    assert client.sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    assert client.sent["temperature"] == 0.0 and "reasoning_effort" not in client.sent
    assert (out.text, out.model_id, out.usage) == (
        "Hello [1]",
        "ollama/qwen3:8b",
        {"input": 12, "output": 3},
    )


def test_openai_sends_max_completion_tokens_and_no_temperature_unless_set() -> None:
    client = FakeOpenAI()
    cfg = LLMConfig(provider="openai", model="gpt-x", reasoning_effort="low")
    OpenAICompatLLM(cfg, client).complete(MESSAGES)
    assert client.sent["max_completion_tokens"] == 1024 and "max_tokens" not in client.sent
    assert client.sent["reasoning_effort"] == "low" and "temperature" not in client.sent
    OpenAICompatLLM(cfg.model_copy(update={"temperature": 0.3}), client).complete(MESSAGES)
    assert client.sent["temperature"] == 0.3


def test_an_odd_openai_compatible_endpoint_can_take_the_other_token_field() -> None:
    client = FakeOpenAI()
    cfg = LLMConfig(
        provider="openai", model="m", max_tokens_param="max_tokens", max_output_tokens=50
    )
    OpenAICompatLLM(cfg, client).complete(MESSAGES)
    assert client.sent["max_tokens"] == 50 and "max_completion_tokens" not in client.sent


@pytest.mark.parametrize(
    "setting",
    [{"temperature": 0.2}, {"reasoning_effort": "low"}, {"max_tokens_param": "max_tokens"}],
)
def test_settings_anthropic_cannot_honour_are_refused_not_dropped(setting: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="not supported by anthropic"):
        LLMConfig(provider="anthropic", model="claude-x", **setting)


def test_a_reasoning_effort_typo_fails_at_config_load() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(reasoning_effort="nope")  # type: ignore[arg-type]


def test_a_failed_call_raises_with_the_model_named() -> None:
    llm = OpenAICompatLLM(LLMConfig(), FakeOpenAI(fail=True))
    with pytest.raises(LLMError, match="ollama/qwen3:8b failed: TimeoutError"):
        llm.complete(MESSAGES)


def test_anthropic_puts_the_system_prompt_apart_and_joins_text_blocks() -> None:
    client = FakeAnthropic()
    llm = AnthropicLLM(LLMConfig(provider="anthropic", model="claude-x"), client)
    out = llm.complete(MESSAGES)
    assert client.sent["system"] == "Be brief."
    assert client.sent["messages"] == [{"role": "user", "content": "Hi"}]
    assert "temperature" not in client.sent
    assert (out.text, out.model_id, out.usage) == (
        "Hello [1]",
        "anthropic/claude-x",
        {"input": 9, "output": 2},
    )


def test_switching_provider_is_config_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert isinstance(build_llm(LLMConfig(provider="ollama")), OpenAICompatLLM)
    assert isinstance(build_llm(LLMConfig(provider="openai", model="gpt-x")), OpenAICompatLLM)
    assert isinstance(build_llm(LLMConfig(provider="anthropic", model="claude-x")), AnthropicLLM)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_a_paid_provider_without_a_key_fails_at_construction(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="API key"):
        build_llm(LLMConfig(provider=provider, model="m"))  # type: ignore[arg-type]


@pytest.mark.parametrize("content", [None, "", "   "])
def test_an_empty_reply_is_a_failure_not_an_answer(content: str | None) -> None:
    llm = OpenAICompatLLM(LLMConfig(), FakeOpenAI(content=content))
    with pytest.raises(LLMError, match="empty"):
        llm.complete(MESSAGES)


def test_an_anthropic_reply_without_text_blocks_is_a_failure() -> None:
    client = FakeAnthropic()
    client.messages = SimpleNamespace(
        create=lambda **kw: SimpleNamespace(
            content=[SimpleNamespace(type="thinking", text="...")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
    )
    with pytest.raises(LLMError, match="empty"):
        AnthropicLLM(LLMConfig(provider="anthropic", model="c"), client).complete(MESSAGES)
