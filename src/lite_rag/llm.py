"""LLMs (design §6), switchable by config: `ollama`, `openai` (any OpenAI-compatible endpoint) and
`anthropic`. No fallback chain: a failed call raises `LLMError` (after `max_retries` of the same
model). Every completion says which `provider/model` produced it.

What differs between providers is data in `PROVIDERS`, not branches in the clients.
"""

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from lite_rag.config import OLLAMA_URL, LLMConfig
from lite_rag.errors import LiteRagError
from lite_rag.extras import api_key, require_extra
from lite_rag.types import model_id


class LLMError(LiteRagError):
    """The model call failed."""

    status = 502


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    model_id: str  # provider/model — recorded with every answer
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderProfile:
    base_url: str | None  # None: the SDK's own
    key_env: str | None  # the usual variable; None: no key needed
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"]
    default_temperature: float | None  # sent when none is configured; None: not sent


PROVIDERS: dict[str, ProviderProfile] = {
    # Ollama ignores max_completion_tokens (checked 2026-11-08) and needs no key.
    "ollama": ProviderProfile(f"{OLLAMA_URL}/v1", None, "max_tokens", 0.0),
    # OpenAI's newer models reject max_tokens, and its reasoning models a non-default
    # temperature: none is sent unless configured.
    "openai": ProviderProfile(None, "OPENAI_API_KEY", "max_completion_tokens", None),
    # anthropic 2.6.0's Messages API has no temperature parameter (SDK source, 2026-11-08).
    "anthropic": ProviderProfile(None, "ANTHROPIC_API_KEY", "max_tokens", None),
}


class LLM(Protocol):
    @property
    def model_id(self) -> str: ...

    def complete(self, messages: list[Message]) -> Completion: ...


class BoundedCalls:
    """An LLM shared by every ingest running at once (a server ingests several documents in
    parallel): at most `concurrency` calls in flight in total. Used by the per-chunk enhancers
    (contextual headers, graph extraction)."""

    def __init__(self, llm: LLM, concurrency: int = 1) -> None:
        self._llm = llm
        self._concurrency = concurrency
        self._slots = threading.BoundedSemaphore(concurrency)

    def text(self, messages: list[Message]) -> str:
        with self._slots:
            return self._llm.complete(messages).text

    def map[T, R](self, one: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """`one` over `items`, in order; up to `concurrency` at once (independent calls)."""
        if self._concurrency == 1 or len(items) <= 1:
            return [one(i) for i in items]
        with ThreadPoolExecutor(min(self._concurrency, len(items))) as pool:
            return list(pool.map(one, items))


class _Client:
    """One provider's client: `_send` makes the call; any failure becomes `LLMError`."""

    def __init__(self, config: LLMConfig, client: Any = None) -> None:
        self._config = config
        self._profile = PROVIDERS[config.provider]
        self.model_id = model_id(config.provider, config.model)
        self._client = client if client is not None else self._connect()

    def complete(self, messages: list[Message]) -> Completion:
        try:
            text, usage = self._send(messages)
        except Exception as exc:
            raise LLMError(f"{self.model_id} failed: {type(exc).__name__}: {exc}") from exc
        if not text.strip():  # e.g. the whole token budget went to thinking
            raise LLMError(f"{self.model_id} returned an empty reply (usage {usage})")
        return Completion(text, self.model_id, usage)

    def _key(self) -> str | None:
        section = f"llm ({self._config.provider})"
        return api_key(section, self._config.api_key_env, self._profile.key_env)

    def _timeout(self, sdk: Any) -> Any:
        """The SDK's own Timeout class (openai 3.x and anthropic 2.x run on httpx2, not httpx)."""
        return sdk.Timeout(self._config.timeout_s, connect=self._config.connect_timeout_s)

    def _connect(self) -> Any:
        raise NotImplementedError

    def _send(self, messages: list[Message]) -> tuple[str, dict[str, int]]:
        raise NotImplementedError


class OpenAICompatLLM(_Client):
    """Chat Completions over the OpenAI SDK, for OpenAI and everything that speaks its API."""

    def _connect(self) -> Any:
        import openai

        return openai.OpenAI(
            base_url=self._config.base_url or self._profile.base_url,
            api_key=self._key() or "unused",  # the SDK insists on one; Ollama ignores it
            timeout=self._timeout(openai),
            max_retries=self._config.max_retries,
        )

    def _send(self, messages: list[Message]) -> tuple[str, dict[str, int]]:
        c, profile = self._config, self._profile
        limit = c.max_tokens_param or profile.max_tokens_param
        params: dict[str, Any] = {limit: c.max_output_tokens}
        temperature = c.temperature if c.temperature is not None else profile.default_temperature
        if temperature is not None:
            params["temperature"] = temperature
        if c.reasoning_effort is not None:
            params["reasoning_effort"] = c.reasoning_effort
        response = self._client.chat.completions.create(
            model=c.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **params,
        )
        usage = response.usage
        tokens = {"input": usage.prompt_tokens, "output": usage.completion_tokens} if usage else {}
        return response.choices[0].message.content or "", tokens


class AnthropicLLM(_Client):
    """Anthropic's Messages API (extra `[anthropic]`). The system prompt is a separate field."""

    def _connect(self) -> Any:
        anthropic = require_extra("anthropic", "anthropic")
        return anthropic.Anthropic(
            api_key=self._key(),
            base_url=self._config.base_url,
            timeout=self._timeout(anthropic),
            max_retries=self._config.max_retries,
        )

    def _send(self, messages: list[Message]) -> tuple[str, dict[str, int]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        chat = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        response = self._client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_output_tokens,
            messages=chat,
            **({"system": system} if system else {}),
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, {"input": response.usage.input_tokens, "output": response.usage.output_tokens}


def build_llm(config: LLMConfig) -> LLM:
    return AnthropicLLM(config) if config.provider == "anthropic" else OpenAICompatLLM(config)
