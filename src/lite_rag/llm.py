"""LLMs (design §6). Every call is lite-model's (its design §12, lite-rag S15): a role —
`answer`, `contextual` or `graph` — maps to a model on a backend (`ollama`, `openai`, `anthropic`,
`gateway`, `litellm`), retries stay on the same model (no fallback chain), and the reply says which
model answered. A failed call raises `LLMError`; its message never carries upstream text.

This module keeps lite-rag's side: `RoleLLM`, one role of a shared lite-model `Model`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import lite_model
from lite_model import Bounded, Message, RolePolicy, RoleSpec
from pydantic import BaseModel

from lite_rag.config import LLMConfig, Role
from lite_rag.errors import ConfigError, LiteRagError
from lite_rag.types import model_id

if TYPE_CHECKING:
    import httpx2

__all__ = ["ROLES", "BadReply", "Completion", "LLMError", "Message", "RoleLLM", "build_model"]

# Only answers may go through a routing alias: contexts and graphs are index identity, and an
# index built by whichever model a router picked could not be rebuilt the same (design §6).
ROLES = RoleSpec({
    "answer": RolePolicy(routable=True),
    "contextual": RolePolicy(routable=False),
    "graph": RolePolicy(routable=False),
})  # fmt: skip


class LLMError(LiteRagError):
    """The model call failed."""

    status = 502


class BadReply(LLMError):
    """The model answered, but not in the shape asked for (a structured reply off its schema)."""


@dataclass(frozen=True)
class Completion:
    text: str
    model_id: str  # backend/model that answered, read from the response — recorded with answers
    usage: dict[str, int] = field(default_factory=dict)
    cached: bool = False  # answered by lite-cache (a hook): no model was called (S18)


def build_model(
    config: LLMConfig, http_client: "httpx2.Client | None" = None
) -> lite_model.LiteModel:
    """One lite-model for every role. The whole config is checked here (an unknown backend, a
    missing key or extra, a routing alias on a pinned role) and refused as ConfigError. Its
    `.hooks` take lite-cache's `CacheHook` when the cache is on (S18)."""
    try:
        return lite_model.build(config, ROLES, http_client=http_client)
    except lite_model.ConfigError as exc:
        raise ConfigError(str(exc)) from exc


class RoleLLM:
    """One role of a lite-model `Model`: what the engine, the contextualizer and the graph
    extractor call."""

    def __init__(self, model: lite_model.Model, role: Role) -> None:
        self._model = model
        self._role = role

    def complete(self, messages: Sequence[Message]) -> Completion:
        try:
            reply = self._model.text(self._role, messages)
        except lite_model.ModelError as exc:
            raise LLMError(str(exc)) from exc
        p, u = reply.provenance, reply.usage
        usage = {k: v for k, v in (("input", u.input), ("output", u.output)) if v is not None}
        # lite-model's Provenance.tag with lite-rag's `/`: answers have recorded `x/y` since S0
        answered = model_id(p.backend, p.answered or f"{p.requested}?")
        return Completion(reply.text, answered, usage, cached=reply.cached)

    def structured[T: BaseModel](self, messages: Sequence[Message], schema: type[T]) -> T:
        """A reply constrained to `schema` (lite-model sends it as a strict JSON schema)."""
        try:
            return self._model.structured(self._role, messages, schema).value
        except lite_model.BadStructuredReply as exc:
            raise BadReply(str(exc)) from exc
        except lite_model.ModelError as exc:
            raise LLMError(str(exc)) from exc

    def bounded(self, concurrency: int) -> "BoundedLLM":
        """The same role with at most `concurrency` calls in flight through it, from any thread
        (one per runtime: it bounds every ingest running at once)."""
        return BoundedLLM(Bounded(self._model, concurrency), self._role)


class BoundedLLM(RoleLLM):
    def __init__(self, model: Bounded, role: Role) -> None:
        super().__init__(model, role)
        self._bounded = model

    def map[T, R](self, one: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """`one` over `items`, in order; up to `concurrency` at once (independent calls). lite-model
        0.3.0 runs each item in a copy of the caller's context, so `call_context()` (lite-cache's
        scope) reaches the hooks of these calls (S18 worked around it here before 0.3.0)."""
        return self._bounded.map(one, items)


# lite-rag's name for an LLM: what the engine and the enhancers take.
LLM = RoleLLM
