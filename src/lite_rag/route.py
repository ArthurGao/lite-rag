"""The retrieval router (design §5, S21): the `route` role — a small model, default local qwen3:1.7b
without thinking — reads one question and picks the retrieval mode for it from the ones the
configuration offers. Measured first (eval/route_ceiling.py): on lite-rag's golden sets a
per-question choice could beat the best fixed mode only on multi-hop questions, where the graph
path finds what vector search misses.

No silent fallback (rule 5): a failed call raises `LLMError`; a reply off the schema, or a mode
that was not offered, is asked for once more and then raises `BadReply`."""

import logging
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, create_model

from lite_rag.llm import LLM, BadReply
from lite_rag.prompts import render_pair
from lite_rag.types import Route, RouteMode

log = logging.getLogger(__name__)

TEMPLATES = ("route_system.j2", "route_user.j2")


class RouteReply(BaseModel):
    """What the model is asked for, as a strict JSON schema; each Router narrows `mode` to the
    modes it offers, so a mode that was not offered is off the schema (constrained decoding on
    endpoints that support it)."""

    model_config = ConfigDict(extra="forbid")
    mode: RouteMode
    reason: str


class Router:
    def __init__(self, llm: LLM, modes: Sequence[RouteMode], hybrid_paths: Sequence[str]) -> None:
        self._llm = llm
        self._modes = list(modes)
        self._hybrid_paths = list(hybrid_paths)
        offered: Any = Literal[tuple(self._modes)]
        self._schema: type[RouteReply] = create_model(
            "RouteReply", __base__=RouteReply, mode=(offered, ...)
        )

    def choose(self, question: str) -> Route:
        """A reply off the schema (or a mode not offered) is asked for once more; a failed call
        is not (lite-model already retried it)."""
        messages = render_pair(
            *TEMPLATES, modes=self._modes, hybrid_paths=self._hybrid_paths, question=question
        )
        try:
            reply, model = self._llm.structured_reply(messages, self._schema)
        except BadReply as exc:
            log.warning("route reply off its schema (%s); asking once more", exc)
            try:
                reply, model = self._llm.structured_reply(messages, self._schema)
            except BadReply as again:
                raise BadReply(f"route: {again}") from again
        return Route(reply.mode, model, reply.reason.strip())
