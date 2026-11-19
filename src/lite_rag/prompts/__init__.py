"""Prompt templates (Jinja2), shipped with the package, and how every prompt is rendered.

Document text always reaches a model inside `<source-NONCE …>` blocks: a document cannot know
the nonce, let alone open or close a block with it. It is random per prompt, except inside
`keyed_nonces(key)` (a call lite-cache may answer, S18): there it is an HMAC of the prompt under
that secret key, so the same prompt is the same request (and the same cache key) while a
document, without the key, still cannot predict it.
"""

import hashlib
import hmac
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from importlib.resources import files
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined

from lite_rag.chunking import chunk_context
from lite_rag.llm import Message


def digest(*names: str) -> str:
    """A short hash of these templates: recorded wherever their output is stored, so a changed
    prompt is a changed index."""
    h = hashlib.sha256()
    for name in names:
        h.update(files(__name__).joinpath(name).read_bytes())
    return h.hexdigest()[:12]


@cache
def templates() -> Environment:
    env = Environment(
        loader=PackageLoader("lite_rag", "prompts"),
        undefined=StrictUndefined,
        autoescape=False,  # plain text for a model, not HTML
        auto_reload=False,  # templates ship with the package
    )
    env.globals["context"] = lambda hit: chunk_context(hit.title, hit.heading_path)
    env.filters["material"] = _material
    env.filters["attribute"] = _attribute
    return env


_NONCE_KEY: ContextVar[bytes | None] = ContextVar("lite_rag_nonce_key", default=None)
# What the HMAC sees where the tag goes: the prompt's shape does not depend on the per-render
# placeholder below. Text that contains this marker only changes the hash input, never the tag.
_TAG_MARK = "\x00TAG\x00"


@contextmanager
def keyed_nonces(key: bytes) -> Iterator[None]:
    """Prompts rendered in this block (and in threads that copy its context) get nonces derived
    from `key` and their own content instead of random ones."""
    token = _NONCE_KEY.set(key)
    try:
        yield
    finally:
        _NONCE_KEY.reset(token)


def render_pair(system: str, user: str, **values: Any) -> list[Message]:
    """[system, user] messages from two templates, with one block tag (`tag`) for both."""
    env = templates()

    def pair(tag: str) -> list[Message]:
        return [
            Message("system", env.get_template(system).render(tag=tag).strip()),
            Message("user", env.get_template(user).render(tag=tag, **values).strip()),
        ]

    key = _NONCE_KEY.get()
    if key is None:
        return pair(f"source-{secrets.token_hex(6)}")  # new per prompt
    # Rendered once with a placeholder nobody can predict (random per render), so no rendered text
    # (a question is not escaped like document text) can contain it. The HMAC, under a key
    # documents do not know, of the prompt with the placeholder normalised is the tag: the same
    # for the same prompt, unpredictable without the key. Only the placeholder becomes the tag.
    placeholder = f"source-{secrets.token_hex(16)}"
    shaped = pair(placeholder)
    shape = "\n".join(m.content.replace(placeholder, _TAG_MARK) for m in shaped)
    tag = "source-" + hmac.new(key, shape.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
    return [Message(m.role, m.content.replace(placeholder, tag)) for m in shaped]


def _material(text: str, tag: str) -> str:
    """Document text inside a source block. Besides the nonce, `<source` and the tag itself are
    escaped, as a second line of defence."""
    escaped = re.sub(r"<(/?)(source)", r"&lt;\1\2", text, flags=re.IGNORECASE)
    return escaped.replace(f"<{tag}", f"&lt;{tag}").replace(f"</{tag}", f"&lt;/{tag}")


def _attribute(text: str) -> str:
    """A value inside `from="…"`: one line, no quote or tag that could end the attribute."""
    return " ".join(text.split()).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
