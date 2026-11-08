"""Prompt templates (Jinja2), shipped with the package, and how every prompt is rendered.

Document text always reaches a model inside `<source-NONCE …>` blocks: the nonce is new per
prompt, so a document cannot know it, let alone open or close a block with it.
"""

import hashlib
import re
import secrets
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


def render_pair(system: str, user: str, **values: Any) -> list[Message]:
    """[system, user] messages from two templates, with a fresh block tag (`tag`) for both."""
    env = templates()
    tag = f"source-{secrets.token_hex(6)}"  # new per prompt: document text cannot predict it
    return [
        Message("system", env.get_template(system).render(tag=tag).strip()),
        Message("user", env.get_template(user).render(tag=tag, **values).strip()),
    ]


def _material(text: str, tag: str) -> str:
    """Document text inside a source block. Besides the nonce, `<source` and the tag itself are
    escaped, as a second line of defence."""
    escaped = re.sub(r"<(/?)(source)", r"&lt;\1\2", text, flags=re.IGNORECASE)
    return escaped.replace(f"<{tag}", f"&lt;{tag}").replace(f"</{tag}", f"&lt;/{tag}")


def _attribute(text: str) -> str:
    """A value inside `from="…"`: one line, no quote or tag that could end the attribute."""
    return " ".join(text.split()).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
