"""Answers with validated citations (design §6).

The model sees the hits as <source-NONCE id="1">..<source-NONCE id="k"> blocks — the nonce is new
per prompt, so document text cannot open or close a block — and must cite them as [n]. Code then
checks every citation:
a number that was not given (the model invented a source) is removed from the text and reported in
`dropped`, never passed on (KB rag-production-guide §4 rule 4). With no hits at all the model is
not called: there is nothing to answer from (`status="no_sources"`).
"""

import re
import secrets
from dataclasses import dataclass, field
from functools import cache
from typing import Literal

from jinja2 import Environment, PackageLoader, StrictUndefined

from lite_rag.chunking import chunk_context
from lite_rag.llm import Message
from lite_rag.types import Hit

# [1] · [1, 3] · [1，3] · [1、3], with the spaces before it (removed with an invalid one)
CITATION = re.compile(r"([ \t]*)\[(\s*\d+(?:\s*[,，、]\s*\d+)*\s*)\]")


@dataclass(frozen=True)
class Citation:
    """A cited source, resolved for the caller (a convenience for JSON / UIs: the same data is
    in `Answer.sources[n - 1]`)."""

    n: int  # the number the model used, 1-based into `sources`
    chunk_id: str
    doc_id: str
    title: str


@dataclass(frozen=True)
class Answer:
    """`status`: `answered`, or `no_sources` — nothing was retrieved, the model was not called
    and `text` is empty (callers render that state in their own language). `dropped` lists cited
    numbers that were not sources: removed from `text`, kept here as a quality signal."""

    question: str
    status: Literal["answered", "no_sources"]
    text: str
    citations: list[Citation]
    sources: list[Hit]
    model_id: str | None  # provider/model; None when the model was not called
    usage: dict[str, int] = field(default_factory=dict)
    dropped: list[int] = field(default_factory=list)


@cache
def _templates() -> Environment:
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


def _material(text: str, tag: str) -> str:
    """Document text inside a source block. The block's tag carries a random nonce chosen per
    prompt, so a document cannot know it, let alone open or close a block with it; `<source`
    and the tag itself are escaped as well, as a second line of defence."""
    escaped = re.sub(r"<(/?)(source)", r"&lt;\1\2", text, flags=re.IGNORECASE)
    return escaped.replace(f"<{tag}", f"&lt;{tag}").replace(f"</{tag}", f"&lt;/{tag}")


def _attribute(text: str) -> str:
    """A value inside `from="…"`: one line, no quote or tag that could end the attribute."""
    return " ".join(text.split()).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def prompt(question: str, hits: list[Hit]) -> list[Message]:
    env = _templates()
    tag = f"source-{secrets.token_hex(6)}"  # new per prompt: document text cannot predict it
    system = env.get_template("answer_system.j2").render(tag=tag).strip()
    user = env.get_template("answer_user.j2").render(question=question, hits=hits, tag=tag)
    return [Message("system", system), Message("user", user)]


def check_citations(text: str, hits: list[Hit]) -> tuple[str, list[Citation], list[int]]:
    """The text with invalid citation numbers removed, the valid citations in first-use order,
    and the numbers that were dropped. Only citation markers are touched."""
    cited: dict[int, Citation] = {}
    dropped: list[int] = []

    def keep_valid(match: re.Match[str]) -> str:
        if all(1 <= int(x) <= len(hits) for x in re.findall(r"\d+", match.group(2))):
            for n in (int(x) for x in re.findall(r"\d+", match.group(2))):
                h = hits[n - 1]
                cited.setdefault(n, Citation(n, h.chunk_id, h.doc_id, h.title))
            return match.group(0)  # all valid: left exactly as the model wrote it
        kept = []
        for n in (int(x) for x in re.findall(r"\d+", match.group(2))):
            if 1 <= n <= len(hits):
                kept.append(f"[{n}]")
                h = hits[n - 1]
                cited.setdefault(n, Citation(n, h.chunk_id, h.doc_id, h.title))
            else:
                dropped.append(n)
        return match.group(1) + "".join(kept) if kept else ""

    return CITATION.sub(keep_valid, text).strip(), list(cited.values()), dropped
