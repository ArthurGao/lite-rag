"""Answers with validated citations (design §6).

The model sees the hits as <source-NONCE id="1">..<source-NONCE id="k"> blocks — the nonce is new
per prompt, so document text cannot open or close a block — and must cite them as [n]. Code then
checks every citation:
a number that was not given (the model invented a source) is removed from the text and reported in
`dropped`, never passed on (KB rag-production-guide §4 rule 4). With no hits at all the model is
not called: there is nothing to answer from (`status="no_sources"`).
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from lite_rag.llm import Message
from lite_rag.prompts import render_pair
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


def prompt(question: str, hits: list[Hit]) -> list[Message]:
    return render_pair("answer_system.j2", "answer_user.j2", question=question, hits=hits)


def check_citations(text: str, hits: list[Hit]) -> tuple[str, list[Citation], list[int]]:
    """The text with invalid citation numbers removed, the valid citations in first-use order,
    and the numbers that were dropped. Only citation markers are touched."""
    cited: dict[int, Citation] = {}
    dropped: list[int] = []

    def keep_valid(match: re.Match[str]) -> str:
        numbers = [int(x) for x in re.findall(r"\d+", match.group(2))]
        valid = [n for n in numbers if 1 <= n <= len(hits)]
        for n in valid:
            h = hits[n - 1]
            cited.setdefault(n, Citation(n, h.chunk_id, h.doc_id, h.title))
        if len(valid) == len(numbers):
            return match.group(0)  # all valid: left exactly as the model wrote it
        dropped.extend(n for n in numbers if n not in valid)
        return match.group(1) + "".join(f"[{n}]" for n in valid) if valid else ""

    return CITATION.sub(keep_valid, text).strip(), list(cited.values()), dropped
