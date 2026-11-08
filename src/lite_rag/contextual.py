"""Contextual chunk headers (design §3.1): an LLM writes one or two sentences that place each
chunk in its document; they are prepended to what is embedded and BM25-indexed — never to the
text a hit returns. One LLM call per chunk, at ingest; a failed call fails the upsert (nothing
is written, no fallback model)."""

from lite_rag.llm import LLM, Message
from lite_rag.prompts import digest, render_pair
from lite_rag.text import TokenCounter
from lite_rag.types import Chunk, Document

TEMPLATES = ("contextual_system.j2", "contextual_user.j2")


# Bump when the code that shapes a context changes (`window`, the reply clean-up): stored
# contexts then no longer match, and the kb identity says so.
CODE_VERSION = "1"


def prompt_digest() -> str:
    return f"{digest(*TEMPLATES)}.{CODE_VERSION}"


def window(text: str, chunk: Chunk, total: int, max_tokens: int) -> str:
    """The document (`total` tokens), or — when it is over `max_tokens` — the part of it centred
    on the chunk that fits (in characters, scaled by the document's characters per token)."""
    if total <= max_tokens:
        return text
    chars = max(chunk.end - chunk.start, int(len(text) * max_tokens / total))
    middle = (chunk.start + chunk.end) // 2
    start = max(0, min(middle - chars // 2, len(text) - chars))
    return text[start : start + chars]


def context_prompt(doc: Document, chunk: Chunk, shown: str) -> list[Message]:
    return render_pair(*TEMPLATES, title=doc.title, document=shown, passage=chunk.text)


class Contextualizer:
    def __init__(self, llm: LLM, counter: TokenCounter, max_doc_tokens: int) -> None:
        self._llm = llm
        self._counter = counter
        self._max = max_doc_tokens

    def contexts(self, doc: Document, chunks: list[Chunk]) -> list[str]:
        total = self._counter.count(doc.text)  # once per document, not per chunk
        out = []
        for chunk in chunks:
            shown = window(doc.text, chunk, total, self._max)
            reply = self._llm.complete(context_prompt(doc, chunk, shown)).text
            out.append(" ".join(reply.split()))
        return out
