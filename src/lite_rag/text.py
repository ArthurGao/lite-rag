"""Text units: BM25 terms (jieba) and token counting for chunk sizes (design §3.1, §5).

Two different things on purpose:
- `terms` feeds the keyword index — words, lowercased, punctuation dropped;
- a `TokenCounter` measures chunk size in the embedding model's own tokens, because sizing by words
  misjudges what the model sees (KB rag-production-guide §3). Without a tokenizer the size is an
  estimate (the counter's identity says `estimate`).
"""

import logging
import re
from functools import cache, cached_property
from typing import Any, Protocol

from lite_rag.config import ChunkingConfig, EmbeddingConfig
from lite_rag.model_profiles import profile_for

DEFAULT_STOPWORDS = frozenset(
    "的 了 和 是 在 也 就 都 而 及 与 或 一个 这 那 有 我 你 他 她 它 我们 吗 呢 吧 啊 "
    "a an the and or of to in on at for is are was were be been it this that with as by".split()
)
_WORDISH = re.compile(r"\w", re.UNICODE)
# Estimate: one token per CJK character, per up to 6 letters/digits (a long unbroken run is
# several tokens, as in a real subword tokenizer), per other visible character.
ESTIMATE_TOKEN = re.compile(
    r"[\u3400-\u9fff\uf900-\ufaff]|[^\W_\u3400-\u9fff\uf900-\ufaff]{1,6}|[^\s\w]", re.UNICODE
)


@cache
def _jieba() -> Any:
    import jieba

    jieba.setLogLevel(logging.WARNING)
    return jieba


def terms(text: str, stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> list[str]:
    """BM25 terms: jieba words (Chinese) and lowercased words (other scripts), no punctuation."""
    out: list[str] = []
    for word in _jieba().lcut(text.lower()):
        word = word.strip()
        if word and _WORDISH.search(word) and word not in stopwords:
            out.append(word)
    return out


Span = tuple[int, int]


class TokenCounter(Protocol):
    @property
    def identity(self) -> str:
        """What sizes the chunks — recorded on a kb: "chars", "estimate" or "hf:<tokenizer>"."""
        ...

    def spans(self, text: str) -> list[Span]:
        """Character offsets of each token, in order."""
        ...

    def count(self, text: str) -> int: ...


class EstimateCounter:
    """No tokenizer: CJK characters, words and symbols each count as one token."""

    identity = "estimate"

    def spans(self, text: str) -> list[Span]:
        return [m.span() for m in ESTIMATE_TOKEN.finditer(text)]

    def count(self, text: str) -> int:
        return sum(1 for _ in ESTIMATE_TOKEN.finditer(text))


class HFTokenCounter:
    """The embedding model's own tokenizer (Hugging Face `tokenizers`, no torch). Downloads the
    tokenizer file once on first use."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.identity = f"hf:{name}"

    @cached_property
    def _tokenizer(self) -> Any:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(self.name)
        # Some tokenizer files ship with truncation on (MiniLM: 128 tokens); measuring and
        # cutting chunks needs every token, or the text past the limit is silently dropped.
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return tokenizer

    def spans(self, text: str) -> list[Span]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return [(s, e) for s, e in encoding.offsets if e > s]

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class CharCounter:
    """`unit: chars`: every character is one unit."""

    identity = "chars"

    def spans(self, text: str) -> list[Span]:
        return [(i, i + 1) for i in range(len(text))]

    def count(self, text: str) -> int:
        return len(text)


def build_counter(chunking: ChunkingConfig, embedding: EmbeddingConfig) -> TokenCounter:
    """Chunks are sized in characters, or in the embedding model's tokens (its profile's
    tokenizer unless configured; an estimate when there is none)."""
    if chunking.unit == "chars":
        return CharCounter()
    tokenizer = profile_for(embedding.model, tokenizer=embedding.tokenizer).tokenizer
    return HFTokenCounter(tokenizer) if tokenizer else EstimateCounter()
