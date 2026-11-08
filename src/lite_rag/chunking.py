"""Chunkers (design §3.1). Every strategy works the same way and differs only in where it may cut:

1. split the text into contiguous *pieces* at the strategy's boundaries (separators stay with the
   piece before them, so the pieces cover the text exactly);
2. a piece still larger than `size` is split again at finer boundaries (recursively), and as a last
   resort at token boundaries;
3. pieces are packed into chunks of at most `size` units; a new chunk repeats up to `overlap` units
   of whole pieces from the end of the previous one; a last chunk below `min_size` is rebalanced
   with the one before it. `size` is a hard limit: no step ever produces a larger chunk.

Offsets are kept throughout, so `doc.text[chunk.start:chunk.end] == chunk.text` always holds.
"""

import re
from collections.abc import Callable
from itertools import pairwise
from typing import Protocol

from lite_rag.config import ChunkingConfig
from lite_rag.text import TokenCounter
from lite_rag.types import Chunk, Document, chunk_id

Region = tuple[int, int]

# Cut points, coarse to fine. Each separator stays with the piece before it.
HEADING = re.compile(r"\n(?=#{1,6}[ \t])")
BLANK_LINE = re.compile(r"\n[ \t]*\n")
LINE = re.compile(r"\n")
SENTENCE = re.compile(r"(?<=[。！？!?])|(?<=[.;；])(?=\s)")
CLAUSE = re.compile(r"(?<=[，,、：:；;])")
SPACE = re.compile(r"(?<= )")
RECURSIVE = (HEADING, BLANK_LINE, LINE, SENTENCE, CLAUSE, SPACE)
ATX = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]: ...


Piece = tuple[Region, int]  # a region and its size in units, counted once
Section = tuple[Region, tuple[str, ...]]  # a region chunks never cross, and its heading path


def split_at(text: str, region: Region, sep: re.Pattern[str]) -> list[Region]:
    """Contiguous pieces of `region`, cut after each match of `sep`."""
    start, end = region
    cuts = [m.end() for m in sep.finditer(text, start, end) if start < m.end() < end]
    bounds = [start, *dict.fromkeys(cuts), end]
    return [(a, b) for a, b in pairwise(bounds) if b > a]


class _Base:
    def __init__(self, config: ChunkingConfig, counter: TokenCounter) -> None:
        self._config = config
        self._counter = counter

    def split(self, doc: Document) -> list[Chunk]:
        regions = [
            (r, path)
            for section, path in self._sections(doc.text)
            for r in self._chunk_region(doc.text, section)
        ]
        return self._to_chunks(doc, regions)

    # --- hooks -------------------------------------------------------------------------------

    def _sections(self, text: str) -> list[Section]:
        """Independent parts of the document; chunks never span two. Default: the whole text."""
        return [((0, len(text)), ())]

    def _pieces(self, text: str, region: Region) -> list[Region]:
        """Where this strategy may cut a section. Default: nowhere (cut only to fit)."""
        return [region]

    # --- shared machinery --------------------------------------------------------------------

    def _chunk_region(self, text: str, region: Region) -> list[Region]:
        fitted = [p for piece in self._pieces(text, region) for p in self._fit(text, piece, 0)]
        return self._pack(fitted)

    def _count(self, text: str, region: Region) -> int:
        return self._counter.count(text[region[0] : region[1]])

    def _fit(self, text: str, region: Region, level: int, count: int | None = None) -> list[Piece]:
        """`region` cut until every piece is at most `size` units; each piece with its count."""
        n = self._count(text, region) if count is None else count
        if n <= self._config.size:
            return [(region, n)]
        if level >= len(RECURSIVE):
            windows = self._token_windows(text, region, self._config.size, 0)
            return [(w, self._count(text, w)) for w in windows]
        pieces = split_at(text, region, RECURSIVE[level])
        if len(pieces) == 1:  # no cut at this level: same region, same count, finer separator
            return self._fit(text, region, level + 1, n)
        return [p for piece in pieces for p in self._fit(text, piece, level + 1)]

    def _token_windows(self, text: str, region: Region, size: int, overlap: int) -> list[Region]:
        start, end = region
        spans = [(start + a, start + b) for a, b in self._counter.spans(text[start:end])]
        if not spans:
            return [region]
        step = max(1, size - overlap)
        out = []
        for i in range(0, len(spans), step):
            window = spans[i : i + size]
            out.append((window[0][0], window[-1][1]))
            if i + size >= len(spans):
                break
        return out

    def _pack(self, pieces: list[Piece]) -> list[Region]:
        size, counts = self._config.size, [n for _, n in pieces]
        chunks: list[list[int]] = []  # indexes into pieces
        current: list[int] = []
        total = 0
        for i, n in enumerate(counts):
            if current and total + n > size:
                chunks.append(current)
                current = self._overlap_tail(current, counts)
                total = sum(counts[j] for j in current)
                if current and total + n > size:  # the overlap alone leaves no room
                    current, total = [], 0
            current.append(i)
            total += n
        if current:
            chunks.append(current)
            if len(chunks) > 1 and total < self._config.min_size:
                self._rebalance_tail(chunks, counts)
        return [(pieces[c[0]][0][0], pieces[c[-1]][0][1]) for c in chunks]

    def _rebalance_tail(self, chunks: list[list[int]], counts: list[int]) -> None:
        """The last chunk is below `min_size`: re-split the pieces of the last two chunks into
        the most even pair that keeps both within `size` and the tail at least `min_size`.
        `size` stays a hard limit, so when no such split exists the chunks stay as they are."""
        first, last = chunks[-2][0], chunks[-1][-1]
        span = list(range(first, last + 1))
        best: tuple[int, int] | None = None  # (imbalance, cut)
        for cut in range(1, len(span)):
            left = sum(counts[j] for j in span[:cut])
            right = sum(counts[j] for j in span[cut:])
            if left <= self._config.size and self._config.min_size <= right <= self._config.size:
                if best is None or abs(left - right) < best[0]:
                    best = (abs(left - right), cut)
        if best is not None:
            chunks[-2:] = [span[: best[1]], span[best[1] :]]

    def _overlap_tail(self, chunk: list[int], counts: list[int]) -> list[int]:
        tail: list[int] = []
        total = 0
        for j in reversed(chunk[1:]):  # never the whole chunk: each chunk must move forward
            if total + counts[j] > self._config.overlap:
                break
            tail.insert(0, j)
            total += counts[j]
        return tail

    def _to_chunks(self, doc: Document, regions: list[Section]) -> list[Chunk]:
        out: list[Chunk] = []
        for (start, end), path in regions:
            raw = doc.text[start:end]
            body = raw.strip()
            if not body:
                continue
            s = start + len(raw) - len(raw.lstrip())
            n = len(out)
            tokens = self._counter.count(body)  # exact for the final chunk text
            out.append(
                Chunk(chunk_id(doc.doc_id, n), doc.doc_id, n, body, s, s + len(body), tokens, path)
            )
        return out


class FixedChunker(_Base):
    """Windows of exactly `size` units, `overlap` units apart, ignoring structure (the baseline)."""

    def _chunk_region(self, text: str, region: Region) -> list[Region]:
        return self._token_windows(text, region, self._config.size, self._config.overlap)


class RecursiveChunker(_Base):
    """Cut at the coarsest boundary that fits: heading, blank line, line, sentence, clause,
    space."""


class SentenceChunker(_Base):
    """Pack whole sentences (Chinese and English end marks, and line ends)."""

    def _pieces(self, text: str, region: Region) -> list[Region]:
        return [s for line in split_at(text, region, LINE) for s in split_at(text, line, SENTENCE)]


class ParagraphChunker(_Base):
    """Pack whole short paragraphs (blank-line separated). A paragraph larger than `size` is a
    section of its own, so its pieces never join a neighbouring paragraph."""

    def _sections(self, text: str) -> list[Section]:
        sections: list[Section] = []
        run: list[Region] = []  # consecutive short paragraphs
        for paragraph in split_at(text, (0, len(text)), BLANK_LINE):
            if self._count(text, paragraph) <= self._config.size:
                run.append(paragraph)
                continue
            if run:
                sections.append(((run[0][0], run[-1][1]), ()))
                run = []
            sections.append((paragraph, ()))
        if run:
            sections.append(((run[0][0], run[-1][1]), ()))
        return sections

    def _pieces(self, text: str, region: Region) -> list[Region]:
        return split_at(text, region, BLANK_LINE)


class MarkdownChunker(RecursiveChunker):
    """One section per ATX heading; chunks never cross a heading and carry its heading path."""

    def _sections(self, text: str) -> list[Section]:
        headings = list(ATX.finditer(text))
        if not headings:
            return [((0, len(text)), ())]
        sections: list[Section] = []
        if headings[0].start() > 0:
            sections.append(((0, headings[0].start()), ()))
        stack: list[tuple[int, str]] = []
        for i, h in enumerate(headings):
            level, title = len(h.group(1)), h.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            sections.append(((h.start(), end), tuple(t for _, t in stack)))
        return sections


STRATEGIES: dict[str, Callable[[ChunkingConfig, TokenCounter], Chunker]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
    "paragraph": ParagraphChunker,
    "markdown": MarkdownChunker,
}


def build_chunker(config: ChunkingConfig, counter: TokenCounter) -> Chunker:
    return STRATEGIES[config.strategy](config, counter)


def embedding_text(chunk: Chunk) -> str:
    """What gets embedded and indexed: the heading path (if any) gives the chunk its context."""
    if not chunk.heading_path:
        return chunk.text
    return " > ".join(chunk.heading_path) + "\n" + chunk.text
