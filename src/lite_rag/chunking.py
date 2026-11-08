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

import math
import re
from bisect import bisect_right
from collections.abc import Callable
from itertools import pairwise
from typing import Protocol

from lite_rag.config import ChunkingConfig
from lite_rag.errors import ConfigError
from lite_rag.loaders import ATX
from lite_rag.text import TokenCounter
from lite_rag.types import Chunk, Document, chunk_id

Region = tuple[int, int]
Embed = Callable[[list[str]], list[list[float]]]

# Cut points, coarse to fine. Each separator stays with the piece before it.
HEADING = re.compile(r"\n(?=#{1,6}[ \t])")
BLANK_LINE = re.compile(r"\n[ \t]*\n")
LINE = re.compile(r"\n")
SENTENCE = re.compile(r"(?<=[。！？!?])|(?<=[.;；])(?=\s)")
CLAUSE = re.compile(r"(?<=[，,、：:；;])")
SPACE = re.compile(r"(?<= )")
RECURSIVE = (HEADING, BLANK_LINE, LINE, SENTENCE, CLAUSE, SPACE)


class Chunker(Protocol):
    # Indexed chunks per parent, at most (1: no parents): how much deeper a search must look
    # to find k distinct parents.
    children_per_parent: int

    def split(self, doc: Document) -> list[Chunk]:
        """The chunks that are embedded and indexed."""
        ...

    def split_parents(self, doc: Document) -> tuple[list[Chunk], list[Chunk]]:
        """(parents, chunks): the larger parents a hit returns — `parent_child` only, otherwise
        none — and the indexed chunks, each naming its parent's `n`."""
        ...


Piece = tuple[Region, int]  # a region and its size in units, counted once
Section = tuple[Region, tuple[str, ...]]  # a region chunks never cross, and its heading path


def split_at(text: str, region: Region, sep: re.Pattern[str]) -> list[Region]:
    """Contiguous pieces of `region`, cut after each match of `sep`."""
    start, end = region
    cuts = [m.end() for m in sep.finditer(text, start, end) if start < m.end() < end]
    bounds = [start, *dict.fromkeys(cuts), end]
    return [(a, b) for a, b in pairwise(bounds) if b > a]


class _Base:
    children_per_parent = 1

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

    def split_parents(self, doc: Document) -> tuple[list[Chunk], list[Chunk]]:
        return [], self.split(doc)

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

    def _to_chunks(
        self, doc: Document, regions: list[Section], parents: list[int] | None = None
    ) -> list[Chunk]:
        """Chunks from regions; `parents[i]`, when given, is the parent of region i."""
        out: list[Chunk] = []
        for i, ((start, end), path) in enumerate(regions):
            raw = doc.text[start:end]
            body = raw.strip()
            if not body:
                continue
            s = start + len(raw) - len(raw.lstrip())
            n = len(out)
            tokens = self._counter.count(body)  # exact for the final chunk text
            page = bisect_right(doc.pages, s) if doc.pages else None
            stop = s + len(body)
            parent = parents[i] if parents is not None else None
            out.append(
                Chunk(
                    chunk_id(doc.doc_id, n),
                    doc.doc_id,
                    n,
                    body,
                    s,
                    stop,
                    tokens,
                    path,
                    page,
                    parent,
                )
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


class ParentChildChunker(RecursiveChunker):
    """Small chunks for precise matching, each inside one larger parent that a hit returns for
    context. Parents: the markdown strategy (never across a heading) at `parent_size`, no
    overlap. Children: the recursive strategy at `size` / `overlap`, cut inside each parent, so
    a child never crosses its parent and inherits its heading path."""

    def __init__(self, config: ChunkingConfig, counter: TokenCounter) -> None:
        super().__init__(config, counter)
        size = config.effective_parent_size
        parent_config = config.model_copy(update={"size": size, "overlap": 0})
        step = max(1, config.size - config.overlap)  # overlapping children: more per parent
        self.children_per_parent = math.ceil(size / step)
        self._parent_chunker = MarkdownChunker(parent_config, counter)

    def split(self, doc: Document) -> list[Chunk]:
        return self.split_parents(doc)[1]

    def split_parents(self, doc: Document) -> tuple[list[Chunk], list[Chunk]]:
        parents = self._parent_chunker.split(doc)
        regions: list[Section] = []
        owners: list[int] = []
        for p in parents:
            for region in self._chunk_region(doc.text, (p.start, p.end)):
                regions.append((region, p.heading_path))
                owners.append(p.n)
        return parents, self._to_chunks(doc, regions, owners)


class SemanticChunker(MarkdownChunker):
    """Cut where the meaning shifts: sentences are embedded, and a cut goes between two adjacent
    sentences whose cosine distance is above the `semantic_threshold` percentile of all adjacent
    distances in the document. Groups between cuts are then packed like any other strategy, so
    `size` stays a hard limit; headings still cut (sections are never crossed). One embedding
    call per document, through the kb's embedder (and its cache)."""

    def __init__(self, config: ChunkingConfig, counter: TokenCounter, embed: Embed) -> None:
        super().__init__(config, counter)
        self._embed = embed

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text
        sections = [
            (path, [p for s in self._sentences(text, section) for p in self._fit(text, s, 0)])
            for section, path in self._sections(text)
        ]
        flat = [p for _, pieces in sections for p in pieces]
        vectors = self._embed([text[a:b].strip() for (a, b), _ in flat]) if flat else []
        vector_of = {region: _unit(v) for ((region, _), v) in zip(flat, vectors, strict=True)}
        gaps = [
            [_distance(vector_of[a[0]], vector_of[b[0]]) for a, b in pairwise(pieces)]
            for _, pieces in sections
        ]
        all_gaps = [g for section_gaps in gaps for g in section_gaps]
        threshold = _percentile(all_gaps, self._config.semantic_threshold) if all_gaps else 0.0
        regions: list[Section] = []
        for (path, pieces), section_gaps in zip(sections, gaps, strict=True):
            group: list[Piece] = []
            for piece, gap in zip(pieces, [0.0, *section_gaps], strict=True):
                if group and gap > threshold:
                    regions.extend((r, path) for r in self._pack(group))
                    group = []
                group.append(piece)
            if group:
                regions.extend((r, path) for r in self._pack(group))
        return self._to_chunks(doc, regions)

    def _sentences(self, text: str, region: Region) -> list[Region]:
        """Sentence pieces with some text (whitespace-only pieces carry no meaning to embed)."""
        return [
            s
            for line in split_at(text, region, LINE)
            for s in split_at(text, line, SENTENCE)
            if text[s[0] : s[1]].strip()
        ]


def _unit(v: list[float]) -> list[float]:
    """`v` scaled to length 1 (a zero vector stays zero: as far from anything as it gets)."""
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


def _distance(a: list[float], b: list[float]) -> float:
    """Cosine distance of two unit vectors."""
    return 1.0 - sum(x * y for x, y in zip(a, b, strict=True))


def _percentile(values: list[float], p: float) -> float:
    """Linear interpolation between the closest ranks: equal values give that value, so a
    document with one topic throughout (all gaps equal) gets no cut."""
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    low = math.floor(k)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


STRATEGIES: dict[str, Callable[[ChunkingConfig, TokenCounter], Chunker]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
    "paragraph": ParagraphChunker,
    "markdown": MarkdownChunker,
    "parent_child": ParentChildChunker,
}


def build_chunker(
    config: ChunkingConfig, counter: TokenCounter, embed: Embed | None = None
) -> Chunker:
    """The configured chunker. `semantic` also needs `embed` (the kb's document embedder), so it
    is not in `STRATEGIES`, the strategies that need no model."""
    if config.strategy == "semantic":
        if embed is None:
            raise ConfigError("chunking strategy semantic needs an embedder to embed sentences")
        if config.semantic_impl == "chonkie":
            from lite_rag.chunking_chonkie import ChonkieSemanticChunker

            return ChonkieSemanticChunker(config, counter, embed)
        return SemanticChunker(config, counter, embed)
    return STRATEGIES[config.strategy](config, counter)


def chunk_context(title: str | None, heading_path: tuple[str, ...]) -> str:
    """Where a chunk sits, as one line: `title > heading > subheading` (parts that exist)."""
    return " > ".join(part for part in (title, *heading_path) if part)


def embedding_text(chunk: Chunk, header: str | None = None) -> str:
    """What gets embedded and indexed: a contextual header (if any), the heading path (if any),
    then the chunk. The document title is left out on purpose: it would repeat in every chunk of
    a document and pull them together in vector space; the reranker and the prompt see it
    (`chunk_context`)."""
    path = chunk_context(None, chunk.heading_path)
    return "\n".join(part for part in (header, path, chunk.text) if part)
