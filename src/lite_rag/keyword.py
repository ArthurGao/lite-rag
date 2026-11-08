"""BM25 (design §5), written out so it can be read and changed.

    score(q, d) = Σ_t IDF(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))
    IDF(t)      = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

- t runs over the *distinct* query terms; f(t,d) is the term's count in chunk d, |d| the chunk's
  length in terms, N the number of chunks in the kb, n(t) the chunks containing t, avgdl the mean
  chunk length. This IDF (the "+1" inside the log, as in Lucene) is never negative.
- N, n(t) and avgdl are statistics of the whole kb; a metadata filter only narrows which chunks
  may be returned, it does not change the statistics.

PgStore can compute the same formula in SQL instead (`store.bm25_impl: sql`); the tests check the
two agree, with and without a metadata filter.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

from lite_rag.ranking import ranked


@dataclass(frozen=True)
class Posting:
    term: str
    chunk_id: str
    tf: int
    length: int  # BM25 terms in the chunk


@dataclass(frozen=True)
class Bm25Inputs:
    """What scoring needs; a store fills it. The statistics are the whole kb's."""

    n_chunks: int  # N
    avg_length: float  # avgdl
    doc_freq: dict[str, int]  # n(t): chunks of the kb containing each term
    postings: list[Posting]  # only those of chunks that may be returned


def idf(n_chunks: int, doc_freq: int) -> float:
    return math.log(1 + (n_chunks - doc_freq + 0.5) / (doc_freq + 0.5))


def term_score(tf: int, length: int, avg_length: float, k1: float, b: float) -> float:
    norm = 1 - b + b * (length / avg_length) if avg_length else 1.0
    return tf * (k1 + 1) / (tf + k1 * norm)


def bm25(inputs: Bm25Inputs, n: int, k1: float, b: float) -> list[tuple[str, float]]:
    """The `n` best chunks as (chunk_id, score), best first; ties broken by chunk id."""
    weights = {t: idf(inputs.n_chunks, df) for t, df in inputs.doc_freq.items()}
    per_chunk: dict[str, list[Posting]] = defaultdict(list)
    for p in inputs.postings:
        per_chunk[p.chunk_id].append(p)
    # Float addition is not associative. `sum` (compensated since Python 3.12) makes a chunk's
    # score independent of the order rows arrive in; summing in term order also matches the
    # order of the SQL sum (ORDER BY term). Any last-bit difference left is a tie (ranking.py).
    scores = {
        chunk: sum(
            weights[p.term] * term_score(p.tf, p.length, inputs.avg_length, k1, b)
            for p in sorted(ps, key=lambda p: p.term)
        )
        for chunk, ps in per_chunk.items()
    }
    return ranked(scores, n)
