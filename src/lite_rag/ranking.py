"""The one ordering every ranked list uses: score descending, then chunk id — so results are
deterministic and the SQL paths agree with the Python ones.

Scores are compared rounded to `PRECISION` decimals: two chunks whose scores differ only by
floating-point noise (the last bits of a sum) are a tie, and a tie goes to the smaller chunk id.
The SQL side orders by `round(score::numeric, 9)` for the same reason."""

import heapq
from collections.abc import Mapping

PRECISION = 9


def ranked(scores: Mapping[str, float], n: int | None = None) -> list[tuple[str, float]]:
    """(chunk_id, score), best first; only the best `n` when given (heap, not a full sort)."""
    key = lambda kv: (-round(kv[1], PRECISION), kv[0])  # noqa: E731
    if n is None:
        return sorted(scores.items(), key=key)
    return heapq.nsmallest(n, scores.items(), key=key)
