"""The one ordering every ranked list uses: score descending, then chunk id — so results are
deterministic and the SQL paths agree with the Python ones.

Scores are compared rounded to `PRECISION` decimals: two chunks whose scores differ only by
floating-point noise (the last bits of a sum) are a tie, and a tie goes to the smaller chunk id.
The SQL side orders by `round(score::numeric, 9)` for the same reason."""

import heapq
from collections.abc import Mapping

PRECISION = 9

# A chunk's key in a search over several kbs: (kb index, chunk id), since a chunk id is unique
# within its kb only. Ranked lists are keyed by a chunk id (one kb, in a store) or by a `Key`.
type Key = tuple[int, str]


def ranked[K: (str, Key)](scores: Mapping[K, float], n: int | None = None) -> list[tuple[K, float]]:
    """(key, score), best first; only the best `n` when given (heap, not a full sort). A key is
    a chunk id, or (kb index, chunk id) when several kbs are searched as one."""
    key = lambda kv: (-round(kv[1], PRECISION), kv[0])  # noqa: E731
    if n is None:
        return sorted(scores.items(), key=key)
    return heapq.nsmallest(n, scores.items(), key=key)
