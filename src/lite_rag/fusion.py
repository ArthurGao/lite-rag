"""Combining ranked lists (design §5).

- RRF: score(d) = Σ_lists 1 / (k + rank_list(d)), ranks starting at 1; a list that does not
  contain d adds nothing. It uses ranks only, so the lists' score scales never matter.
- Weighted: each list's scores are min-max normalised to [0, 1], then
  score(d) = Σ_lists weight · normalised(d); a list that does not contain d contributes 0.

Both break ties by chunk id, so results are deterministic.
"""

from collections.abc import Mapping, Sequence

from lite_rag.ranking import ranked

Ranking = Sequence[tuple[str, float]]  # (chunk_id, score), best first


def rrf(rankings: Mapping[str, Ranking], k: int = 60) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in rankings.values():
        for rank, (chunk, _) in enumerate(ranking, start=1):
            fused[chunk] = fused.get(chunk, 0.0) + 1.0 / (k + rank)
    return ranked(fused)


def _min_max(ranking: Ranking) -> dict[str, float]:
    if not ranking:
        return {}
    values = [s for _, s in ranking]
    low, high = min(values), max(values)
    if high == low:
        return {chunk: 1.0 for chunk, _ in ranking}
    return {chunk: (s - low) / (high - low) for chunk, s in ranking}


def weighted(
    rankings: Mapping[str, Ranking], weights: Mapping[str, float]
) -> list[tuple[str, float]]:
    """Sum of each list's min-max normalised score times its weight."""
    fused: dict[str, float] = {}
    for stage, ranking in rankings.items():
        for chunk, score in _min_max(ranking).items():
            fused[chunk] = fused.get(chunk, 0.0) + weights[stage] * score
    return ranked(fused)
