import pytest

from lite_rag.fusion import rrf, weighted


def test_rrf_adds_reciprocal_ranks_across_lists() -> None:
    fused = dict(rrf({"vector": [("a", 0.9), ("b", 0.8)], "bm25": [("b", 7.0), ("c", 3.0)]}, k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)  # 2nd in one list, 1st in the other
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_ignores_score_scales_and_orders_ties_by_id() -> None:
    fused = rrf({"x": [("z", 1000.0)], "y": [("a", 0.001)]}, k=60)
    assert fused == [("a", pytest.approx(1 / 61)), ("z", pytest.approx(1 / 61))]


def test_weighted_normalises_each_list_then_mixes_them() -> None:
    lists = {"vector": [("a", 0.9), ("b", 0.5)], "bm25": [("b", 10.0), ("c", 2.0)]}
    fused = dict(weighted(lists, {"vector": 0.7, "bm25": 0.3}))
    assert fused["a"] == pytest.approx(0.7 * 1.0)  # best vector, absent from bm25
    assert fused["b"] == pytest.approx(0.7 * 0.0 + 0.3 * 1.0)
    assert fused["c"] == pytest.approx(0.0)


def test_weighted_handles_a_single_hit_and_an_empty_list() -> None:
    lists = {"vector": [("a", 0.4)], "bm25": []}
    assert weighted(lists, {"vector": 0.5, "bm25": 0.5}) == [("a", pytest.approx(0.5))]
