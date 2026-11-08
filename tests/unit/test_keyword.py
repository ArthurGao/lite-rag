import math

import pytest

from lite_rag.keyword import Bm25Inputs, Posting, bm25, idf, term_score

# A hand-checkable kb of three chunks (N = 3, avgdl = 2):
#   c1 = "xx yy"      (length 2)
#   c2 = "xx"         (length 1)
#   c3 = "zz zz ww"   (length 3)
CORPUS = {"c1": ["xx", "yy"], "c2": ["xx"], "c3": ["zz", "zz", "ww"]}


def _inputs(query: list[str]) -> Bm25Inputs:
    postings = [
        Posting(term, chunk, terms.count(term), len(terms))
        for chunk, terms in CORPUS.items()
        for term in dict.fromkeys(terms)
        if term in query
    ]
    df = {t: sum(t in terms for terms in CORPUS.values()) for t in query}
    return Bm25Inputs(3, 2.0, df, postings)


def test_idf_matches_the_formula_and_is_never_negative() -> None:
    assert idf(3, 2) == pytest.approx(math.log(1 + 1.5 / 2.5))  # ln 1.6 = 0.470004
    assert idf(3, 3) > 0  # a term in every chunk still scores above zero


def test_scores_match_hand_computed_values() -> None:
    # xx: n=2, idf = ln 1.6. c1: |d| = avgdl -> tf part 2.5/2.5 = 1. c2: |d| = 1 -> 2.5/1.9375.
    assert bm25(_inputs(["xx"]), 10, 1.5, 0.75) == [
        ("c2", pytest.approx(0.606456296)),
        ("c1", pytest.approx(0.470003629)),
    ]
    # zz: n=1, tf=2, |d|=3
    [(chunk, score)] = bm25(_inputs(["zz"]), 10, 1.5, 0.75)
    assert chunk == "c3" and score == pytest.approx(1.207174465)


def test_shorter_chunks_win_ties_in_term_frequency_and_b_controls_it() -> None:
    assert term_score(1, 1, 2.0, 1.5, 0.75) > term_score(1, 3, 2.0, 1.5, 0.75)
    assert term_score(1, 1, 2.0, 1.5, 0.0) == term_score(1, 3, 2.0, 1.5, 0.0)  # b=0: no length


def test_scores_add_over_query_terms_and_n_limits_the_result() -> None:
    both = dict(bm25(_inputs(["xx", "yy"]), 10, 1.5, 0.75))
    assert both["c1"] == pytest.approx(0.470003629 + idf(3, 1) * 1.0)
    assert len(bm25(_inputs(["xx", "yy"]), 1, 1.5, 0.75)) == 1


def test_equal_scores_are_ordered_by_chunk_id() -> None:
    inputs = Bm25Inputs(2, 1.0, {"t": 2}, [Posting("t", "b", 1, 1), Posting("t", "a", 1, 1)])
    assert [c for c, _ in bm25(inputs, 10, 1.5, 0.75)] == ["a", "b"]


def test_a_score_does_not_depend_on_the_order_postings_arrive_in() -> None:
    """Float addition is not associative: summing per-term scores in arrival order would make a
    chunk's score — and with it tie order — depend on how the store happened to return rows."""
    # N=4, n(t) = 1, 1, 2: summing these three weights in different orders differs in the last bit
    df = {"t1": 1, "t2": 1, "t3": 2}
    postings = [Posting(t, "c", 1, 3) for t in ("t1", "t2", "t3")]
    results = {
        bm25(Bm25Inputs(4, 3.0, df, [postings[i] for i in order]), 5, 1.5, 0.75)[0][1]
        for order in [(0, 1, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)]
    }
    assert len(results) == 1  # exactly the same score, not approximately


def test_near_ties_are_broken_by_chunk_id() -> None:
    from lite_rag.ranking import ranked

    # "b" is higher only by float noise: that is a tie, and ties go by chunk id
    assert [c for c, _ in ranked({"a": 0.6, "b": 0.6000000000000001})] == ["a", "b"]
