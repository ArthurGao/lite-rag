"""lite-rag's eval semantics, now computed by lite-eval: the same hand-computed values as before
the move, plus scorer equivalence with the frozen original (lite-eval design §18.16, layer 1)."""

import random
from statistics import fmean
from typing import Any

import pytest
from lite_eval.contracts import Case, Output
from lite_eval.packs.cost import nearest_rank
from lite_eval.scorers.base import safe_score
from lite_eval.scorers.compose import build_scorer
from lite_eval.scorers.ranking import evidence_ranks

from lite_rag.config import load_settings
from lite_rag.eval import Question, billing, hit_record, index_identity, kb_for, slug
from lite_rag.types import Hit
from tests import legacy_eval as legacy

Q = Question("q1", "How hot?", "doc-a", "between 92 and 96", "keyword")


def _hit(doc: str, text: str, n: int = 0) -> Hit:
    return Hit(f"{doc}#{n}", doc, "t", text, 1.0, 0, len(text), kb="kb")


def ranks(q: Question, hits: list[Hit]) -> tuple[int | None, int | None]:
    return evidence_ranks([hit_record(h) for h in hits], q.doc, q.evidence)


def test_a_hit_answers_only_with_the_right_document_and_the_quote() -> None:
    hits = [
        _hit("doc-b", "between 92 and 96"),  # right quote, wrong document
        _hit("doc-a", "something else"),  # right document, no quote
        # whitespace differs: still a match
        _hit("doc-a", "use water   between 92\nand 96 degrees"),
    ]
    assert ranks(Q, hits) == (3, 2)


def test_no_answer_in_the_list_is_none() -> None:
    assert ranks(Q, [_hit("doc-b", "x")])[0] is None


def test_quotes_match_through_markdown_markup_and_pdf_line_breaks() -> None:
    q = Question("q", "?", "d", "机票退款期限为七天")
    docling = _hit("d", "## 机票\n\n**机票**退款期限为\n七天。")  # a line break inside the quote
    assert ranks(q, [docling])[0] == 1
    q_en = Question("q", "?", "d", "refund within 14 days")
    assert ranks(q_en, [_hit("d", "Hotels: refund\nwithin **14** days.")])[0] == 1
    assert ranks(q_en, [_hit("d", "refund within 41 days")])[0] is None


WORDS = ["refund", "within", "14", "days", "**", "## ", "\n", "机票", "七天", "\\_", "x", "  "]


def _random_hits(rng: random.Random) -> list[Hit]:
    return [
        _hit(rng.choice(["d", "e"]), " ".join(rng.choices(WORDS, k=rng.randint(1, 8))), n)
        for n in range(rng.randint(0, 10))
    ]


def test_the_scorer_finds_the_same_ranks_as_the_original_judge() -> None:
    """§18.16 layer 1: the same hits through the frozen judge and through lite-eval."""
    rng = random.Random(20261114)
    for _ in range(2000):
        evidence = " ".join(rng.choices(WORDS[:4] + WORDS[7:9], k=rng.randint(1, 3)))
        hits = _random_hits(rng)
        old = legacy.judge(legacy.Question("q", "?", "d", evidence), hits, 0.0)
        assert ranks(Question("q", "?", "d", evidence), hits) == (old.answer_rank, old.doc_rank)


def _hits_for(answer_rank: int | None) -> list[dict[str, str]]:
    """Ten hits from the right document with the quote at `answer_rank` only."""
    return [
        {"doc": "doc-a", "text": "between 92 and 96" if r == answer_rank else "no"}
        for r in range(1, 11)
    ]


def test_metrics_follow_their_definitions_as_the_original_did() -> None:
    answer_ranks = [1, 2, None, 6]
    expected = {"doc": Q.doc, "evidence": Q.evidence}
    ours = {
        name: fmean(
            safe_score(
                build_scorer(name),
                Case(id="c", input="", expected=expected),
                Output(case_id="c", value=_hits_for(r)),
            ).value
            for r in answer_ranks
        )
        for name in ("hit@1", "hit@5", "mrr@10")
    }
    old = legacy.metrics([legacy.Outcome(Q, r, 1, 1.0) for r in answer_ranks])

    assert ours["hit@1"] == old.hit_at_1 == 0.25
    assert ours["hit@5"] == old.hit_at_5 == 0.5  # ranks 1 and 2
    assert ours["mrr@10"] == pytest.approx(old.mrr_at_10, abs=1e-12)
    assert ours["mrr@10"] == pytest.approx((1 + 1 / 2 + 1 / 6) / 4)


@pytest.mark.parametrize("latencies", [[1.0, 3.0], [float(ms) for ms in range(1, 21)], [7.0]])
def test_p95_is_the_same_observed_latency(latencies: list[float]) -> None:
    assert nearest_rank(latencies, 0.95) == legacy._nearest_rank(latencies, 0.95)


def test_config_names_become_lite_eval_slugs() -> None:
    import re

    assert slug("chunk-markdown") == "chunk-markdown"
    assert slug("Mode Vector") == "mode-vector"
    assert slug("???") == "config"
    lite_eval_name = re.compile(r"^[a-z0-9][a-z0-9._-]*$")  # lite_eval.suite's rule
    for name in ("_x", ".hidden", "-a", "__", "Ünïcode name", "a b/c"):
        assert lite_eval_name.match(slug(name)), name


def test_configs_that_only_differ_in_retrieval_share_one_kb() -> None:
    base = {"chunking": {"strategy": "markdown", "size": 120, "overlap": 20}}
    hybrid = load_settings(**base, retrieval={"mode": "hybrid"})
    vector = load_settings(**base, retrieval={"mode": "vector"}, rerank={"enabled": True})
    other_chunks = load_settings(chunking={"strategy": "fixed", "size": 120, "overlap": 20})
    assert kb_for(index_identity(hybrid)) == kb_for(index_identity(vector))
    assert kb_for(index_identity(hybrid)) != kb_for(index_identity(other_chunks))
    assert kb_for(index_identity(hybrid)).startswith("eval-")


def test_an_embedding_endpoint_change_is_another_index() -> None:
    a = load_settings(embedding={"provider": "openai", "model": "m", "base_url": "https://a/v1"})
    b = load_settings(embedding={"provider": "openai", "model": "m", "base_url": "https://b/v1"})
    batch = load_settings(
        embedding={"provider": "openai", "model": "m", "base_url": "https://a/v1", "batch_size": 4}
    )
    assert kb_for(index_identity(a)) != kb_for(index_identity(b))
    assert kb_for(index_identity(a)) == kb_for(index_identity(batch))  # batching: same vectors


def test_the_loader_is_part_of_the_index() -> None:
    a = load_settings(loader={"pdf": "pypdf"})
    b = load_settings(loader={"pdf": "docling"})
    assert kb_for(index_identity(a)) != kb_for(index_identity(b))


def test_a_graph_is_part_of_the_index_so_it_gets_its_own_kb() -> None:
    plain = load_settings(chunking={"strategy": "markdown"})
    graph = load_settings(chunking={"strategy": "markdown"}, graph={"enabled": True})
    graph_bigger = load_settings(
        chunking={"strategy": "markdown"}, graph={"enabled": True, "llm": {"model": "qwen3:14b"}}
    )
    graph_deeper = load_settings(
        chunking={"strategy": "markdown"}, graph={"enabled": True, "hops": 3}
    )
    assert kb_for(index_identity(plain)) != kb_for(index_identity(graph))
    assert kb_for(index_identity(graph)) != kb_for(index_identity(graph_bigger))
    assert kb_for(index_identity(graph)) == kb_for(index_identity(graph_deeper))  # query-time only
    assert "graph" not in index_identity(plain)  # existing eval kbs keep their names


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, "free"),  # ollama embeddings
        ({"embedding": {"provider": "fastembed", "model": "m"}}, "free"),
        ({"embedding": {"provider": "voyage", "model": "m"}}, "per_token"),
        ({"embedding": {"provider": "openai", "model": "m"}}, "unknown"),
        ({"graph": {"enabled": True, "llm": {"provider": "anthropic", "model": "m"}}}, "per_token"),
    ],
)
def test_billing_is_free_only_when_every_model_is_local(
    settings: dict[str, Any], expected: str
) -> None:
    assert billing(load_settings(**settings)) == expected
