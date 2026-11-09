import pytest

from lite_rag.config import load_settings
from lite_rag.eval import Outcome, Question, index_identity, judge, kb_for, metrics, report, table
from lite_rag.types import Hit

Q = Question("q1", "How hot?", "doc-a", "between 92 and 96", "keyword")


def _hit(doc: str, text: str) -> Hit:
    return Hit(f"{doc}#0", doc, "t", text, 1.0, 0, len(text), kb="kb")


def test_a_hit_answers_only_with_the_right_document_and_the_quote() -> None:
    hits = [
        _hit("doc-b", "between 92 and 96"),  # right quote, wrong document
        _hit("doc-a", "something else"),  # right document, no quote
        # whitespace differs: still a match
        _hit("doc-a", "use water   between 92\nand 96 degrees"),
    ]
    outcome = judge(Q, hits, 12.0)
    assert (outcome.answer_rank, outcome.doc_rank, outcome.latency_ms) == (3, 2, 12.0)


def test_no_answer_in_the_list_is_none() -> None:
    assert judge(Q, [_hit("doc-b", "x")], 1.0).answer_rank is None


def test_metrics_follow_their_definitions() -> None:
    ranks = [1, 2, None, 6]  # answer ranks of four questions
    outcomes = [Outcome(Q, r, r, float(ms)) for r, ms in zip(ranks, [10, 20, 30, 40], strict=True)]
    m = metrics(outcomes)
    assert m.n == 4
    assert m.hit_at_1 == 0.25
    assert m.hit_at_5 == 0.5  # ranks 1 and 2
    assert m.mrr_at_10 == pytest.approx((1 + 1 / 2 + 1 / 6) / 4)
    assert m.doc_hit_at_5 == 0.5
    assert m.p50_ms == 25.0


def test_reports_split_by_question_kind_and_render_a_table() -> None:
    para = Question("q2", "?", "doc-a", "x", "paraphrase")
    r = report("chunk-markdown", {}, [Outcome(Q, 1, 1, 5.0), Outcome(para, None, 3, 7.0)])
    assert r.by_kind["keyword"].hit_at_5 == 1.0
    assert r.by_kind["paraphrase"].hit_at_5 == 0.0
    text = table([r])
    assert "| chunk-markdown | 0.50 | 0.50 | 0.500 | 1.00 |" in text
    assert "hit@5 keyword" in text and "hit@5 paraphrase" in text


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


def test_p95_is_an_observed_latency() -> None:
    outcomes = [Outcome(Q, 1, 1, 1.0), Outcome(Q, 1, 1, 3.0)]
    assert metrics(outcomes).p95_ms == 3.0  # nearest rank: never beyond the largest sample
    many = [Outcome(Q, 1, 1, float(ms)) for ms in range(1, 21)]
    assert metrics(many).p95_ms == 19.0  # ceil(0.95 * 20) = 19th smallest


def test_quotes_match_through_markdown_markup_and_pdf_line_breaks() -> None:
    q = Question("q", "?", "d", "机票退款期限为七天")
    docling = _hit("d", "## 机票\n\n**机票**退款期限为\n七天。")  # a line break inside the quote
    assert judge(q, [docling], 1.0).answer_rank == 1
    q_en = Question("q", "?", "d", "refund within 14 days")
    assert judge(q_en, [_hit("d", "Hotels: refund\nwithin **14** days.")], 1.0).answer_rank == 1
    assert judge(q_en, [_hit("d", "refund within 41 days")], 1.0).answer_rank is None


def test_the_loader_is_part_of_the_index() -> None:
    a = load_settings(loader={"pdf": "pypdf"})
    b = load_settings(loader={"pdf": "docling"})
    assert kb_for(index_identity(a)) != kb_for(index_identity(b))
