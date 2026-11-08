"""Opt-in: real local models from Hugging Face (`uv run --all-extras pytest -m model`)."""

import pytest

from lite_rag.config import RerankConfig
from lite_rag.rerank import FastEmbedReranker

pytestmark = pytest.mark.model


def test_the_multilingual_reranker_puts_the_answer_first() -> None:
    reranker = FastEmbedReranker(RerankConfig())
    texts = [
        "今天北京天气晴朗。",
        "退款会在七天内原路退回。",
        "Electronics carry a one-year warranty.",
    ]
    scores = reranker.score("退款要多久？", texts)
    assert max(range(3), key=scores.__getitem__) == 1
    scores = reranker.score("how long is the warranty", texts)
    assert max(range(3), key=scores.__getitem__) == 2
