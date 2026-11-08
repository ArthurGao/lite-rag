"""Opt-in: the real Ollama server and the real bge-m3 tokenizer (`uv run pytest -m ollama`)."""

from pathlib import Path

import psycopg
import pytest
from typer.testing import CliRunner

from lite_rag.cli import app
from lite_rag.config import EmbeddingConfig, load_settings
from lite_rag.db import migrate
from lite_rag.embedding import OllamaEmbedder
from lite_rag.engine import RagEngine
from lite_rag.text import HFTokenCounter
from tests.conftest import TEST_SCHEMA

pytestmark = pytest.mark.ollama


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_bge_m3_embeds_1024_d_and_puts_paraphrases_closer() -> None:
    e = OllamaEmbedder(EmbeddingConfig())
    q = e.embed_query("退款要多久？")
    near, far = e.embed_documents(["退款会在七天内处理完成。", "今天北京天气晴朗。"])
    assert e.dim == 1024 and len(near) == 1024
    assert _cos(q, near) > _cos(q, far)


def test_the_bge_m3_tokenizer_counts_tokens_with_offsets() -> None:
    counter = HFTokenCounter("BAAI/bge-m3")
    text = "我住北京。I live in Beijing."
    spans = counter.spans(text)
    assert counter.count(text) == len(spans) > 4
    assert "".join(text[a:b] for a, b in spans).replace(" ", "") == text.replace(" ", "")


def test_cli_ingest_end_to_end(pg: psycopg.Connection, pg_url: str, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "refunds.md").write_text("# 退款\n\n退款会在七天内处理。\n\n超过七天请联系客服。")
    (docs / "shipping.txt").write_text("Orders over fifty dollars ship for free.")
    env = {
        "LITERAG_DATABASE_URL": pg_url,
        "LITERAG_STORE__SCHEMA": TEST_SCHEMA,
        "LITERAG_NAMESPACE": "e2e",
    }
    runner = CliRunner()
    assert runner.invoke(app, ["migrate"], env=env).exit_code == 0

    first = runner.invoke(app, ["ingest", "demo", str(docs)], env=env)
    again = runner.invoke(app, ["ingest", "demo", str(docs)], env=env)
    stats = runner.invoke(app, ["stats", "demo"], env=env)

    assert first.exit_code == 0, first.output
    assert first.output.count("chunks") == 2
    assert again.output.count("unchanged") == 2
    assert '"dim": 1024' in stats.output and '"documents": 2' in stats.output
    assert '"embedding": "ollama/bge-m3:567m"' in stats.output
    assert '"namespace": "e2e"' in stats.output

    found = runner.invoke(app, ["search", "demo", "多久能退款", "-k", "1"], env=env)
    assert found.exit_code == 0, found.output
    assert found.output.startswith("1. ") and "退款" in found.output and "vector #" in found.output
    missing = runner.invoke(app, ["search", "nope", "x"], env=env)
    assert missing.exit_code == 1 and "does not exist" in missing.output


def test_real_search_finds_paraphrases_in_both_languages(
    pg: psycopg.Connection, pg_url: str
) -> None:
    migrate(pg, TEST_SCHEMA)
    settings = load_settings(database_url=pg_url, store={"schema": TEST_SCHEMA})
    engine = RagEngine.from_settings(settings, namespace="e2e")
    try:
        docs = {
            "refund": "退款会在七天内原路退回到您的支付账户。",
            "shipping": "订单满五十元包邮，工作日当天发货。",
            "returns": "Items can be returned within thirty days if unused.",
            "warranty": "Electronics come with a one-year manufacturer warranty.",
        }
        for doc_id, text in docs.items():
            engine.upsert_text("faq", doc_id, doc_id, text)
        # no shared keywords with the answer: only the embedding can find these
        assert engine.search("faq", "钱多久能退给我", mode="vector")[0].doc_id == "refund"
        assert (
            engine.search("faq", "how long do I have to send it back", mode="vector")[0].doc_id
            == "returns"
        )
        hybrid = engine.search("faq", "包邮", top_k=2)
        assert hybrid[0].doc_id == "shipping" and set(hybrid[0].ranks) == {"vector", "bm25"}
    finally:
        engine.close()
