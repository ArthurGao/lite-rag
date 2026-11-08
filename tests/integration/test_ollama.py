"""Opt-in: the real Ollama server and the real bge-m3 tokenizer (`uv run pytest -m ollama`)."""

from pathlib import Path

import psycopg
import pytest
from typer.testing import CliRunner

from lite_rag.cli import app
from lite_rag.config import EmbeddingConfig
from lite_rag.embedding import OllamaEmbedder
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
