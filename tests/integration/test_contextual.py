import psycopg
import pytest

from lite_rag.config import load_settings
from lite_rag.embedding import CachingEmbedder
from lite_rag.engine import RagEngine
from lite_rag.errors import ConfigError
from lite_rag.llm import LLMError
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeEmbedder, FakeLLM
from tests.conftest import TEST_SCHEMA
from tests.integration.support import CHUNKING, NS

TEXT = "Refunds take seven days.\n\nShipping is free above fifty dollars."
CONTEXT = "Situated in the handbook's money chapter."


def _engine(store: PgStore, llm: FakeLLM, embedder: FakeEmbedder, model: str = "qwen3:8b"):
    settings = load_settings(
        chunking={**CHUNKING, "contextual": {"enabled": True, "llm": {"model": model}}}
    )
    return RagEngine(
        settings, store, CachingEmbedder(embedder, store), namespace=NS, context_llm=llm
    )


def test_the_context_is_indexed_but_never_returned(store: PgStore, pg: psycopg.Connection) -> None:
    llm, embedder = FakeLLM(reply=CONTEXT), FakeEmbedder(dim=8)
    engine = _engine(store, llm, embedder)
    engine.upsert_text("kb", "d", "Handbook", TEXT)

    assert len(llm.calls) == 2  # one call per chunk
    inputs = [t for batch in embedder.calls for t in batch]
    assert len(inputs) == 2 and all(t.startswith(CONTEXT + "\n") for t in inputs)
    hits = engine.search("kb", "situated chapter", mode="bm25")  # words only in the context
    assert len(hits) == 2
    assert all(CONTEXT not in h.text for h in hits)  # the hit text is the document's own
    stored = pg.execute(f"SELECT DISTINCT context FROM {TEST_SCHEMA}.chunks").fetchall()
    assert stored == [(CONTEXT,)]


def test_a_failed_context_call_writes_nothing(store: PgStore, pg: psycopg.Connection) -> None:
    class Broken(FakeLLM):
        def complete(self, messages):  # type: ignore[no-untyped-def]
            raise LLMError("model went away")

    with pytest.raises(LLMError):
        _engine(store, Broken(), FakeEmbedder(dim=8)).upsert_text("kb", "d", "Handbook", TEXT)
    assert pg.execute(f"SELECT count(*) FROM {TEST_SCHEMA}.chunks").fetchone() == (0,)


def test_a_kb_refuses_another_context_model(store: PgStore) -> None:
    _engine(store, FakeLLM(reply=CONTEXT), FakeEmbedder(dim=8)).upsert_text("kb", "d", "H", TEXT)
    other = _engine(store, FakeLLM(reply=CONTEXT), FakeEmbedder(dim=8), model="qwen3:14b")
    with pytest.raises(ConfigError, match="contextual"):
        other.search("kb", "refunds")
