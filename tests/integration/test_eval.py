import json
from pathlib import Path

import psycopg
import pytest
from typer.testing import CliRunner

from lite_rag.cli import app
from lite_rag.config import load_settings
from lite_rag.embedding import EmbeddingError
from lite_rag.errors import ConfigError, StoreError
from lite_rag.eval import load_corpus, load_golden, run_configs
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeEmbedder
from tests.conftest import TEST_SCHEMA

REPO = Path(__file__).resolve().parents[2]
CORPUS, GOLDEN = REPO / "eval" / "corpus", REPO / "eval" / "golden.jsonl"


def test_the_shipped_golden_set_matches_its_corpus() -> None:
    corpus = load_corpus(CORPUS)
    questions = load_golden(GOLDEN, corpus)
    assert len(corpus) == 8 and len(questions) == 32
    assert {q.kind for q in questions} == {"keyword", "paraphrase"}
    assert {q.lang for q in questions} == {"en", "zh"}
    assert all(sum(q.doc == d for q in questions) == 4 for d in corpus)


def test_the_exact_golden_set_matches_its_corpus() -> None:
    """Identifier, code and model-number questions (`exact`), with paraphrases of the same
    documents, for comparing retrieval paths where BM25 is expected to help."""
    corpus = load_corpus(REPO / "eval" / "corpus-exact")
    questions = load_golden(REPO / "eval" / "golden-exact.jsonl", corpus)
    assert len(corpus) == 2 and len(questions) == 12
    assert [q.kind for q in questions].count("exact") == 8
    assert {q.lang for q in questions} == {"en", "zh"}


def test_a_golden_line_whose_quote_is_not_in_its_document_is_refused(tmp_path: Path) -> None:
    corpus = {"doc": ("Doc", "Refunds take seven days.")}
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps({"id": "x", "question": "?", "doc": "doc", "evidence": "nine"}))
    with pytest.raises(ConfigError, match="evidence is not in 'doc'"):
        load_golden(golden, corpus)
    golden.write_text(json.dumps({"id": "x", "question": "?", "doc": "nope", "evidence": "x"}))
    with pytest.raises(ConfigError, match="not in the corpus"):
        load_golden(golden, corpus)


def test_run_configs_shares_a_kb_per_index_and_drops_it(
    store: PgStore, pg: psycopg.Connection
) -> None:
    corpus = load_corpus(CORPUS)
    questions = load_golden(GOLDEN, corpus)
    chunking = {"unit": "chars", "size": 300, "overlap": 30}
    configs = [
        ("bm25", load_settings(chunking=chunking, retrieval={"mode": "bm25"})),
        ("vector", load_settings(chunking=chunking, retrieval={"mode": "vector"})),
        ("small-chunks", load_settings(chunking={**chunking, "size": 150})),
    ]
    embedders: list[FakeEmbedder] = []

    def embedder_for(config: object) -> FakeEmbedder:
        embedders.append(FakeEmbedder(dim=16))
        return embedders[-1]

    reports = run_configs(configs, corpus, questions, store, embedder_for=embedder_for)

    assert [r.config for r in reports] == ["bm25", "vector", "small-chunks"]
    bm25 = reports[0].overall
    assert bm25.n == 32 and bm25.hit_at_5 > 0.5  # keyword overlap finds most answers
    assert 0 <= bm25.mrr_at_10 <= bm25.hit_at_5 <= bm25.doc_hit_at_5 <= 1
    assert reports[1].settings["retrieval"]["mode"] == "vector"  # reports describe themselves
    assert reports[2].settings["chunking"]["size"] == 150
    assert len(embedders) == 1  # one embedder for one embedding config
    # the first two share their index: the corpus was embedded for two kbs, not three
    built = [batch for batch in embedders[0].calls]
    assert len(built) == 2 * len(corpus)
    left = pg.execute(f"SELECT count(*) FROM {TEST_SCHEMA}.kbs").fetchone()
    assert left == (0,)  # every evaluation kb was dropped


class FailingEmbedder(FakeEmbedder):
    def __init__(self, fail_after: int) -> None:
        super().__init__(dim=16)
        self.fail_after = fail_after

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(self.calls) >= self.fail_after:
            raise EmbeddingError("model went away")
        return super().embed_documents(texts)


def test_a_build_that_fails_halfway_leaves_no_kb(store: PgStore, pg: psycopg.Connection) -> None:
    corpus = load_corpus(CORPUS)
    questions = load_golden(GOLDEN, corpus)
    configs = [("x", load_settings(chunking={"unit": "chars", "size": 300, "overlap": 30}))]
    with pytest.raises(EmbeddingError, match="went away"):
        run_configs(configs, corpus, questions, store, embedder_for=lambda c: FailingEmbedder(2))
    assert pg.execute(f"SELECT count(*) FROM {TEST_SCHEMA}.kbs").fetchone() == (0,)


def test_a_cleanup_failure_does_not_hide_the_real_error(
    store: PgStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = load_corpus(CORPUS)
    questions = load_golden(GOLDEN, corpus)
    configs = [("x", load_settings(chunking={"unit": "chars", "size": 300, "overlap": 30}))]

    real_delete = store.delete_kb
    deletes: list[object] = []

    def broken_cleanup(ref: object) -> bool:  # the delete before the build works, cleanup fails
        deletes.append(ref)
        if len(deletes) > 1:
            raise StoreError("connection lost")
        return real_delete(ref)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "delete_kb", broken_cleanup)
    with pytest.raises(EmbeddingError, match="went away") as raised:
        run_configs(configs, corpus, questions, store, embedder_for=lambda c: FailingEmbedder(2))
    assert any("connection lost" in note for note in raised.value.__notes__)


def test_the_cli_refuses_configs_on_different_databases(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("database_url: postgresql://u@h1/db\n")
    b.write_text("database_url: postgresql://u@h2/db\n")
    result = CliRunner().invoke(app, ["eval", "-c", str(a), "-c", str(b)])
    assert result.exit_code != 0 and "one database" in result.output


def test_each_pdf_loader_gets_its_own_text_and_kb(store: PgStore) -> None:
    corpus = {"doc": ("Doc", "Refunds take seven days.")}
    questions = [load_golden_line("Refunds take seven days")]
    asked: list[str] = []

    def corpus_for(pdf: str) -> dict[str, tuple[str, str]]:
        asked.append(pdf)
        return corpus

    chunking = {"unit": "chars", "size": 300, "overlap": 30}
    configs = [
        ("pypdf", load_settings(chunking=chunking, loader={"pdf": "pypdf"})),
        ("docling", load_settings(chunking=chunking, loader={"pdf": "docling"})),
    ]
    reports = run_configs(
        configs, corpus_for, questions, store, embedder_for=lambda c: FakeEmbedder(dim=16)
    )
    assert asked == ["pypdf", "docling"]  # one build per loader
    assert [r.settings["loader"]["pdf"] for r in reports] == ["pypdf", "docling"]


def load_golden_line(evidence: str):  # type: ignore[no-untyped-def]
    from lite_rag.eval import Question

    return Question("q", "How long do refunds take?", "doc", evidence)
