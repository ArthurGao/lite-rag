import json
from pathlib import Path

import psycopg
import pytest
from lite_eval.errors import RunError
from lite_eval.store import load_run
from typer.testing import CliRunner

from lite_rag.cli import app
from lite_rag.config import load_settings
from lite_rag.embedding import EmbeddingError
from lite_rag.errors import ConfigError, StoreError
from lite_rag.eval import evaluate, load_corpus, load_golden, outcomes, table
from lite_rag.loaders import Loaded, PdfLoader
from lite_rag.stores.base import KbRef
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeEmbedder
from tests.conftest import TEST_SCHEMA

REPO = Path(__file__).resolve().parents[2]
CORPUS, GOLDEN = REPO / "eval" / "corpus", REPO / "eval" / "golden.jsonl"
CHUNKING = {"unit": "chars", "size": 300, "overlap": 30}


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


def test_the_multihop_golden_set_matches_its_corpus() -> None:
    """Questions whose answer sits two or three documents away from what they name (S11)."""
    corpus = load_corpus(REPO / "eval" / "corpus-multihop")
    questions = load_golden(REPO / "eval" / "golden-multihop.jsonl", corpus)
    assert len(corpus) == 8 and len(questions) == 16
    assert [q.kind for q in questions].count("multihop") == 12
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


def test_evaluate_shares_a_kb_per_index_and_drops_it(
    store: PgStore, pg: psycopg.Connection, tmp_path: Path
) -> None:
    corpus = load_corpus(CORPUS)
    configs = [
        ("bm25", load_settings(chunking=CHUNKING, retrieval={"mode": "bm25"})),
        ("vector", load_settings(chunking=CHUNKING, retrieval={"mode": "vector"})),
        ("small-chunks", load_settings(chunking={**CHUNKING, "size": 150})),
    ]
    embedders: list[FakeEmbedder] = []

    def embedder_for(config: object) -> FakeEmbedder:
        embedders.append(FakeEmbedder(dim=16))
        return embedders[-1]

    results = evaluate(configs, CORPUS, GOLDEN, store, root=tmp_path, embedder_for=embedder_for)

    assert [r.record.target.config["name"] for r in results] == ["bm25", "vector", "small-chunks"]
    bm25 = results[0].record.summary
    assert bm25.n_cases == 32 and bm25.metrics["hit@5"].value > 0.5  # keyword overlap finds most
    m = bm25.metrics
    assert 0 <= m["mrr@10"].value <= m["hit@5"].value <= m["doc_hit@5"].value <= 1
    assert (
        results[1].record.target.config["retrieval"]["mode"] == "vector"
    )  # runs describe themselves
    assert results[2].record.target.config["chunking"]["size"] == 150
    assert len(embedders) == 1  # one embedder for one embedding config
    # the first two share their index: the corpus was embedded for two kbs, not three
    assert len(embedders[0].calls) == 2 * len(corpus)
    left = pg.execute(f"SELECT count(*) FROM {TEST_SCHEMA}.kbs").fetchone()
    assert left == (0,)  # every evaluation kb was dropped
    assert all(load_run(r.directory).record.status == "complete" for r in results)


class FailingEmbedder(FakeEmbedder):
    def __init__(self, fail_after: int) -> None:
        super().__init__(dim=16)
        self.fail_after = fail_after

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(self.calls) >= self.fail_after:
            raise EmbeddingError("model went away")
        return super().embed_documents(texts)


def _one_config() -> list[tuple[str, object, None]]:
    return [("x", load_settings(chunking=CHUNKING))]


def test_a_build_that_fails_halfway_leaves_no_kb(
    store: PgStore, pg: psycopg.Connection, tmp_path: Path
) -> None:
    with pytest.raises(RunError, match="went away") as raised:
        evaluate(
            _one_config(),
            CORPUS,
            GOLDEN,
            store,
            root=tmp_path,
            embedder_for=lambda c: FailingEmbedder(2),
        )
    assert isinstance(raised.value.__cause__, EmbeddingError)
    assert pg.execute(f"SELECT count(*) FROM {TEST_SCHEMA}.kbs").fetchone() == (0,)


def test_a_cleanup_failure_does_not_hide_the_real_error(
    store: PgStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_delete = store.delete_kb
    deletes: list[KbRef] = []

    def broken_cleanup(ref: KbRef) -> bool:  # the delete before the build works, cleanup fails
        deletes.append(ref)
        if len(deletes) > 1:
            raise StoreError("connection lost")
        return real_delete(ref)

    monkeypatch.setattr(store, "delete_kb", broken_cleanup)
    with pytest.raises(RunError, match="went away") as raised:
        evaluate(
            _one_config(),
            CORPUS,
            GOLDEN,
            store,
            root=tmp_path,
            embedder_for=lambda c: FailingEmbedder(2),
        )
    cause = raised.value.__cause__
    assert isinstance(cause, EmbeddingError)
    assert any("connection lost" in note for note in cause.__notes__)


def test_the_cli_refuses_configs_on_different_databases(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("database_url: postgresql://u@h1/db\n")
    b.write_text("database_url: postgresql://u@h2/db\n")
    result = CliRunner().invoke(app, ["eval", "-c", str(a), "-c", str(b)])
    assert result.exit_code != 0 and "one database" in result.output


def test_each_pdf_loader_gets_its_own_kb(store: PgStore, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text("# Doc\n\nRefunds take seven days.\n")
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps({"id": "q", "question": "How long?", "doc": "doc", "evidence": "seven days"})
    )
    embedder = FakeEmbedder(dim=16)
    configs = [
        ("pypdf", load_settings(chunking=CHUNKING, loader={"pdf": "pypdf"})),
        ("docling", load_settings(chunking=CHUNKING, loader={"pdf": "docling"})),
    ]

    results = evaluate(
        configs, corpus, golden, store, root=tmp_path / "root", embedder_for=lambda c: embedder
    )

    assert [r.record.target.config["loader"]["pdf"] for r in results] == ["pypdf", "docling"]
    assert len(embedder.calls) == 2  # one document embedded per build: one build per loader


def test_the_table_and_the_per_question_report(store: PgStore, tmp_path: Path) -> None:
    configs = [("chunk-x", load_settings(chunking=CHUNKING, retrieval={"mode": "bm25"}))]
    (result,) = evaluate(
        configs, CORPUS, GOLDEN, store, root=tmp_path, embedder_for=lambda c: FakeEmbedder(dim=16)
    )

    text = table([result])
    rows = outcomes(result)
    assert text.splitlines()[0] == (
        "| config | hit@1 | hit@5 | MRR@10 | doc hit@5 | p50 ms | p95 ms "
        "| hit@5 keyword | hit@5 paraphrase |"
    )
    row = text.splitlines()[2]
    by_kind = result.record.summary.groups["kind"]
    assert row.startswith("| chunk-x | ")
    assert row.count("|") == text.splitlines()[0].count("|")  # a cell under every header
    assert row.endswith(
        f"| {by_kind['keyword']['hit@5']:.2f} | {by_kind['paraphrase']['hit@5']:.2f} |"
    )
    assert "32 questions" in text
    assert len(rows) == 32 and {"answer_rank", "doc_rank", "latency_ms"} <= set(rows[0])
    hit5 = sum(r["answer_rank"] is not None and r["answer_rank"] <= 5 for r in rows) / 32
    assert hit5 == result.record.summary.metrics["hit@5"].value


class _QueriesFail(FakeEmbedder):
    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingError("query model went away")


def test_a_setup_that_fails_after_the_build_still_drops_the_kb(
    store: PgStore, pg: psycopg.Connection, tmp_path: Path
) -> None:
    with pytest.raises(RunError, match="query model went away"):
        evaluate(
            _one_config(),
            CORPUS,
            GOLDEN,
            store,
            root=tmp_path,
            embedder_for=lambda c: _QueriesFail(dim=16),
        )
    assert pg.execute(f"SELECT count(*) FROM {TEST_SCHEMA}.kbs").fetchone() == (0,)


def test_the_corpus_is_parsed_once_per_loader(
    store: PgStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lite_rag.eval as eval_module

    parsed: list[Path] = []
    real = eval_module.load_file

    def counting(path: Path, pdf: PdfLoader) -> Loaded:
        parsed.append(path)
        return real(path, pdf)

    files = len(load_corpus(CORPUS))  # counted before the patch
    monkeypatch.setattr(eval_module, "load_file", counting)
    configs = [  # two setup groups (different chunking), one loader
        ("a", load_settings(chunking=CHUNKING)),
        ("b", load_settings(chunking={**CHUNKING, "size": 150})),
    ]

    evaluate(
        configs, CORPUS, GOLDEN, store, root=tmp_path, embedder_for=lambda c: FakeEmbedder(dim=16)
    )

    assert len(parsed) == files


def test_a_lite_eval_error_is_a_one_line_cli_error(
    pg_url: str, pg: psycopg.Connection, tmp_path: Path
) -> None:
    config = tmp_path / "unreachable.yaml"
    config.write_text(
        f"database_url: {pg_url}\n"
        "embedding: {provider: openai, model: m, max_retries: 0,\n"
        "  base_url: 'http://127.0.0.1:9/v1'}\n"
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "-c",
            str(config),
            "--corpus",
            str(CORPUS),
            "--golden",
            str(GOLDEN),
            "--schema",
            TEST_SCHEMA,
            "--root",
            str(tmp_path / "root"),
        ],
    )

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "error: setup of config 'unreachable' failed" in result.output


def test_the_table_says_when_latency_includes_embedding_the_query(
    store: PgStore, tmp_path: Path
) -> None:
    configs = [
        ("cached", load_settings(chunking=CHUNKING)),
        ("uncached", load_settings(chunking=CHUNKING, embedding={"cache": False})),
    ]
    results = evaluate(
        configs, CORPUS, GOLDEN, store, root=tmp_path, embedder_for=lambda c: FakeEmbedder(dim=16)
    )

    scopes = [r.record.target.info["latency_scope"] for r in results]
    assert scopes == [
        "retrieval (+ rerank), query vector precomputed",
        "retrieval (+ rerank), query vector embedded in the timed search (embedding.cache is off)",
    ]
    footnote = table(results).split("\n\n", 1)[1]
    assert "query vector already computed" not in footnote
    assert "Latency: cached: retrieval (+ rerank), query vector precomputed;" in footnote
    assert "uncached: retrieval (+ rerank), query vector embedded in the timed search" in footnote
    one = table(results[:1]).split("\n\n", 1)[1]
    assert "Latency: retrieval (+ rerank), query vector precomputed." in one
