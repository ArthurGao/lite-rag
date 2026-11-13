"""lite-eval design §18.16, layer 2: the frozen original `literag eval` and the lite-eval one, in
one process on one store, give every question the same answer_rank and doc_rank, and overall and
per-kind hit@1, hit@5, MRR@10 and doc hit@5 within 1e-9. Latency is not compared.

CI runs it with FakeEmbedder; `-m ollama` runs it with the real bge-m3 embeddings."""

from pathlib import Path
from typing import Any

import pytest

from lite_rag.config import Settings, load_settings
from lite_rag.eval import evaluate, outcomes
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeEmbedder
from tests import legacy_eval as legacy

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval"
SETS = [
    ("corpus", "golden.jsonl", "pypdf"),
    ("corpus-exact", "golden-exact.jsonl", "pypdf"),
    ("corpus-multihop", "golden-multihop.jsonl", "pypdf"),
    ("corpus-pdf", "golden-pdf.jsonl", "pypdf"),
]
CHUNKING = {"unit": "chars", "size": 300, "overlap": 30}
METRICS = {
    "hit_at_1": "hit@1",
    "hit_at_5": "hit@5",
    "mrr_at_10": "mrr@10",
    "doc_hit_at_5": "doc_hit@5",
}


def _configs(**extra: Any) -> list[tuple[str, Settings]]:
    return [
        (mode, load_settings(chunking=CHUNKING, retrieval={"mode": mode}, **extra))
        for mode in ("hybrid", "vector", "bm25")
    ]


def _assert_equivalent(
    store: PgStore,
    root: Path,
    corpus_dir: Path,
    golden: Path,
    configs: list[tuple[str, Settings]],
    embedder_for: Any,
) -> None:
    pdf = configs[0][1].loader.pdf
    questions = legacy.load_golden(golden, legacy.load_corpus(corpus_dir, pdf))
    old = legacy.run_configs(
        configs, legacy.load_corpus(corpus_dir, pdf), questions, store, embedder_for=embedder_for
    )
    new = evaluate(
        configs,
        corpus_dir,
        golden,
        store,
        root=root,
        embedder_for=embedder_for,
    )

    assert [r.config for r in old] == [str(r.record.target.config["name"]) for r in new]
    for report, result in zip(old, new, strict=True):
        theirs = {o.question.id: (o.answer_rank, o.doc_rank) for o in report.outcomes}
        ours = {row["id"]: (row["answer_rank"], row["doc_rank"]) for row in outcomes(result)}
        assert ours == theirs, report.config
        summary = result.record.summary
        for field, metric in METRICS.items():
            assert summary.metrics[metric].value == pytest.approx(
                getattr(report.overall, field), abs=1e-9
            ), (report.config, metric)
            for kind, m in report.by_kind.items():
                assert summary.groups["kind"][kind][metric] == pytest.approx(
                    getattr(m, field), abs=1e-9
                ), (report.config, kind, metric)


@pytest.mark.parametrize(("corpus", "golden", "pdf"), SETS)
def test_the_lite_eval_run_equals_the_original_with_fake_embeddings(
    store: PgStore, tmp_path: Path, corpus: str, golden: str, pdf: str
) -> None:
    _assert_equivalent(
        store,
        tmp_path,
        EVAL / corpus,
        EVAL / golden,
        _configs(loader={"pdf": pdf}),
        embedder_for=lambda c: FakeEmbedder(dim=64),
    )


@pytest.mark.ollama
def test_the_lite_eval_run_equals_the_original_with_bge_m3(store: PgStore, tmp_path: Path) -> None:
    """The real model (Ollama, bge-m3): needs `brew services run ollama` and the model pulled."""
    from lite_rag.embedding import build_embedder

    _assert_equivalent(
        store,
        tmp_path,
        EVAL / "corpus",
        EVAL / "golden.jsonl",
        _configs(),
        embedder_for=lambda c: build_embedder(c, cache=store),
    )
