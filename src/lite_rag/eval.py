"""Evaluation (design §8): does a change to any layer make retrieval better or worse?

A golden set is a JSONL file of questions, each naming the document that answers it and a short
verbatim `evidence` quote from it. A hit *answers* a question when it comes from that document
and contains the quote (whitespace-normalised). Labelling by quote, not by chunk id, keeps one
golden set valid across chunkers — chunk ids change with every chunking config. A quote split
across two chunks counts as not answered: that is a real weakness of the chunker, not of the set.

How a run is made comparable:
- Configs with the same index identity (chunking + embedding) share one kb, built once; each
  config then searches it with its own retrieval and rerank settings.
- Latency is retrieval (+ rerank) only: every question's query vector is computed before timing
  and one untimed search warms the models up, so the order of configs does not matter.
- Search depth is `TOP_K` (10, for MRR@10); in hybrid mode each path returns 10 * candidate_factor.
"""

import hashlib
import json
import math
import re
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lite_rag.config import EmbeddingConfig, Settings
from lite_rag.embedding import Embedder, build_embedder
from lite_rag.engine import RagEngine
from lite_rag.errors import ConfigError, StoreError
from lite_rag.loaders import PdfLoader, find_files, load_file
from lite_rag.rerank import Reranker, build_reranker
from lite_rag.stores.base import KbRef
from lite_rag.stores.pg import PgStore
from lite_rag.types import Hit

TOP_K = 10  # MRR@10 needs ten hits
NAMESPACE = "eval"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    doc: str  # document id: the corpus file's stem
    evidence: str  # a verbatim quote from that document
    kind: str = "keyword"  # free label; reports group by it (keyword | paraphrase)
    lang: str = ""


@dataclass(frozen=True)
class Outcome:
    question: Question
    answer_rank: int | None  # 1-based rank of the first answering chunk; None: not in top 10
    doc_rank: int | None  # 1-based rank of the first chunk from the right document
    latency_ms: float


@dataclass(frozen=True)
class Metrics:
    n: int
    hit_at_1: float
    hit_at_5: float
    mrr_at_10: float
    doc_hit_at_5: float
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class Report:
    config: str
    settings: dict[str, Any]  # what was actually in effect: reports describe themselves
    outcomes: list[Outcome]
    overall: Metrics
    by_kind: dict[str, Metrics]


def report(config: str, settings: dict[str, Any], outcomes: list[Outcome]) -> Report:
    groups: dict[str, list[Outcome]] = {}
    for o in outcomes:
        groups.setdefault(o.question.kind, []).append(o)
    by_kind = {kind: metrics(group) for kind, group in sorted(groups.items())}
    return Report(config, settings, outcomes, metrics(outcomes), by_kind)


_MARKUP = re.compile(r"^#{1,6}[ \t]+|\*\*|__|\\(?=[_*#\\])", re.MULTILINE)
_CJK_GAP = re.compile(
    r"(?<=[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef])\s+(?=[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef])"
)


def _norm(text: str) -> str:
    """Text as a quote is matched against it: Markdown markup gone (docling writes headings,
    bold and escapes), whitespace collapsed, and none between CJK characters (a PDF line break
    inside a Chinese sentence)."""
    return _CJK_GAP.sub("", " ".join(_MARKUP.sub("", text).split()))


Corpus = dict[str, tuple[str, str]]  # doc id (file stem) -> (title, text)


def load_corpus(directory: Path, pdf: PdfLoader = "pypdf") -> Corpus:
    """Every supported file, in a stable order; `pdf` picks the PDF loader."""
    files = find_files([directory]) if directory.is_dir() else []
    if not files:
        raise ConfigError(f"no corpus files in {directory}")
    if len({p.stem for p in files}) != len(files):
        raise ConfigError(f"{directory}: two corpus files share a name; ids are file names")
    corpus: Corpus = {}
    for p in files:
        loaded = load_file(p, pdf)
        corpus[p.stem] = (loaded.title, loaded.text)
    return corpus


def load_golden(path: Path, corpus: dict[str, tuple[str, str]]) -> list[Question]:
    """The questions, each checked against the corpus: its document exists and contains its
    evidence quote — a golden set that does not match its corpus measures nothing."""
    questions = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        q = Question(**json.loads(line))
        if q.doc not in corpus:
            raise ConfigError(f"{path}:{n}: {q.id} names document {q.doc!r}, not in the corpus")
        if _norm(q.evidence) not in _norm(corpus[q.doc][1]):
            raise ConfigError(f"{path}:{n}: {q.id}: evidence is not in {q.doc!r}: {q.evidence!r}")
        questions.append(q)
    if not questions:
        raise ConfigError(f"{path}: no questions")
    return questions


def judge(question: Question, hits: list[Hit], latency_ms: float) -> Outcome:
    evidence = _norm(question.evidence)
    answer = doc = None
    for rank, hit in enumerate(hits, start=1):
        if hit.doc_id == question.doc:
            doc = doc or rank
            if evidence in _norm(hit.text):
                answer = rank
                break
    return Outcome(question, answer, doc, latency_ms)


def _within(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def metrics(outcomes: list[Outcome]) -> Metrics:
    n = len(outcomes)
    if n == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    answers = [o.answer_rank for o in outcomes]
    latencies = [o.latency_ms for o in outcomes]
    return Metrics(
        n=n,
        hit_at_1=_within(answers, 1),
        hit_at_5=_within(answers, 5),
        mrr_at_10=sum(1 / r for r in answers if r) / n,
        doc_hit_at_5=_within([o.doc_rank for o in outcomes], 5),
        p50_ms=statistics.median(latencies),
        p95_ms=_nearest_rank(latencies, 0.95),
    )


def _nearest_rank(values: list[float], q: float) -> float:
    """The q-quantile as the smallest observed value with at least q of the sample at or below
    it: always a latency that was measured, even for a handful of questions."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


# Embedding settings that never change a vector (how it is fetched, not what it is).
_TRANSPORT_ONLY = {
    "batch_size",
    "max_concurrency",
    "max_retries",
    "timeout_s",
    "cache",
    "cache_dir",
}


def index_identity(settings: Settings) -> dict[str, Any]:
    """What decides the index: configs that agree on it search the same kb (and share an
    embedder). Every embedding setting counts except transport ones — a different endpoint
    serving "the same" model is a different index until shown otherwise."""
    return {
        "loader": settings.loader.model_dump(),  # another loader: another text
        "chunking": settings.chunking.identity(),
        "embedding": settings.embedding.model_dump(exclude=_TRANSPORT_ONLY),
    }


def kb_for(identity: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return f"eval-{digest[:12]}"


def effective_settings(settings: Settings) -> dict[str, Any]:
    return {
        **index_identity(settings),
        "retrieval": settings.retrieval.model_dump(),
        "rerank": settings.rerank.model_dump(),
    }


EmbedderFor = Callable[[EmbeddingConfig], Embedder]


def run_configs(
    configs: list[tuple[str, Settings]],
    corpus: Corpus | Callable[[PdfLoader], Corpus],
    questions: Iterable[Question],
    store: PgStore,
    *,
    embedder_for: EmbedderFor | None = None,
    keep: bool = False,
) -> list[Report]:
    """One report per config, all on `store`. Embedders and rerankers are built once per
    distinct config; a kb once per index identity, dropped afterwards unless `keep`. `corpus`:
    the documents, or a function giving them for a PDF loader (configs may differ in it)."""
    questions = list(questions)
    make_embedder = embedder_for or (lambda c: build_embedder(c, cache=store))
    embedders: dict[str, Embedder] = {}
    rerankers: dict[str, Reranker] = {}
    built: set[str] = set()
    reports = []
    try:
        for name, settings in configs:
            identity = index_identity(settings)
            kb = kb_for(identity)
            key = json.dumps(identity["embedding"], sort_keys=True)
            if key not in embedders:  # not setdefault: that would build one every time
                embedders[key] = make_embedder(settings.embedding)
            embedder = embedders[key]
            reranker = None
            if settings.rerank.enabled:
                rkey = settings.rerank.model_dump_json()
                if rkey not in rerankers:
                    rerankers[rkey] = build_reranker(settings.rerank)
                reranker = rerankers[rkey]
            engine = RagEngine(settings, store, embedder, namespace=NAMESPACE, reranker=reranker)
            if kb not in built:
                built.add(kb)  # before the build: a half-built kb is cleaned up too
                engine.delete_kb(kb)  # a leftover from an interrupted run would mix builds
                docs = corpus(settings.loader.pdf) if callable(corpus) else corpus
                for doc_id, (title, text) in docs.items():
                    engine.upsert_text(kb, doc_id, title, text)
            outcomes = _ask_all(engine, kb, questions)
            reports.append(report(name, effective_settings(settings), outcomes))
    except BaseException as exc:
        for problem in _drop(store, built, keep):
            exc.add_note(f"cleanup: {problem}")  # the evaluation error stays the one raised
        raise
    problems = _drop(store, built, keep)
    if problems:
        raise StoreError("could not drop evaluation kbs: " + "; ".join(problems))
    return reports


def _drop(store: PgStore, kbs: set[str], keep: bool) -> list[str]:
    """Delete each kb independently; returns what failed."""
    if keep:
        return []
    problems = []
    for kb in sorted(kbs):
        try:
            store.delete_kb(KbRef(NAMESPACE, kb))
        except Exception as exc:
            problems.append(f"{kb}: {type(exc).__name__}: {exc}")
    return problems


def _ask_all(engine: RagEngine, kb: str, questions: list[Question]) -> list[Outcome]:
    # Untimed: query vectors (cached for the timed pass) and one warm-up search for lazy models.
    for q in questions:
        engine.search(kb, q.question, top_k=1, mode="vector")
    engine.search(kb, questions[0].question, top_k=TOP_K)
    outcomes = []
    for q in questions:
        start = time.perf_counter()
        hits = engine.search(kb, q.question, top_k=TOP_K)
        outcomes.append(judge(q, hits, (time.perf_counter() - start) * 1000))
    return outcomes


COLUMNS = (  # header, Metrics field, format
    ("hit@1", "hit_at_1", ".2f"),
    ("hit@5", "hit_at_5", ".2f"),
    ("MRR@10", "mrr_at_10", ".3f"),
    ("doc hit@5", "doc_hit_at_5", ".2f"),
    ("p50 ms", "p50_ms", ".0f"),
    ("p95 ms", "p95_ms", ".0f"),
)


def table(reports: list[Report]) -> str:
    """A Markdown table, one row per configuration, then hit@5 per question kind."""
    kinds = sorted({k for r in reports for k in r.by_kind})
    head = ["config", *(c[0] for c in COLUMNS), *(f"hit@5 {k}" for k in kinds)]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in reports:
        cells = [r.config, *(format(getattr(r.overall, f), fmt) for _, f, fmt in COLUMNS)]
        cells += [f"{r.by_kind[k].hit_at_5:.2f}" if k in r.by_kind else "—" for k in kinds]
        rows.append("| " + " | ".join(cells) + " |")
    n = reports[0].overall.n if reports else 0
    return "\n".join(rows) + (
        f"\n\n{n} questions; a hit answers when it is from the right document and contains the "
        f"evidence quote. Search depth {TOP_K}; latency is retrieval (+ rerank) with the query "
        "vector already computed."
    )
