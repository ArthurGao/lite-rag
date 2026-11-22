"""Evaluation (design §8): does a change to any layer make retrieval better or worse?

`literag eval` runs on lite-eval (its retrieval pack): this module is lite-rag's adapter.

A golden set is a JSONL file of questions, each naming the document that answers it and a short
verbatim `evidence` quote from it. A hit *answers* a question when it comes from that document
and contains the quote (normalised: Markdown markup gone, whitespace collapsed, none between CJK
characters). Labelling by quote, not by chunk id, keeps one golden set valid across chunkers.
A quote split across two chunks counts as not answered: a real weakness of the chunker.

`evaluate` turns a golden set and its corpus into a lite-eval dataset and runs one
`LiteRagTarget` per config. How a run is made comparable (lite-eval design §18.18):
- Configs with the same index identity (chunking + embedding + loader + graph) form one setup
  group: the first builds the kb, the others share it; it is dropped after the group ran.
- Latency is retrieval (+ rerank) only: every question's query vector is computed in setup (kept
  by the embedding cache, `embedding.cache`), and lite-eval makes one untimed warm-up search
  before the timed ones.
- Search depth is `TOP_K` (10, for MRR@10); in hybrid mode each path returns 10 * candidate_factor.
"""

import asyncio
import hashlib
import json
import re
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from lite_eval.config import ci_from_env
from lite_eval.contracts import Case, Output, Span, Trace
from lite_eval.datasets.local import Dataset, load_dataset
from lite_eval.runner import RunOptions, RunResult, execute
from lite_eval.scorers.ranking import normalise
from lite_eval.store import load_run
from lite_eval.suite import load_suite

from lite_rag import __version__
from lite_rag.config import EmbeddingConfig, LLMConfig, RerankConfig, Role, Settings
from lite_rag.embedding import Embedder, build_embedder
from lite_rag.engine import RagEngine, RagRuntime
from lite_rag.errors import ConfigError, StoreError
from lite_rag.llm import LLM
from lite_rag.loaders import PdfLoader, find_files, load_file
from lite_rag.rerank import Reranker, build_reranker
from lite_rag.stores.base import CachedStore, KbRef
from lite_rag.types import Hit, Route

TOP_K = 10  # MRR@10 needs ten hits
NAMESPACE = "eval"
SCORERS = ["hit@1", "hit@5", "mrr@10", "doc_hit@5"]


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    doc: str  # document id: the corpus file's stem
    evidence: str  # a verbatim quote from that document
    # A free label; reports group by it (keyword | paraphrase | exact | multihop).
    kind: str = "keyword"
    lang: str = ""
    doc_lang: str = ""  # S22: the document's language, when it is not the question's
    # S14: a case-insensitive regex a correct answer matches (`literag eval-answers` checks it).
    key_fact: str | None = None


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


def corpus_groups(directory: Path) -> dict[str, str]:
    """Each document's subdirectory of `directory` ("" at the top), for `--kb-per-dir` (S22)."""
    return {
        p.stem: (p.relative_to(directory).parts[:-1] or ("",))[0] for p in find_files([directory])
    }


def load_golden(path: Path, corpus: Corpus) -> list[Question]:
    """The questions, each checked against the corpus: its document exists and contains its
    evidence quote — a golden set that does not match its corpus measures nothing."""
    documents: dict[str, str] = {}  # normalised once per document, not once per question
    questions = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        q = Question(**json.loads(line))
        if q.doc not in corpus:
            raise ConfigError(f"{path}:{n}: {q.id} names document {q.doc!r}, not in the corpus")
        if q.doc not in documents:
            documents[q.doc] = normalise(corpus[q.doc][1])
        if normalise(q.evidence) not in documents[q.doc]:
            raise ConfigError(f"{path}:{n}: {q.id}: evidence is not in {q.doc!r}: {q.evidence!r}")
        if q.key_fact is not None:
            try:
                fact = re.compile(q.key_fact, re.IGNORECASE)
            except re.error as exc:
                raise ConfigError(f"{path}:{n}: {q.id}: key_fact is not a regex: {exc}") from exc
            if fact.search(q.question):  # restating the question would pass
                raise ConfigError(f"{path}:{n}: {q.id}: key_fact matches the question itself")
        questions.append(q)
    if not questions:
        raise ConfigError(f"{path}: no questions")
    return questions


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
        "chunking": settings.chunking.identity(settings.llm),
        "embedding": settings.embedding.model_dump(exclude=_TRANSPORT_ONLY),
        **settings.graph.index_identity(settings.llm),  # a graph is built into the index
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


_LOCAL_EMBEDDERS = {"ollama", "fastembed", "sentence_transformers"}
_PAID = {"voyage", "cohere", "gemini"}  # embedding providers


def billing(
    settings: Settings, *, answers: bool = False, embeddings: Sequence[str] = ()
) -> Literal["free", "per_token", "unknown"]:
    """For lite-eval's CI guard: free only when every model the run calls is local (`answers`:
    the run also calls the answer LLM). An LLM role counts by the channel lite-model resolves
    for it (S15): local, api (paid per token) or subscription."""
    on: dict[Role, bool] = {
        "answer": answers,
        "contextual": settings.chunking.contextual.enabled,
        "graph": settings.graph.enabled,
        "route": settings.retrieval.mode == "route",  # the router, on every question (S21)
    }
    channels = {settings.llm.target_of(role).channel for role, used in on.items() if used}
    # S22: the named embedders a kb is built with (`--kb-per-dir`) are called too
    providers = {
        settings.embedding.provider,
        *(settings.embeddings[n].provider for n in embeddings),
    }
    if providers & _PAID or "api" in channels:
        return "per_token"
    if providers <= _LOCAL_EMBEDDERS and channels <= {"local"}:
        return "free"
    return "unknown"  # an OpenAI-compatible embedder, or a subscription


def latency_scope(settings: Settings) -> str:
    """What a recorded latency covers (lite-eval design §5.4)."""
    query = (
        "query vector precomputed"
        if settings.embedding.cache
        else "query vector embedded in the timed search (embedding.cache is off)"
    )
    return f"retrieval (+ rerank), {query}"


def hit_record(hit: Hit) -> dict[str, Any]:
    """A hit as lite-eval's retrieval scorers read it."""
    return {
        "doc": hit.doc_id,
        "text": hit.text,
        "id": hit.chunk_id,
        "title": hit.title,
        "score": hit.score,
    }


EmbedderFor = Callable[[EmbeddingConfig], Embedder]
LlmFor = Callable[[LLMConfig], LLM]
RerankerFor = Callable[[RerankConfig], Reranker]
CorpusFor = Callable[[PdfLoader], Corpus]


def route_span(route: Route) -> Span:
    """The router's choice for one case (S21), kept in the run's outputs."""
    return Span(
        kind="route",
        name=route.mode,
        output={"mode": route.mode, "reason": route.reason},
        attrs={"model": route.model_id},
    )


class LiteRagTarget:
    """One lite-rag config as a lite-eval target (lite-eval design §5.3), built by `evaluate`."""

    name = "lite-rag"
    query_key = "query"  # the case input searched for

    def __init__(
        self,
        settings: Settings,
        *,
        store: CachedStore,
        embedder_for: EmbedderFor,
        reranker_for: RerankerFor,
        corpus_for: CorpusFor,
        llm_for: LlmFor | None = None,  # retrieval never calls an LLM; answers do
        keep: bool = False,
        groups: dict[str, str] | None = None,  # doc -> subdirectory: a kb per subdirectory
    ) -> None:
        self.settings = settings
        self.store = store
        self._embedder_for = embedder_for
        self._reranker_for = reranker_for
        self._llm_for = llm_for
        self._corpus_for = corpus_for
        self.keep = keep
        self._groups = groups or {}  # doc -> subdirectory; none: one kb for the corpus
        self._identity = index_identity(settings)
        dirs = sorted(set(self._groups.values()))
        named = {d: settings.embeddings[d] for d in dirs if d in settings.embeddings}
        if groups is not None:  # S22: which kbs, and the named embedders they are built with
            self._identity |= {
                "kb_per_dir": dirs,
                "embeddings": {d: c.model_dump(exclude=_TRANSPORT_ONLY) for d, c in named.items()},
            }
        self.kb = kb_for(self._identity)
        # kb -> (its subdirectory, its embedder name or None for the default)
        # kb names by position: a directory name need not be a valid kb name (Codex S22)
        self.kbs: dict[str, tuple[str, str | None]] = (
            {f"{self.kb}-d{i}": (d, d if d in named else None) for i, d in enumerate(dirs)}
            if groups is not None
            else {self.kb: ("", None)}
        )
        self.embedder: Embedder | None = None
        self.engine: RagEngine | None = None
        self.owns_kb = False  # the setup group's leader builds the kb and drops it

    def config(self) -> dict[str, Any]:
        return effective_settings(self.settings)

    def versions(self) -> dict[str, str]:
        return {"lite-rag": __version__}

    def info(self) -> dict[str, str]:
        return {"latency_scope": latency_scope(self.settings)}

    def billing(self) -> Literal["free", "per_token", "unknown"]:
        used = sorted({e for _, e in self.kbs.values() if e is not None})
        return billing(self.settings, embeddings=used)

    def setup_identity(self) -> dict[str, Any]:
        return self._identity

    def _start(self, embedder: Embedder) -> RagEngine:
        self.embedder = embedder
        rerank = self.settings.rerank
        reranker = self._reranker_for(rerank) if rerank.enabled else None
        llm = self._llm_for(self.settings.llm) if self._llm_for else None
        used = {e for _, e in self.kbs.values() if e is not None}  # only what a kb is built with
        named = {n: self._embedder_for(self.settings.embeddings[n]) for n in used}
        runtime = RagRuntime(
            self.settings, self.store, embedder, reranker=reranker, llm=llm, embedders=named
        )
        self.engine = runtime.engine(NAMESPACE)
        return self.engine

    def query(self, case: Case) -> str:
        assert isinstance(case.input, dict)
        return str(case.input[self.query_key])

    async def setup(self, dataset: Dataset) -> None:
        """Build the kb and compute every query vector, untimed. A failure drops what was built:
        lite-eval tears down only targets whose setup succeeded."""
        engine = self._start(self._embedder_for(self.settings.embedding))
        self.owns_kb = True
        try:
            corpus = self._corpus_for(self.settings.loader.pdf)
            for kb, (group, embedding) in self.kbs.items():
                engine.delete_kb(kb)  # a leftover from an interrupted run would mix builds
                engine.create_kb(kb, embedding=embedding)
                for doc_id, (title, text) in corpus.items():
                    if self._groups.get(doc_id, "") == group:
                        engine.upsert_text(kb, doc_id, title, text)
            for case in dataset.cases:  # query vectors, kept by the embedding cache
                engine.search_kbs(list(self.kbs), self.query(case), top_k=1, mode="vector")
        except BaseException as exc:
            problem = self._drop()
            if problem:
                exc.add_note(f"cleanup: {problem}")  # the build error stays the one raised
            raise

    async def share_setup_from(self, leader: "LiteRagTarget") -> None:
        """Same index identity: search the leader's kb with the leader's embedder."""
        assert leader.embedder is not None
        self._start(leader.embedder)

    async def run(self, case: Case) -> Output:
        assert self.engine is not None
        start = time.perf_counter()
        hits = self.engine.search_kbs(list(self.kbs), self.query(case), top_k=TOP_K)
        latency_ms = (time.perf_counter() - start) * 1000  # with the routing call, if any
        route = hits[0].route if hits else None
        spans = [] if route is None else [route_span(route)]
        return Output(
            case_id=case.id,
            value=[hit_record(h) for h in hits],
            trace=Trace(
                spans=spans, latency_ms=latency_ms, latency_scope=latency_scope(self.settings)
            ),
        )

    def _drop(self) -> str | None:
        """Delete the kbs; returns what failed, if anything."""
        if self.keep:
            return None
        problems = []
        for kb in self.kbs:
            try:
                self.store.delete_kb(KbRef(NAMESPACE, kb))
            except Exception as exc:
                problems.append(f"{kb}: {type(exc).__name__}: {exc}")
        return "; ".join(problems) or None

    async def teardown(self) -> None:
        problem = self._drop() if self.owns_kb else None
        if problem:
            raise StoreError(f"could not drop evaluation kb {problem}")


def built_in_code(**config: Any) -> LiteRagTarget:
    """The `python:` entry of a suite `literag eval` generates. Its targets are built in code
    (one store, shared embedders), so the suite records the run but cannot rerun it."""
    raise ConfigError("this suite was generated by `literag eval`; run that command instead")


# ---------------------------------------------------------------- `literag eval`


def slug(name: str) -> str:
    """A lite-eval config or suite name: lowercase letters, digits, `.`, `_`, `-`, starting
    with a letter or digit (lite_eval.suite's rule)."""
    s = re.sub(r"[^a-z0-9._-]+", "-", name.lower())
    return re.sub(r"^[^a-z0-9]+|[-.]+$", "", s) or "config"


def write_generated_dataset(
    directory: Path, meta: dict[str, Any], cases: Sequence[dict[str, Any]]
) -> Path:
    """A generated lite-eval dataset, rewritten whole every time: `dataset.yaml` and
    `cases.jsonl` (the retrieval eval, the answer eval and the judge's calibration set)."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    dump = yaml.safe_dump(meta, sort_keys=False)  # as before S13: the retrieval hash holds
    (directory / "dataset.yaml").write_text(dump, "utf-8")
    lines = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    (directory / "cases.jsonl").write_text(lines, "utf-8")
    return directory


def golden_cases(questions: Sequence[Question], input_key: str) -> list[dict[str, Any]]:
    """The golden questions as lite-eval cases; the question goes under `input_key` (`query` to
    search, `question` to answer)."""
    return [
        {
            "id": q.id,
            "input": {input_key: q.question},
            "expected": {
                "doc": q.doc,
                "evidence": q.evidence,
                **({"key_fact": q.key_fact} if q.key_fact else {}),
            },
            "tags": {
                "kind": q.kind,
                **({"lang": q.lang} if q.lang else {}),
                **({"doc_lang": q.doc_lang} if q.doc_lang else {}),  # S22
            },
        }
        for q in questions
    ]


def write_dataset(root: Path, golden: Path, corpus: Path, questions: Sequence[Question]) -> Path:
    """The golden set and its corpus as a lite-eval dataset under `root/datasets/`."""
    meta = {
        "name": slug(golden.stem),
        "version": "1",
        "pack": "retrieval",
        "description": f"lite-rag golden set {golden.name} over {corpus.name}/ (generated)",
        "license": "Apache-2.0",
    }
    directory = root / "datasets" / slug(golden.stem)
    write_generated_dataset(directory, meta, golden_cases(questions, "query"))
    for path in find_files([corpus]):
        target = directory / "corpus" / path.relative_to(corpus)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return directory


def write_suite(
    root: Path,
    name: str,
    configs: Sequence[str],
    schema: str,
    *,
    pack: str = "retrieval",
    scorers: Sequence[Any] = SCORERS,
    **extra: Any,
) -> Path:
    """The suite a run records (lite-eval keeps its path and hash in run.json), for the dataset
    `name` under `root/datasets/`; `extra` adds suite keys (cache, judges, ...)."""
    suite = {
        "schema_version": 1,
        "name": f"literag-{name}",
        "pack": pack,
        "dataset": f"../datasets/{name}",
        "targets": [
            {
                "python": "lite_rag.eval:built_in_code",
                "configs": [{"name": config, "schema": schema} for config in configs],
            }
        ],
        "scorers": list(scorers),
        "concurrency": 1,  # lite-rag is synchronous; one at a time keeps latencies honest
        "redact": ["api_keys", "connection_strings"],
        **extra,
    }
    path = root / "suites" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(suite, sort_keys=False, allow_unicode=True), "utf-8")
    return path


def evaluate(
    configs: Sequence[tuple[str, Settings]],
    corpus: Path,
    golden: Path,
    store: CachedStore,
    *,
    root: Path,
    schema: str = "lite_rag_eval",
    embedder_for: EmbedderFor | None = None,
    reranker_for: RerankerFor | None = None,
    keep: bool = False,
    ci: bool | None = None,
    crosscheck: bool = False,
    kb_per_dir: bool = False,
) -> list[RunResult]:
    """One lite-eval run per config, all on `store`, in the order given. The corpus is parsed
    once per PDF loader, embedders and rerankers built once per distinct config; the golden set
    must hold in every loader's text. `ci=None` reads `CI` like lite-eval does."""
    corpora = {pdf: load_corpus(corpus, pdf) for pdf in sorted({s.loader.pdf for _, s in configs})}
    questions = [load_golden(golden, docs) for docs in corpora.values()][-1]  # same ids in each
    names = [slug(name) for name, _ in configs]
    if len(set(names)) != len(names):
        raise ConfigError(f"config names must differ: {names}")
    write_dataset(root, golden, corpus, questions)
    # Latency is a measurement of this run: a cached output would carry an older run's.
    suite_path = write_suite(root, slug(golden.stem), names, schema, cache="off")
    suite = load_suite(suite_path, root)
    dataset = load_dataset(suite.dataset_path)

    groups = corpus_groups(corpus) if kb_per_dir else None
    embedders: dict[str, Embedder] = {}
    rerankers: dict[str, Reranker] = {}
    make_embedder = embedder_for or (lambda c: build_embedder(c, cache=store))
    make_reranker = reranker_for or build_reranker

    def embedder(c: EmbeddingConfig) -> Embedder:
        key = c.model_dump_json(exclude=_TRANSPORT_ONLY)
        if key not in embedders:
            embedders[key] = make_embedder(c)
        return embedders[key]

    def reranker(c: RerankConfig) -> Reranker:
        key = c.model_dump_json()
        if key not in rerankers:
            rerankers[key] = make_reranker(c)
        return rerankers[key]

    targets = [
        (
            name,
            LiteRagTarget(
                settings,
                store=store,
                embedder_for=embedder,
                reranker_for=reranker,
                corpus_for=corpora.__getitem__,
                keep=keep,
                groups=groups,
            ),
        )
        for name, (_, settings) in zip(names, configs, strict=True)
    ]
    options = RunOptions(ci=ci_from_env() if ci is None else ci, crosscheck=crosscheck)
    results = asyncio.run(execute(suite, dataset, targets, options))
    order = {name: i for i, name in enumerate(names)}
    return sorted(results, key=lambda r: order[str(r.record.target.config["name"])])


COLUMNS = (  # header, metric, format
    ("hit@1", "hit@1", ".2f"),
    ("hit@5", "hit@5", ".2f"),
    ("MRR@10", "mrr@10", ".3f"),
    ("doc hit@5", "doc_hit@5", ".2f"),
)


def table(results: Sequence[RunResult]) -> str:
    """A Markdown table, one row per configuration, then hit@5 per question kind."""
    kinds = sorted({k for r in results for k in r.record.summary.groups.get("kind", {})})
    head = ["config", *(c[0] for c in COLUMNS), "p50 ms", "p95 ms", *(f"hit@5 {k}" for k in kinds)]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        s, c = r.record.summary, r.record.cost
        cells = [str(r.record.target.config["name"])]
        cells += [format(s.metrics[m].value, fmt) for _, m, fmt in COLUMNS]
        cells += [format(c.latency_p50_ms or 0.0, ".0f"), format(c.latency_p95_ms or 0.0, ".0f")]
        by_kind = s.groups.get("kind", {})
        cells += [f"{by_kind[k]['hit@5']:.2f}" if k in by_kind else "—" for k in kinds]
        rows.append("| " + " | ".join(cells) + " |")
    n = results[0].record.summary.n_cases if results else 0
    scopes = {
        str(r.record.target.config["name"]): r.record.target.info.get("latency_scope", "?")
        for r in results
    }
    if len(set(scopes.values())) == 1:
        latency = next(iter(scopes.values()))
    else:  # embedding.cache differs between configs: their latencies measure different work
        latency = "; ".join(f"{name}: {scope}" for name, scope in scopes.items())
    return "\n".join(rows) + (
        f"\n\n{n} questions; a hit answers when it is from the right document and contains the "
        f"evidence quote. Search depth {TOP_K}. Latency: {latency}. Runs are lite-eval run "
        "directories: compare two with `lite-eval compare <run> <run>`."
    )


def outcomes(result: RunResult) -> list[dict[str, Any]]:
    """Per question: answer_rank, doc_rank and latency."""
    latency = {o.case_id: o.trace.latency_ms for o in load_run(result.directory).outputs}
    return [
        {
            "id": s.case_id,
            "answer_rank": s.evidence.get("answer_rank"),
            "doc_rank": s.evidence.get("doc_rank"),
            "latency_ms": latency.get(s.case_id),
        }
        for s in result.scores
        if s.scorer == "hit@1" and isinstance(s.evidence, dict)
    ]


def report(result: RunResult) -> dict[str, Any]:
    """`literag eval --out`: the lite-eval run record plus each question's ranks."""
    return {
        "run": str(result.directory),
        "record": result.record.model_dump(mode="json"),
        "outcomes": outcomes(result),
    }
