"""The grounded judge's calibration set (design §8, S13; lite-eval design §9, §18.24).

Labels are fixed by construction (methodology §2): for each golden question, the passing answer
is the sentence of its document that holds the evidence quote, citing the paragraph it comes from
as source [1] — grounded by construction. Each answer is then changed in one known way with
lite-eval's construction tools: a figure changed, a sentence from another document added, another
question's answer swapped in (all unsupported), or its citation removed (still grounded: the
rubric does not judge citation markers).

The set is generated, committed and checked against the generator in a test:

    uv run python eval/make_judge_grounded.py

Construction inherits the shape of the base answers (one short sentence from one paragraph), so a
judge calibrated here is also checked on real `literag ask` answers before it is trusted (L2).
"""

import asyncio
import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from lite_eval.config import ci_from_env
from lite_eval.contracts import Case, Output, Trace, Usage
from lite_eval.datasets.local import load_dataset
from lite_eval.files import read_jsonl
from lite_eval.judges.client import JudgeClient
from lite_eval.judges.construct import (
    Constructed,
    change_number,
    drop_citations,
    insert_sentence,
    swap_answer,
)
from lite_eval.judges.spec import JudgeSpec
from lite_eval.runner import RunOptions, RunResult, execute
from lite_eval.scorers.ranking import normalise
from lite_eval.store import read_record, root_of
from lite_eval.suite import load_suite

from lite_rag.config import Settings
from lite_rag.embedding import build_embedder
from lite_rag.errors import ConfigError
from lite_rag.eval import (
    Corpus,
    EmbedderFor,
    LiteRagTarget,
    LlmFor,
    Question,
    billing,
    golden_cases,
    load_corpus,
    load_golden,
    slug,
    write_generated_dataset,
    write_suite,
)
from lite_rag.llm import build_llm
from lite_rag.rerank import build_reranker
from lite_rag.stores.base import CachedStore

SEED = 20261118  # the committed set's seed; another seed is another calibration set

# A sentence ends at English end marks followed by space, or right after Chinese ones. Narrower
# than chunking's splitter on purpose (that one also splits at `;` and line ends): a base answer
# is one whole sentence as a reader would quote it.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")
_NOT_PROSE = ("#", "-", "*", "|", ">", "`", "<!--")


def sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()]


def _paragraphs(text: str) -> list[str]:
    """Prose paragraphs: blank-line separated, headings and lists left out."""
    blocks = (" ".join(b.split()) for b in re.split(r"\n\s*\n", text))
    return [b for b in blocks if b and not b.startswith(_NOT_PROSE)]


def _cite(sentence: str) -> str:
    """`[1]` after the sentence: spaced in English, directly after a Chinese end mark."""
    return f"{sentence}[1]" if sentence.endswith(("。", "！", "？")) else f"{sentence} [1]"


@dataclass(frozen=True)
class Base:
    question: Question
    source: str  # the paragraph holding the evidence: source [1]
    answer: str  # its sentence holding the evidence, citing [1]


def grounded_base(q: Question, corpus: Corpus) -> Base | None:
    """None when the evidence is not inside one sentence of one paragraph (no clean base)."""
    want = normalise(q.evidence)
    for paragraph in _paragraphs(corpus[q.doc][1]):
        for sentence in sentences(paragraph):
            if want in normalise(sentence):
                return Base(q, paragraph, _cite(sentence))
    return None


def _foreign_sentences(corpus: Corpus, doc_lang: dict[str, str]) -> list[tuple[str, str]]:
    """(doc, sentence) a corruption can add: prose long enough to be a claim of its own, from
    the documents whose language the golden set gives."""
    found = []
    for doc, lang in doc_lang.items():
        for paragraph in _paragraphs(corpus[doc][1]):
            for s in sentences(paragraph):
                if len(s) >= (12 if lang == "zh" else 30):
                    found.append((doc, s))
    return found


Hand = dict[str, dict[str, str]]  # question id -> {contradict, elaborate}


def load_hand(path: Path) -> Hand:
    """Hand-written corruptions closer to real errors than the mechanical ones (S13, after the
    first real-answer check): per question, `contradict` (the answer with one statement the
    source makes reversed) and `elaborate` (a plausible claim no source makes)."""
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    _unique([r["question"] for r in rows], path.name)
    return {r["question"]: {k: r[k] for k in ("contradict", "elaborate")} for r in rows}


def _unique(ids: list[str], where: str) -> None:
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"{where}: repeated question ids {repeated}")


def _case(
    base: Base, construction: str, answer: str, label: str, from_doc: str | None = None
) -> dict[str, Any]:
    q = base.question
    tags = {"construction": construction, "question": q.id, "kind": q.kind}
    tags |= {"lang": q.lang} if q.lang else {}
    tags |= {"from_doc": from_doc} if from_doc else {}
    return {
        "id": f"{q.id}-{construction}",
        "input": {"question": q.question, "answer": answer, "sources": [{"text": base.source}]},
        "expected": {"label": label},
        "tags": tags,
    }


def _made(base: Base, made: Constructed, from_doc: str | None = None) -> dict[str, Any]:
    return _case(base, made.construction, made.answer, made.label, from_doc)


def calibration_cases(
    questions: list[Question], corpus: Corpus, *, seed: int, hand: Hand | None = None
) -> list[dict[str, Any]]:
    """Every question with a clean base: the base, then each construction that applies. The
    added sentence and the swapped answer come from another document in the question's language
    (a document's language is its golden questions', never guessed from the script: an English
    sentence may quote a Chinese character). `hand` adds a contradiction of the source and a
    plausible elaboration per question, both unsupported."""
    rng = random.Random(seed)
    doc_lang = {q.doc: q.lang for q in questions}
    bases = [b for q in questions if (b := grounded_base(q, corpus)) is not None]
    pool = _foreign_sentences(corpus, doc_lang)
    cases = []
    for base in bases:
        doc, lang = base.question.doc, base.question.lang
        cases.append(_case(base, "none", base.answer, "grounded"))
        cases.append(_made(base, drop_citations(base.answer)))
        changed = change_number(base.answer, rng)
        if changed is not None:
            cases.append(_made(base, changed))
        foreign = [(d, s) for d, s in pool if d != doc and doc_lang[d] == lang]
        from_doc, sentence = rng.choice(foreign)
        cases.append(_made(base, insert_sentence(base.answer, sentence), from_doc))
        others = [b for b in bases if b.question.doc != doc and b.question.lang == lang]
        other = rng.choice(others)
        swapped = swap_answer(base.answer, [other.answer], rng)
        assert swapped is not None  # another document's answer is never this one
        cases.append(_made(base, swapped, other.question.doc))
        if hand is not None:
            h = hand[base.question.id]
            cases.append(_case(base, "contradict_source", h["contradict"], "unsupported"))
            elaborated = insert_sentence(base.answer, h["elaborate"]).answer
            cases.append(_case(base, "plausible_elaboration", elaborated, "unsupported"))
    return cases


def real_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Real `literag ask` answers labelled by reading them (L2): every source they were given;
    a row labelled `unclear` stays on file and is left out (methodology §2)."""
    labels = {"grounded", "unsupported", "unclear"}
    if wrong := sorted({r["label"] for r in rows} - labels):
        raise ValueError(f"real answers: unknown labels {wrong}, expected one of {sorted(labels)}")
    _unique([r["id"] for r in rows], "real answers")
    return [
        {
            "id": f"{r['id']}-real",
            "input": {"question": r["question"], "answer": r["answer"], "sources": r["sources"]},
            "expected": {"label": r["label"]},
            "tags": {"construction": "real", "question": r["id"]},
        }
        for r in rows
        if r["label"] != "unclear"
    ]


def write_calibration(
    root: Path, golden: Path, corpus_dir: Path, *, seed: int, hand: Path, real: Path
) -> Path:
    """The calibration set as a lite-eval judging dataset under `root/datasets/`: the
    constructed cases (mechanical and hand-written), then the real slice."""
    corpus = load_corpus(corpus_dir)
    questions = load_golden(golden, corpus)
    cases = calibration_cases(questions, corpus, seed=seed, hand=load_hand(hand))
    used = {c["tags"]["question"] for c in cases}
    left_out = [q.id for q in questions if q.id not in used]
    rows = [json.loads(line) for line in real.read_text("utf-8").splitlines() if line.strip()]
    cases += real_cases(rows)
    name = f"grounded-{slug(golden.stem)}"
    left = f"; left out (evidence not in one sentence): {', '.join(left_out)}" if left_out else ""
    meta = {
        "name": name,
        "version": f"seed{seed}",
        "pack": "judging",
        "description": (
            f"grounded-judge calibration set from {golden.name} over {corpus_dir.name}/ "
            f"(generated by eval/make_judge_grounded.py, seed {seed}): labels by construction, "
            f"mechanical and hand-written ({hand.name}), and real answers labelled by reading "
            f"them ({real.name})" + left
        ),
        "license": "Apache-2.0",
    }
    return write_generated_dataset(root / "datasets" / name, meta, cases)


# ---------------------------------------------------------------- judging real answers

# LLM settings that never change an answer (how it is fetched, not what it is).
_LLM_TRANSPORT = {"api_key_env", "timeout_s", "connect_timeout_s", "max_retries"}
JudgeFactory = Callable[[JudgeSpec, float], JudgeClient]


class LiteRagAnswerTarget(LiteRagTarget):
    """One lite-rag config answering with `ask` (design §6): `{question, answer, sources: [{text,
    doc, id}], status, dropped}`, which lite-eval's grounded judge reads. Builds and drops its kb
    like the retrieval target (give it `llm_for`); latency covers the search and the answer."""

    name = "lite-rag-answer"
    query_key = "question"  # what `ask` searches for and the judge reads

    def config(self) -> dict[str, Any]:
        return {**super().config(), "llm": self.settings.llm.model_dump(exclude=_LLM_TRANSPORT)}

    def info(self) -> dict[str, str]:
        return {"latency_scope": "search (+ rerank) and one answer call"}

    def billing(self) -> Literal["free", "per_token", "unknown"]:
        return billing(self.settings, answers=True)

    async def run(self, case: Case) -> Output:
        assert self.engine is not None
        question = self.query(case)
        start = time.perf_counter()
        answer = self.engine.ask(self.kb, question)
        latency_ms = (time.perf_counter() - start) * 1000
        sources = [{"text": h.text, "doc": h.doc_id, "id": h.chunk_id} for h in answer.sources]
        usage = Usage(
            input_tokens=answer.usage.get("input", 0), output_tokens=answer.usage.get("output", 0)
        )
        value = {
            "question": question,
            "answer": answer.text,
            "sources": sources,
            "status": answer.status,
            "dropped": answer.dropped,
        }
        scope = self.info()["latency_scope"]
        return Output(
            case_id=case.id,
            value=value,
            trace=Trace(usage=usage, latency_ms=latency_ms, latency_scope=scope),
        )


def calibrated_judge(calibration: Path) -> dict[str, Any]:
    """The judge a calibration run measured, as a suite `judges:` entry: its config in the
    calibration suite, which holds what the record leaves out (key variable, price, the URL as
    written). lite-eval then checks it against the run, key for key, before any call."""
    record = read_record(calibration)
    if record.target.name != "judge" or record.suite.path is None:
        raise ConfigError(f"{calibration} is not a judge calibration run")
    suite = yaml.safe_load((root_of(calibration) / record.suite.path).read_text("utf-8"))
    name = record.target.config.get("name")
    for target in suite.get("targets", []):
        for config in target.get("configs", []):
            if config.get("name") == name:
                return {k: v for k, v in config.items() if k not in {"name", "rubric"}}
    raise ConfigError(f"{calibration}: config {name!r} is not in its suite {record.suite.path}")


def judge_answers(
    name: str,
    settings: Settings,
    corpus_dir: Path,
    golden: Path,
    store: CachedStore,
    *,
    root: Path,
    calibration: Path,
    min_agreement: float,
    schema: str = "lite_rag_eval",
    limit: int | None = None,
    embedder_for: EmbedderFor | None = None,
    llm_for: LlmFor | None = None,
    judge_factory: JudgeFactory | None = None,
) -> RunResult:
    """Answer the golden questions with one config and let the calibrated grounded judge score
    each answer (L2: the flag rate on real outputs, before the judge is trusted). The judge is
    the one `calibration` measured; lite-eval refuses it, before any call, when that run is
    weaker than `min_agreement`."""
    corpus = load_corpus(corpus_dir, settings.loader.pdf)
    questions = load_golden(golden, corpus)[:limit]
    dataset = f"answers-{slug(golden.stem)}"
    meta = {
        "name": dataset,
        "version": "1",
        "pack": "grounded",
        "description": f"lite-rag golden questions {golden.name}, answered by `ask` (generated)",
        "license": "Apache-2.0",
    }
    write_generated_dataset(root / "datasets" / dataset, meta, golden_cases(questions, "question"))
    run = os.path.relpath(calibration.resolve(), root.resolve())
    scorer = {
        "judge": {"rubric": "grounded", "calibration": {"run": run, "min_agreement": min_agreement}}
    }
    suite_path = write_suite(
        root,
        dataset,
        [slug(name)],
        schema,
        pack="grounded",
        scorers=[scorer],
        judges={"grounded": calibrated_judge(calibration)},
        judge_timeout_s=300,
    )
    suite = load_suite(suite_path, root)
    target = LiteRagAnswerTarget(
        settings,
        store=store,
        embedder_for=embedder_for or (lambda c: build_embedder(c, cache=store)),
        reranker_for=build_reranker,
        corpus_for=lambda pdf: corpus,
        llm_for=llm_for or build_llm,
    )
    options = RunOptions(ci=ci_from_env(), judge_factory=judge_factory)
    (result,) = asyncio.run(
        execute(suite, load_dataset(suite.dataset_path), [(slug(name), target)], options)
    )
    return result


def flagged_answers(result: RunResult) -> list[dict[str, Any]]:
    """Every answer the judge did not pass (a scoring error included), with what it quoted as
    unsupported, in case order: each one is read before the judge is trusted (methodology §3:
    is the judge wrong, or the answer?)."""
    failed = (s for s in result.scores if not s.passed)
    flagged = sorted(failed, key=lambda s: (s.case_id, s.repeat))
    if not flagged:
        return []
    outputs = {
        (o.case_id, o.repeat): o for o in read_jsonl(result.directory / "outputs.jsonl", Output)
    }
    rows = []
    for score in flagged:
        value = outputs[(score.case_id, score.repeat)].value
        shown = value if isinstance(value, dict) else {}
        evidence = score.evidence if isinstance(score.evidence, dict) else {}
        rows.append(
            {
                "id": score.case_id,
                "question": shown.get("question"),
                "answer": shown.get("answer"),
                "reason": score.reason,
                "evidence": evidence.get("evidence", []),
            }
        )
    return rows
