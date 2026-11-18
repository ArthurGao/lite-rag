"""The grounded-judge calibration set (design §8, lite-eval design §9): labels fixed by
construction from the golden questions, the same set for the same seed, and the committed set
equal to what the generator makes."""

import json
from collections import Counter
from pathlib import Path

from lite_eval.scorers.ranking import normalise

from lite_rag.eval import load_corpus, load_golden
from lite_rag.eval_grounded import (
    SEED,
    calibration_cases,
    grounded_base,
    load_hand,
    real_cases,
    sentences,
    write_calibration,
)

EVAL = Path(__file__).parents[2] / "eval"
JUDGE = EVAL / "judge-grounded"
CORPUS = load_corpus(EVAL / "corpus")
QUESTIONS = load_golden(EVAL / "golden.jsonl", CORPUS)
BY_ID = {q.id: q for q in QUESTIONS}
HAND = load_hand(JUDGE / "hand-corruptions.jsonl")


def test_sentences_split_on_english_and_chinese_ends() -> None:
    split = sentences("One. Two? Three! 四。五？六！")
    assert split == ["One.", "Two?", "Three!", "四。", "五？", "六！"]
    assert sentences("No end mark") == ["No end mark"]


def test_a_base_answer_is_the_evidence_sentence_citing_its_paragraph() -> None:
    base = grounded_base(BY_ID["q01"], CORPUS)
    assert base is not None
    assert base.answer == "Start with a ratio of 1 gram of coffee to 16 grams of water. [1]"
    assert base.source.startswith("Start with a ratio of 1 gram")
    assert base.source.endswith("if it tastes bitter, grind coarser.")
    zh = grounded_base(BY_ID[next(q.id for q in QUESTIONS if "600 元" in q.evidence)], CORPUS)
    assert zh is not None
    assert zh.answer == "一线城市每晚住宿上限为 600 元，其他城市每晚上限为 400 元。[1]"


def test_each_question_gets_its_constructions_with_their_labels() -> None:
    cases = calibration_cases(QUESTIONS, CORPUS, seed=SEED, hand=HAND)
    q01 = {c["tags"]["construction"]: c for c in cases if c["id"].startswith("q01-")}
    assert {k: c["expected"]["label"] for k, c in q01.items()} == {
        "none": "grounded",
        "drop_citations": "grounded",
        "change_number": "unsupported",
        "insert_sentence": "unsupported",
        "swap_answer": "unsupported",
        "contradict_source": "unsupported",
        "plausible_elaboration": "unsupported",
    }
    base = q01["none"]["input"]
    assert base["question"] == BY_ID["q01"].question
    q01_base = grounded_base(BY_ID["q01"], CORPUS)
    assert q01_base is not None and base["sources"] == [{"text": q01_base.source}]
    assert q01["drop_citations"]["input"]["answer"] == base["answer"].removesuffix(" [1]")
    assert q01["change_number"]["input"]["answer"] != base["answer"]
    inserted = q01["insert_sentence"]["input"]["answer"]
    assert inserted.startswith(base["answer"] + " ") and len(inserted) > len(base["answer"]) + 20
    swapped = q01["swap_answer"]["input"]["answer"]
    assert swapped != base["answer"] and q01["swap_answer"]["input"]["sources"] == base["sources"]
    assert q01["contradict_source"]["input"]["answer"] == HAND["q01"]["contradict"]
    elaborated = q01["plausible_elaboration"]["input"]["answer"]
    assert elaborated == base["answer"] + " " + HAND["q01"]["elaborate"]


def test_the_corruptions_stay_in_the_question_s_language_and_outside_its_document() -> None:
    """The added sentence and the swapped answer come from another document in the question's
    language (from the golden set, not guessed from the script) and never from its source."""
    doc_lang = {q.doc: q.lang for q in QUESTIONS}
    cases = calibration_cases(QUESTIONS, CORPUS, seed=SEED)
    foreign = [c for c in cases if c["tags"]["construction"] in {"insert_sentence", "swap_answer"}]
    assert len(foreign) == 2 * 31
    for c in foreign:
        q = BY_ID[c["tags"]["question"]]
        assert c["tags"]["from_doc"] != q.doc and doc_lang[c["tags"]["from_doc"]] == q.lang, c["id"]


def test_no_corruption_s_new_content_is_in_its_source() -> None:
    """Unsupported by construction: what was added, swapped in or rewritten is not in the
    source text, nor in the whole document (a hand-written elaboration must be new)."""
    added = {"insert_sentence", "plausible_elaboration"}
    replaced = {"swap_answer", "contradict_source"}
    for c in calibration_cases(QUESTIONS, CORPUS, seed=SEED, hand=HAND):
        construction = c["tags"]["construction"]
        if construction not in added | replaced:
            continue
        q = BY_ID[c["tags"]["question"]]
        base = grounded_base(q, CORPUS)
        assert base is not None
        answer = c["input"]["answer"]
        new = answer.removeprefix(base.answer) if construction in added else answer
        assert normalise(new.replace("[1]", "")) not in normalise(CORPUS[q.doc][1]), c["id"]


def test_a_mixed_script_sentence_keeps_its_document_s_language() -> None:
    """An English sentence quoting a Chinese character is still English, and a Chinese one
    naming an English term still Chinese: language is the document's, not the script's."""
    from lite_rag.eval import Question

    corpus = {
        "zh": ("中文", "请用 Python 3.12 版本运行这个项目，并且先创建虚拟环境。"),
        "zh2": ("中文二", "茶叶应该放在阴凉干燥的地方保存，避免阳光直射。"),
        "en": ("English", "The character 茶 is read cha and means tea in many languages."),
        "en2": ("English 2", "The sign 水 marks drinking water at the station on every floor."),
    }
    questions = [
        Question(id="a", question="?", doc="zh2", evidence="阴凉干燥", lang="zh"),
        Question(id="b", question="?", doc="zh", evidence="Python 3.12", lang="zh"),
        Question(id="c", question="?", doc="en2", evidence="drinking water", lang="en"),
        Question(id="d", question="?", doc="en", evidence="means tea", lang="en"),
    ]
    for seed in range(5):
        for c in calibration_cases(questions, corpus, seed=seed):
            if "from_doc" in c["tags"]:
                q = {x.id: x for x in questions}[c["tags"]["question"]]
                assert c["tags"]["from_doc"][:2] == q.doc[:2], (seed, c["id"])


def test_the_set_is_fixed_by_its_seed_and_counts_every_construction() -> None:
    first = calibration_cases(QUESTIONS, CORPUS, seed=SEED, hand=HAND)
    assert first == calibration_cases(QUESTIONS, CORPUS, seed=SEED, hand=HAND)
    assert first != calibration_cases(QUESTIONS, CORPUS, seed=SEED + 1, hand=HAND)
    counts = Counter(c["tags"]["construction"] for c in first)
    n = sum(grounded_base(q, CORPUS) is not None for q in QUESTIONS)
    assert n == len(QUESTIONS) - 1  # q07's evidence spans two sentences: no clean base
    assert counts["none"] == counts["drop_citations"] == n
    assert counts["insert_sentence"] == counts["swap_answer"] == n
    assert 0 < counts["change_number"] < n  # only answers with a figure in digits
    assert counts["contradict_source"] == counts["plausible_elaboration"] == n  # one each
    assert len({c["id"] for c in first}) == len(first)


def test_the_committed_set_is_what_the_generator_makes(tmp_path: Path) -> None:
    """eval/judge-grounded/ is generated (`uv run python eval/make_judge_grounded.py`): a
    changed corpus, golden set or generator must regenerate it."""
    made = write_calibration(
        tmp_path,
        EVAL / "golden.jsonl",
        EVAL / "corpus",
        seed=SEED,
        hand=JUDGE / "hand-corruptions.jsonl",
        real=JUDGE / "real-answers.jsonl",
    )
    committed = EVAL / "judge-grounded" / "datasets" / made.name
    for name in ("dataset.yaml", "cases.jsonl"):
        assert (made / name).read_text("utf-8") == (committed / name).read_text("utf-8"), name
    first = json.loads((made / "cases.jsonl").read_text("utf-8").splitlines()[0])
    assert set(first) == {"id", "input", "expected", "tags"}


def test_a_question_whose_evidence_spans_two_paragraphs_is_left_out() -> None:
    from lite_rag.eval import Question

    corpus = {"d": ("D", "First part of the\n\nquote here.")}
    q = Question(id="x", question="q?", doc="d", evidence="the quote")
    assert grounded_base(q, corpus) is None


def test_an_answer_run_counts_the_answer_model_in_its_billing() -> None:
    """lite-eval's CI guard runs only `free` targets: an answer run calls the LLM too."""
    from lite_rag.config import load_settings
    from lite_rag.eval import billing

    local = load_settings()
    assert billing(local) == billing(local, answers=True) == "free"
    gateway = load_settings(llm={"provider": "openai", "base_url": "http://127.0.0.1:8787/v1"})
    assert billing(gateway) == "free" and billing(gateway, answers=True) == "unknown"
    anthropic = load_settings(llm={"provider": "anthropic", "model": "m"})
    assert billing(anthropic, answers=True) == "per_token"


def test_hand_corruptions_cover_every_base_and_nothing_else() -> None:
    bases = {q.id for q in QUESTIONS if grounded_base(q, CORPUS) is not None}
    assert set(HAND) == bases
    assert all(set(h) == {"contradict", "elaborate"} for h in HAND.values())


def test_real_answers_keep_their_sources_and_skip_unclear_labels() -> None:
    """The real slice (L2): answers `literag ask` gave, labelled by reading them; an answer
    whose label is unclear is kept on file but not used."""
    rows = [
        json.loads(line) for line in (JUDGE / "real-answers.jsonl").read_text("utf-8").splitlines()
    ]
    cases = {c["id"]: c for c in real_cases(rows)}
    unclear = {r["id"] for r in rows if r["label"] == "unclear"}
    assert unclear == {"q03", "q28"} and not any(f"{u}-real" in cases for u in unclear)
    assert len(cases) == len(rows) - len(unclear)
    q20 = cases["q20-real"]
    assert q20["expected"] == {"label": "unsupported"} and q20["tags"]["construction"] == "real"
    assert len(q20["input"]["sources"]) > 1  # every source the answer was given
    assert cases["q27-real"]["expected"] == {"label": "grounded"}


def test_a_duplicate_hand_row_is_refused(tmp_path: Path) -> None:
    """A second row for one question would silently replace the first."""
    import pytest

    row = {"question": "q01", "contradict": "a", "elaborate": "b"}
    path = tmp_path / "hand.jsonl"
    path.write_text("\n".join(json.dumps(row) for _ in range(2)) + "\n", "utf-8")
    with pytest.raises(ValueError, match="q01"):
        load_hand(path)


def test_a_real_row_with_an_unknown_label_or_a_repeated_id_is_refused() -> None:
    import pytest

    row = {"id": "q01", "question": "?", "answer": "a", "sources": [], "label": "grounded"}
    with pytest.raises(ValueError, match="supported"):
        real_cases([row | {"label": "supported"}])
    with pytest.raises(ValueError, match="q01"):
        real_cases([row, row | {"label": "unsupported"}])


def test_every_real_row_with_a_label_becomes_one_case() -> None:
    rows = [
        json.loads(line) for line in (JUDGE / "real-answers.jsonl").read_text("utf-8").splitlines()
    ]
    ids = [c["id"] for c in real_cases(rows)]
    assert ids == [f"{r['id']}-real" for r in rows if r["label"] != "unclear"]
