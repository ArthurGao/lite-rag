"""S14: each golden question's `key_fact` (a case-insensitive regex) is what a correct answer
must state. Labels by construction: it matches the question's correct answer, the evidence
sentence of the grounded judge's calibration set, or the evidence itself where that set has no
base. (`load_golden` refuses a key fact that is not a regex or matches its own question.)"""

import json
import re
from pathlib import Path

from lite_rag.eval import load_corpus, load_golden

REPO = Path(__file__).resolve().parents[2]
QUESTIONS = load_golden(REPO / "eval" / "golden.jsonl", load_corpus(REPO / "eval" / "corpus"))
CASES = REPO / "eval/judge-grounded/datasets/grounded-golden/cases.jsonl"
BASES = {
    c["tags"]["question"]: c["input"]["answer"]
    for c in map(json.loads, CASES.read_text().splitlines())
    if c["tags"]["construction"] == "none"
}


def test_every_question_has_a_key_fact() -> None:
    assert [q.id for q in QUESTIONS if not q.key_fact] == []


def test_a_key_fact_matches_the_correct_answer() -> None:
    for q in QUESTIONS:
        assert q.key_fact is not None
        correct = BASES.get(q.id, q.evidence)
        assert re.search(q.key_fact, correct, re.IGNORECASE), (q.id, q.key_fact, correct)


WRONG = [  # wrong answers a key fact can tell apart: a changed figure, another question's answer
    c
    for c in map(json.loads, CASES.read_text().splitlines())
    if c["tags"]["construction"] in {"change_number", "swap_answer"}
    and c["id"] != "q12-change_number"  # it changes the tilt; q12 asks which planet: still right
]


def test_a_key_fact_does_not_match_a_changed_figure_or_another_answer() -> None:
    # Review of S14: `16` matched "7 gram of coffee to 16 grams", `不予受理` matched "超过 38 天".
    # A reversed claim ("grind coarser") still matches: polarity is the grounded judge's job,
    # so the headline is "key fact AND grounded", never the key fact alone.
    facts = {q.id: q.key_fact for q in QUESTIONS}
    passing = [
        (c["id"], c["input"]["answer"])
        for c in WRONG
        if re.search(facts[c["tags"]["question"]] or "", c["input"]["answer"], re.IGNORECASE)
    ]
    assert passing == []
