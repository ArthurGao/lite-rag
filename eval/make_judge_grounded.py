"""Regenerate the grounded judge's calibration set (design §8, S13): committed under
eval/judge-grounded/datasets/, checked against this generator by tests/unit/test_eval_grounded.py.

    uv run python eval/make_judge_grounded.py
"""

from pathlib import Path

from lite_rag.eval_grounded import SEED, write_calibration

EVAL = Path(__file__).parent

if __name__ == "__main__":
    judge = EVAL / "judge-grounded"
    made = write_calibration(
        judge,
        EVAL / "golden.jsonl",
        EVAL / "corpus",
        seed=SEED,
        hand=judge / "hand-corruptions.jsonl",
        real=judge / "real-answers.jsonl",
    )
    print(f"wrote {made}")
