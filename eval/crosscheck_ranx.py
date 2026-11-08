"""Cross-check lite-rag's metrics against ranx, an independent IR-metrics library.

Not a dependency: run it on demand (ranx pulls in pandas, numba, seaborn…):

    uv run --with ranx==0.3.21 python eval/crosscheck_ranx.py eval/out/s5-baseline.json

It rebuilds, from a saved `literag eval --out` report, one run per configuration in which the
answering chunk is the single relevant document of each question, and compares hit@1, hit@5 and
MRR@10 with what lite-rag reported. Exit code 1 on any difference beyond 1e-9. It re-checks the
arithmetic of `metrics()`; whether a hit answers a question (`judge()`) is the unit tests' job.
"""

import json
import sys
from pathlib import Path

from ranx import Qrels, Run, evaluate


def main(path: str) -> int:
    reports = json.loads(Path(path).read_text("utf-8"))
    failures = 0
    for report in reports:
        qrels: dict[str, dict[str, int]] = {}
        run: dict[str, dict[str, float]] = {}
        for outcome in report["outcomes"]:
            qid = outcome["question"]["id"]
            qrels[qid] = {"answer": 1}
            rank = outcome["answer_rank"]
            # ranks 1..10, the answer at its rank (if found), filler documents elsewhere
            run[qid] = {("answer" if rank == r else f"other-{r}"): 1.0 / r for r in range(1, 11)}
        theirs = evaluate(Qrels(qrels), Run(run), ["hit_rate@1", "hit_rate@5", "mrr@10"])
        ours = report["overall"]
        pairs = {
            "hit@1": (ours["hit_at_1"], theirs["hit_rate@1"]),
            "hit@5": (ours["hit_at_5"], theirs["hit_rate@5"]),
            "MRR@10": (ours["mrr_at_10"], theirs["mrr@10"]),
        }
        line = "  ".join(f"{k} {a:.4f}/{b:.4f}" for k, (a, b) in pairs.items())
        bad = [k for k, (a, b) in pairs.items() if abs(a - b) > 1e-9]
        failures += bool(bad)
        print(f"{report['config']:16} {line}  {'MISMATCH ' + ','.join(bad) if bad else 'ok'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
