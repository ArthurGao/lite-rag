"""How much a per-question retrieval router could gain at most (design §12 S21): reads stored
lite-eval runs, never runs a model. For each golden set, the latest run of each candidate config;
per question, the oracle picks the config with the best hit@5, then mrr@10, then the lowest
latency. Prints the best fixed config, the oracle, and the latency the oracle saves.

    uv run python eval/route_ceiling.py [--runs eval/lite-eval/runs]
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from lite_eval.store import load_run

SUITES = {
    "literag-golden": "eval/golden.jsonl",
    "literag-golden-exact": "eval/golden-exact.jsonl",
    "literag-golden-multihop": "eval/golden-multihop.jsonl",
}
# config name -> what a router would choose (mode, rerank)
CANDIDATES = {
    "chunk-markdown": "hybrid",
    "hybrid-rerank": "hybrid+rerank",
    "mode-vector": "vector",
    "mode-bm25": "bm25",
    "graph-hybrid": "graph",
    "graph-only": "graph-only",
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _latest(suite_dir: Path, config: str) -> Path | None:
    """The newest complete run of `config` (run directories start with their UTC start time)."""
    runs = [
        d
        for d in sorted(suite_dir.iterdir())
        if (r := load_run(d, outputs=False).record).config_name == config and r.status == "complete"
    ]
    return runs[-1] if runs else None


def _per_case(run: Path) -> dict[str, dict[str, float]]:
    stored = load_run(run)
    cases: dict[str, dict[str, float]] = {}
    for s in stored.scores:
        cases.setdefault(s.case_id, {})[s.scorer] = float(s.value or 0.0)
    for o in stored.outputs:
        cases.setdefault(o.case_id, {})["latency_ms"] = o.trace.latency_ms or 0.0
    return cases


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "hit@5": sum(r["hit@5"] for r in rows),
        "mrr@10": round(statistics.mean(r["mrr@10"] for r in rows), 3),
        "p50_ms": round(statistics.median(r["latency_ms"] for r in rows), 1),
    }


def _key(row: dict[str, float]) -> tuple[float, float, float]:
    return (row["hit@5"], row["mrr@10"], -row["latency_ms"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("eval/lite-eval/runs"))
    args = parser.parse_args()
    report = {}
    for suite, golden in SUITES.items():
        kinds = {c["id"]: c["kind"] for c in _jsonl(Path(golden))}
        table = {}
        for config, label in CANDIDATES.items():
            run = _latest(args.runs / suite, config)
            if run is not None:
                table[label] = (run.name, _per_case(run))
        ids = sorted(kinds)
        fixed = {label: _summary([cases[i] for i in ids]) for label, (_, cases) in table.items()}
        best_label = max(
            fixed, key=lambda k: (fixed[k]["hit@5"], fixed[k]["mrr@10"], -fixed[k]["p50_ms"])
        )
        picks = {i: max(table, key=lambda k: _key(table[k][1][i])) for i in ids}
        oracle_rows = [table[picks[i]][1][i] for i in ids]
        # the same hit@5 as the best fixed config per question, at the lowest latency
        best = table[best_label][1]
        cheap = {
            i: min(
                (k for k in table if table[k][1][i]["hit@5"] >= best[i]["hit@5"]),
                key=lambda k: table[k][1][i]["latency_ms"],
            )
            for i in ids
        }
        report[suite] = {
            "cases": len(ids),
            "runs": {label: run for label, (run, _) in table.items()},
            "fixed": fixed,
            "best_fixed": best_label,
            "oracle": _summary(oracle_rows) | {"picks": dict(Counter(picks.values()))},
            "same_hit_cheapest": {
                "p50_ms": round(
                    statistics.median(table[cheap[i]][1][i]["latency_ms"] for i in ids), 1
                ),
                "picks": dict(Counter(cheap.values())),
            },
            "by_kind": {
                kind: {
                    label: sum(cases[i]["hit@5"] for i in ids if kinds[i] == kind)
                    for label, (_, cases) in table.items()
                }
                | {"n": sum(1 for i in ids if kinds[i] == kind)}
                for kind in sorted(set(kinds.values()))
            },
        }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
