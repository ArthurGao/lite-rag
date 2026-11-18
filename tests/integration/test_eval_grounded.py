"""Judging real `literag ask` answers (design §8, S13): the answer target builds the kb like the
retrieval eval, answers each golden question with the LLM, and the grounded judge named by a
calibration run scores them. Fakes only: FakeEmbedder, FakeLLM, lite-eval's FakeJudge."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from lite_eval.datasets.local import load_dataset
from lite_eval.errors import ConfigError as LiteEvalConfigError
from lite_eval.judges.fake import FakeJudge, by_marker
from lite_eval.judges.target import JudgeTarget
from lite_eval.runner import RunOptions, RunResult, execute
from lite_eval.store import load_run
from lite_eval.suite import load_suite

from lite_rag.config import load_settings
from lite_rag.eval_grounded import flagged_answers, judge_answers
from lite_rag.stores.pg import PgStore
from lite_rag.testing import FakeEmbedder, FakeLLM

REPO = Path(__file__).resolve().parents[2]
CORPUS, GOLDEN = REPO / "eval" / "corpus", REPO / "eval" / "golden.jsonl"
SETTINGS = load_settings(chunking={"unit": "chars", "size": 300, "overlap": 30})
MARK = "and the moon is made of cheese"  # what the fake answer model adds to some answers
JUDGE: dict[str, Any] = {
    "rubric": "grounded",
    "provider": "ollama",
    "model": "qwen3:8b",
    "base_url": "http://127.0.0.1:1",  # never reached: FakeJudge
    "think": False,
    "billing": "free",
}


def fake_judge(spec: Any, timeout_s: float | None) -> FakeJudge:
    return FakeJudge(spec, by_marker(MARK, "unsupported", "grounded"))


def answer(messages: list[Any]) -> str:
    """Cites source [1]; questions about coffee also get the made-up claim."""
    question = str(messages[-1].content).rsplit("Question:", 1)[1]  # not the sources above it
    return f"From the sources [1]{' ' + MARK if 'coffee' in question.lower() else ''}."


def calibrate(root: Path, judge: Any = fake_judge) -> RunResult:
    """A judging run of the fake judge on two labelled cases, as the calibration to name."""
    dataset = root / "datasets" / "cal"
    dataset.mkdir(parents=True)
    (dataset / "dataset.yaml").write_text("name: cal\nversion: '1'\npack: judging\n")
    sources = [{"text": "Wellington is the capital."}]
    cases = [
        {"id": "a", "input": {"question": "q", "answer": "Wellington [1]", "sources": sources},
         "expected": {"label": "grounded"}},
        {"id": "b", "input": {"question": "q", "answer": f"x {MARK}", "sources": sources},
         "expected": {"label": "unsupported"}},
    ]  # fmt: skip
    (dataset / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    suite_path = root / "suites" / "cal.yaml"
    suite_path.parent.mkdir()
    suite_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "cal",
                "pack": "judging",
                "dataset": "../datasets/cal",
                "targets": [
                    {
                        "python": "lite_eval.judges.target:judge_target",
                        "configs": [{"name": "fake-nothink", **JUDGE}],
                    }
                ],
                "repeats": 1,
            }
        )
    )
    suite = load_suite(suite_path, root)
    target = JudgeTarget(client_factory=judge, **JUDGE)
    (result,) = asyncio.run(
        execute(suite, load_dataset(suite.dataset_path), [("fake-nothink", target)], RunOptions())
    )
    return result


def run(
    store: PgStore,
    tmp_path: Path,
    *,
    calibration: Path | None = None,
    min_agreement: float = 1.0,
    llm: FakeLLM | None = None,
    limit: int | None = None,
) -> RunResult:
    return judge_answers(
        "chars-300",
        SETTINGS,
        CORPUS,
        GOLDEN,
        store,
        root=tmp_path / "answers",
        calibration=calibration or calibrate(tmp_path / "judge").directory,
        min_agreement=min_agreement,
        limit=limit,
        embedder_for=lambda c: FakeEmbedder(dim=16),
        llm_for=lambda c: llm or FakeLLM(answer),
        judge_factory=fake_judge,
    )


def test_answers_are_judged_by_the_calibrated_judge(store: PgStore, tmp_path: Path) -> None:
    result = run(store, tmp_path, limit=8)
    stored = load_run(result.directory)
    assert result.record.summary.n_cases == 8 and result.record.status == "complete"
    (ref,) = result.record.scorers
    assert ref.calibrated is True
    assert all(w.code != "judge_uncalibrated" for w in result.record.warnings)
    flagged = {s.case_id for s in stored.scores if s.value == 0.0}
    coffee = {o.case_id for o in stored.outputs if MARK in o.value["answer"]}
    assert flagged == coffee == {"q01", "q02"}  # of the first eight, these name coffee
    out = stored.outputs[0]
    assert set(out.value) == {"question", "answer", "sources", "status", "dropped"}
    assert out.value["status"] == "answered" and out.value["sources"][0]["text"]
    assert out.trace.usage is not None and out.trace.latency_ms is not None
    config = result.record.target.config
    assert config["llm"]["model"] == "qwen3:8b" and "api_key_env" not in config["llm"]
    assert result.record.cost.judge_calls == 8
    rows = flagged_answers(result)
    assert [r["id"] for r in rows] == ["q01", "q02"]
    assert rows[0]["question"] == "What coffee to water ratio should I use for pour-over?"
    assert rows[0]["answer"].endswith(MARK + ".") and rows[0]["evidence"] == [MARK]


def test_the_judge_is_the_one_the_calibration_run_measured(store: PgStore, tmp_path: Path) -> None:
    result = run(store, tmp_path, limit=2)
    suite = yaml.safe_load((tmp_path / "answers" / "suites" / "answers-golden.yaml").read_text())
    judge = suite["judges"]["grounded"]
    assert judge == {k: v for k, v in JUDGE.items() if k != "rubric"}  # as the calibration had it
    (scorer,) = suite["scorers"]
    assert scorer["judge"]["calibration"]["min_agreement"] == 1.0
    assert result.record.scorers[0].calibrated is True


def test_a_calibration_below_the_threshold_refuses_before_answering(
    store: PgStore, tmp_path: Path
) -> None:
    def blind(spec: Any, timeout_s: float | None) -> FakeJudge:
        return FakeJudge(spec, by_marker("never there", "unsupported", "grounded"))

    weak = calibrate(tmp_path / "judge", blind)  # misses the corrupted case: agreement 0.5
    llm = FakeLLM(answer)
    with pytest.raises(LiteEvalConfigError, match=r"agreement 0\.5000 < 0\.9"):
        run(store, tmp_path, calibration=weak.directory, min_agreement=0.9, llm=llm, limit=2)
    assert llm.calls == []  # refused before any answer was generated


def test_the_command_refuses_a_calibration_that_is_not_a_run(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from lite_rag.cli import app

    config = tmp_path / "c.yaml"
    config.write_text("chunking: {unit: chars, size: 300, overlap: 30}\n")
    done = CliRunner().invoke(
        app,
        ["eval-answers", "-c", str(config), "--calibration", str(tmp_path / "nope"),
         "--min-agreement", "0.9", "--root", str(tmp_path / "answers")],
    )  # fmt: skip
    assert done.exit_code == 1, done.output  # lite-rag reports every error with exit 1
    assert "not a finished run directory" in done.output
