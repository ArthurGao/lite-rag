import json
from typing import Any

import pytest
from typer.testing import CliRunner

from lite_rag.cli import app


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://rag:s3cret-pw@localhost:5432/lite_rag",
        "host=localhost port=5432 dbname=lite_rag user=rag password=s3cret-pw",
        "postgresql://rag:s3cret%40pw@localhost/lite_rag",  # '@' inside an encoded password
        "postgresql://s3cret:s3cret@localhost/lite_rag",  # password equal to the user name
    ],
)
def test_config_never_prints_the_database_password(url: str) -> None:
    result = CliRunner().invoke(app, ["config"], env={"LITERAG_DATABASE_URL": url})
    assert result.exit_code == 0, result.output
    shown = json.loads(result.output)["database_url"]
    assert "password=***" in shown and "localhost" in shown
    assert "password=s3cret" not in shown and "s3cret-pw" not in shown


def test_an_unparseable_url_is_replaced_not_echoed() -> None:
    result = CliRunner().invoke(app, ["config"], env={"LITERAG_DATABASE_URL": "pw=s3cret ??"})
    assert result.exit_code == 0, result.output
    assert "s3cret" not in result.output


def test_serve_refuses_a_public_address_without_keys() -> None:
    from typer.testing import CliRunner

    from lite_rag.cli import app

    result = CliRunner().invoke(app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 1 and "no API keys configured" in result.output


def test_key_create_prints_the_key_once_and_a_config_entry_with_its_hash() -> None:
    import json

    from typer.testing import CliRunner

    from lite_rag.cli import app
    from lite_rag.config import ApiKeyConfig
    from lite_rag.server.auth import key_hash

    out = (
        CliRunner()
        .invoke(
            app,
            ["key", "create", "--name", "steward", "--namespace", "steward/*", "--mode", "write"],
        )
        .output
    )
    key = out.split("): ", 1)[1].split()[0]
    entry = ApiKeyConfig(**json.loads(out.strip().splitlines()[-1]))
    assert entry.sha256 == key_hash(key) and key not in out.splitlines()[-1]
    assert (entry.namespaces, entry.kbs, entry.mode) == (["steward/*"], ["*"], "write")


def _answered(monkeypatch: pytest.MonkeyPatch, *, cached: bool) -> None:
    """`literag ask` over an engine stub: only the CLI's rendering is under test here."""
    from contextlib import contextmanager

    from lite_rag import cli
    from lite_rag.answer import Answer

    answer = Answer("q?", "answered", "Ratio 1:16 [1].", [], [], "ollama/qwen3:8b", cached=cached)

    class Stub:
        def ask(self, *args: Any, **kwargs: Any) -> Answer:
            return answer

    @contextmanager
    def engine(*args: Any) -> Any:
        yield Stub()

    monkeypatch.setattr(cli, "_engine", engine)


def test_ask_marks_a_cached_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _answered(monkeypatch, cached=True)
    out = CliRunner().invoke(app, ["ask", "kb", "q?"])
    assert out.exit_code == 0, out.output
    assert "(cached answer" in out.output
    assert "saved" in out.output  # its tokens were not spent again (final review S18)
    _answered(monkeypatch, cached=False)
    assert "cached" not in CliRunner().invoke(app, ["ask", "kb", "q?"]).output


def test_ask_json_carries_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    _answered(monkeypatch, cached=True)
    out = CliRunner().invoke(app, ["ask", "kb", "q?", "--json"])
    assert json.loads(out.output)["cached"] is True


def test_ingest_and_reindex_take_an_embedder_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """S22: `--embedding` creates the kb with that embedder (ingest) or builds the target with it
    (reindex); without it nothing changes."""
    from contextlib import contextmanager

    from lite_rag import cli
    from lite_rag.engine import Upserted

    calls: list[tuple[str, Any]] = []

    class Stub:
        def create_kb(self, kb: str, *, embedding: str | None = None) -> None:
            calls.append(("create_kb", (kb, embedding)))

        def ingest(self, kb: str, path: Any) -> Upserted:
            calls.append(("ingest", kb))
            return Upserted("a", 1, skipped=False)

        def reindex(self, kb: str, into: str, *, embedding: str | None = None) -> int:
            calls.append(("reindex", (kb, into, embedding)))
            return 3

    @contextmanager
    def engine(*args: Any) -> Any:
        yield Stub()

    monkeypatch.setattr(cli, "_engine", engine)
    doc = tmp_path / "a.md"
    doc.write_text("# A\n\ntext")
    assert CliRunner().invoke(app, ["ingest", "kb", str(doc), "--embedding", "e5"]).exit_code == 0
    assert CliRunner().invoke(app, ["ingest", "kb", str(doc)]).exit_code == 0
    out = CliRunner().invoke(app, ["reindex", "kb", "--into", "kb2", "--embedding", "e5"])
    assert out.exit_code == 0, out.output
    assert calls == [
        ("create_kb", ("kb", "e5")), ("ingest", "kb"), ("ingest", "kb"),
        ("reindex", ("kb", "kb2", "e5")),
    ]  # fmt: skip
