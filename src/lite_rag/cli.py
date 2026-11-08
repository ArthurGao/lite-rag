"""`literag` command line. Commands are added as their stage lands (design §12)."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from lite_rag import __version__
from lite_rag.chunking import chunk_context
from lite_rag.config import Mode, Settings, load_settings
from lite_rag.db import check_vector, connect, migrate, redact_conninfo
from lite_rag.engine import RagEngine
from lite_rag.errors import LiteRagError
from lite_rag.loaders import find_files

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="YAML config file (env vars still override it)"),
]


def _settings(config: Path | None) -> Settings:
    try:
        return load_settings(config)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


@contextmanager
def _errors() -> Iterator[None]:
    """A lite-rag error ends the command with its message and exit code 1, not a traceback."""
    try:
        yield
    except LiteRagError as exc:
        _fail(str(exc))


def _masked(url: str | None) -> str | None:
    return redact_conninfo(url) if url else url


@app.command()
def version() -> None:
    """Print the lite-rag version."""
    typer.echo(__version__)


@app.command("config")
def show_config(config: ConfigOption = None) -> None:
    """Print the effective configuration (the database password is masked)."""
    settings = _settings(config)
    data = settings.model_dump(mode="json", by_alias=True)
    data["database_url"] = _masked(settings.database_url)
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))


@app.command("migrate")
def migrate_command(config: ConfigOption = None) -> None:
    """Create or upgrade the lite-rag schema in PostgreSQL."""
    settings = _settings(config)
    schema = settings.store.schema_name
    with _errors(), connect(settings.database_url) as conn:
        applied = migrate(conn, schema)
        check_vector(conn)
    target = f"{_masked(settings.database_url)} schema {schema}"
    typer.echo(f"applied {', '.join(applied)} to {target}" if applied else f"{target}: up to date")


@contextmanager
def _engine(config: Path | None, namespace: str | None) -> Iterator[RagEngine]:
    settings = _settings(config)
    with _errors():
        engine = RagEngine.from_settings(settings, namespace)
        try:
            yield engine
        finally:
            engine.close()


KbArg = Annotated[str, typer.Argument(help="Knowledge base name")]
RerankOption = Annotated[
    bool | None,
    typer.Option("--rerank/--no-rerank", help="Re-score candidates with the cross-encoder"),
]
NamespaceOption = Annotated[
    str | None,
    typer.Option(
        "--namespace", "-n", help="Namespace, e.g. myproject/tenant_42 (or LITERAG_NAMESPACE)"
    ),
]


@app.command()
def ingest(
    kb: KbArg,
    paths: Annotated[list[Path], typer.Argument(help="Files or directories (.md, .txt)")],
    config: ConfigOption = None,
    namespace: NamespaceOption = None,
) -> None:
    """Chunk, embed and index files into a knowledge base (unchanged files are skipped)."""
    files = find_files(paths)
    if not files:
        _fail("no supported files found")
    with _engine(config, namespace) as engine:
        for path in files:
            result = engine.ingest(kb, path)
            state = "unchanged" if result.skipped else f"{result.chunks} chunks"
            typer.echo(f"{path}: {state} ({result.doc_id})")


@app.command()
def stats(kb: KbArg, config: ConfigOption = None, namespace: NamespaceOption = None) -> None:
    """Show a knowledge base's size and how it was built."""
    with _engine(config, namespace) as engine:
        result = engine.stats(kb)
    if result is None:
        _fail(f"knowledge base {kb!r} does not exist")
    typer.echo(json.dumps(asdict(result), indent=2, ensure_ascii=False))


@app.command()
def delete(
    kb: KbArg,
    doc_id: Annotated[str, typer.Argument(help="Document id")],
    config: ConfigOption = None,
    namespace: NamespaceOption = None,
) -> None:
    """Delete one document from a knowledge base."""
    with _engine(config, namespace) as engine:
        found = engine.delete(kb, doc_id)
    typer.echo("deleted" if found else "not found")


@app.command("drop-kb")
def drop_kb(
    kb: KbArg,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm: this deletes every document")
    ] = False,
    config: ConfigOption = None,
    namespace: NamespaceOption = None,
) -> None:
    """Delete a knowledge base and everything in it."""
    if not yes:
        _fail("pass --yes to delete the knowledge base and all its documents")
    with _engine(config, namespace) as engine:
        found = engine.delete_kb(kb)
    typer.echo("dropped" if found else "not found")


@app.command()
def search(
    kb: KbArg,
    query: Annotated[str, typer.Argument(help="The question or keywords")],
    top_k: Annotated[int | None, typer.Option("--top-k", "-k", help="How many hits")] = None,
    mode: Annotated[Mode | None, typer.Option("--mode", "-m")] = None,
    rerank: RerankOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the hits as JSON")] = False,
    config: ConfigOption = None,
    namespace: NamespaceOption = None,
) -> None:
    """Search a knowledge base; shows each stage's score and rank for every hit."""
    with _engine(config, namespace) as engine:
        hits = engine.search(kb, query, top_k=top_k, mode=mode, rerank=rerank)
    if as_json:
        typer.echo(json.dumps([asdict(h) for h in hits], indent=2, ensure_ascii=False))
        return
    if not hits:
        typer.echo("no hits")
    for i, h in enumerate(hits, start=1):
        stages = "  ".join(
            f"{stage} #{h.ranks[stage]} {h.scores[stage]:.4f}" for stage in sorted(h.ranks)
        )
        where = chunk_context(h.title, h.heading_path)
        typer.echo(f"{i}. {h.score:.4f}  [{stages}]  {where}  ({h.chunk_id})")
        typer.echo("   " + " ".join(h.text.split())[:200])


@app.command()
def ask(
    kb: KbArg,
    question: Annotated[str, typer.Argument(help="The question")],
    top_k: Annotated[int | None, typer.Option("--top-k", "-k", help="Sources to give")] = None,
    mode: Annotated[Mode | None, typer.Option("--mode", "-m")] = None,
    rerank: RerankOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the answer as JSON")] = False,
    config: ConfigOption = None,
    namespace: NamespaceOption = None,
) -> None:
    """Answer a question from a knowledge base, citing the sources it used."""
    with _engine(config, namespace) as engine:
        answer = engine.ask(kb, question, top_k=top_k, mode=mode, rerank=rerank)
    if as_json:
        typer.echo(json.dumps(asdict(answer), indent=2, ensure_ascii=False))
        return
    if answer.status == "no_sources":
        typer.echo("No relevant material was found in the knowledge base.")
        return
    typer.echo(answer.text)
    if answer.citations:
        typer.echo("")
    for c in answer.citations:
        typer.echo(f"[{c.n}] {c.title}  ({c.chunk_id})")
    if answer.dropped:
        typer.echo(f"(removed citations to sources that were not given: {answer.dropped})")
    if answer.model_id:
        typer.echo(f"— {answer.model_id}, tokens {answer.usage}")
