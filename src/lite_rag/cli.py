"""`literag` command line. Commands are added as their stage lands (design §12)."""

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn

import typer

from lite_rag import __version__
from lite_rag.chunking import chunk_context
from lite_rag.config import Mode, Settings, load_settings
from lite_rag.db import redact_conninfo
from lite_rag.engine import RagEngine, RagRuntime
from lite_rag.errors import LiteRagError
from lite_rag.extras import require_extra
from lite_rag.loaders import find_files
from lite_rag.stores.factory import evaluation_store, location, migrate_store, open_store

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
    except Exception as exc:
        if _is_lite_eval_error(exc):  # lite-eval runs `literag eval`
            _fail(str(exc))
        raise


def _is_lite_eval_error(exc: Exception) -> bool:
    """Without importing lite-eval (the optional `eval` extra): one that was never imported
    cannot have raised."""
    errors = sys.modules.get("lite_eval.errors")
    return errors is not None and isinstance(exc, errors.LiteEvalError)


def _echo_json(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


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
    _echo_json(data)


@app.command("migrate")
def migrate_command(config: ConfigOption = None) -> None:
    """Create or upgrade the lite-rag schema: in PostgreSQL, or the data directory (chroma)."""
    settings = _settings(config)
    store = settings.store
    with _errors():
        applied = migrate_store(store, settings.database_url)
    target = location(store, settings.database_url)
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
    _echo_json(asdict(result))


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


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Address to bind (default: server.host)")] = None,
    port: Annotated[int | None, typer.Option(help="Port (default: server.port, 8700)")] = None,
    config: ConfigOption = None,
) -> None:
    """Serve the HTTP API (design §7; extra `server`). Without API keys only a loopback address
    may be bound: an open server must not be reachable from other machines."""
    from lite_rag.server import check_bind, run

    settings = _settings(config)
    host = host or settings.server.host
    with _errors():
        check_bind(settings, host)  # before connecting to anything
    with _errors():
        # one store connection per request the server runs at once: none waits on the pool
        runtime = RagRuntime.from_settings(
            settings, max_connections=max(4, settings.server.max_workers)
        )
        try:
            run(settings, runtime, host, port or settings.server.port)
        finally:
            runtime.close()


key_app = typer.Typer(help="API keys for the HTTP service", no_args_is_help=True)
app.add_typer(key_app, name="key")


@key_app.command("create")
def key_create(
    name: Annotated[str, typer.Option(help="Who uses it, e.g. llm-steward")],
    namespace: Annotated[
        list[str], typer.Option("--namespace", help="Namespace glob it may use; repeatable")
    ],
    kb: Annotated[list[str] | None, typer.Option("--kb", help="kb glob; repeatable")] = None,
    mode: Annotated[Literal["read", "write"], typer.Option(help="read or write")] = "read",
) -> None:
    """Print a new API key (shown once) and the entry to add to `server.keys`. Only its sha256
    goes into the config; the key itself is not stored anywhere."""
    from lite_rag.config import ApiKeyConfig
    from lite_rag.server.auth import new_key

    key, digest = new_key()
    scope: dict[str, Any] = {"namespaces": namespace, "mode": mode} | ({"kbs": kb} if kb else {})
    entry = ApiKeyConfig(name=name, sha256=digest, **scope)
    typer.echo(f"key (shown once, give it to {name}): {key}")
    typer.echo("add to server.keys (YAML) or LITERAG_SERVER__KEYS (a JSON list):")
    typer.echo(json.dumps(entry.model_dump(), ensure_ascii=False))


@app.command()
def reindex(
    kb: KbArg,
    into: Annotated[str, typer.Option("--into", help="The new knowledge base to build")],
    config: ConfigOption = None,
    namespace: NamespaceOption = None,
) -> None:
    """Rebuild a knowledge base under the current settings (another model or chunker) into a new
    one, from the documents' stored text. The old kb stays; switch clients to the new name."""
    with _engine(config, namespace) as engine:
        n = engine.reindex(kb, into)
    typer.echo(f"{n} documents reindexed into {into}")


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
        _echo_json([asdict(h) for h in hits])
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
        _echo_json(asdict(answer))
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


@app.command("eval")
def eval_command(
    configs: Annotated[
        list[Path],
        typer.Option("--config", "-c", help="A YAML config to compare; repeat for several"),
    ],
    corpus: Annotated[Path, typer.Option(help="Corpus directory")] = Path("eval/corpus"),
    golden: Annotated[Path, typer.Option(help="Golden questions (JSONL)")] = Path(
        "eval/golden.jsonl"
    ),
    schema: Annotated[
        str,
        typer.Option(
            help="Schema (or, chroma, subdirectory of store.data_dir) for the evaluation kbs,"
            " apart from application data"
        ),
    ] = "lite_rag_eval",
    out: Annotated[Path | None, typer.Option(help="Also write the full report as JSON")] = None,
    keep: Annotated[bool, typer.Option(help="Keep the evaluation kbs afterwards")] = False,
    root: Annotated[
        Path, typer.Option(help="lite-eval evaluation root: generated dataset, suite and runs")
    ] = Path("eval/lite-eval"),
    crosscheck: Annotated[
        bool,
        typer.Option(
            help="Recompute the ranking metrics with ranx and warn on any gap (needs ranx: "
            "`uv run --with ranx==0.3.21 ...`)"
        ),
    ] = False,
) -> None:
    """Build a kb per index config from the corpus, run the golden questions, compare.
    Each config is a lite-eval run (written under eval/lite-eval/runs/). Config files outrank
    LITERAG_* variables and `.env` here."""
    with _errors():
        require_extra("lite_eval", "eval")
    from lite_eval.report.terminal import render_run

    from lite_rag.eval import evaluate, report, table

    if not configs:
        raise typer.BadParameter("give at least one --config")
    try:
        loaded = [
            (p.stem, load_settings(p, file_wins=True, store={"schema": schema})) for p in configs
        ]
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if len({location(s.store, s.database_url) for _, s in loaded}) > 1:
        _fail("all configs must use one database: they are compared on one store")
    with _errors():
        first = loaded[0][1]
        apart = evaluation_store(first.store, schema)
        migrate_store(apart, first.database_url)
        store = open_store(apart, first.database_url)
        try:
            results = evaluate(
                loaded,
                corpus,
                golden,
                store,
                root=root,
                schema=schema,
                keep=keep,
                crosscheck=crosscheck,
            )
        finally:
            store.close()
    typer.echo(table(results))
    for r in results:  # status, errors and warnings of each run, not only its numbers
        typer.echo("\n" + render_run(r.record, str(r.directory)))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([report(r) for r in results], indent=2, ensure_ascii=False), "utf-8"
        )
        typer.echo(f"report: {out}")
