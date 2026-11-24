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
from lite_rag.config import Mode, Settings, StoreConfig, load_settings
from lite_rag.db import redact_conninfo
from lite_rag.engine import RagEngine, RagRuntime
from lite_rag.errors import LiteRagError
from lite_rag.extras import require_extra
from lite_rag.lineage import Related
from lite_rag.loaders import find_files
from lite_rag.maintenance import collect
from lite_rag.stores.base import CachedStore
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
def _store(config: StoreConfig, url: str | None, **kwargs: Any) -> Iterator[CachedStore]:
    """A store, closed afterwards (commands that work on a whole store, not one namespace)."""
    store = open_store(config, url, **kwargs)
    try:
        yield store
    finally:
        store.close()


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
EmbeddingOption = Annotated[
    str | None,
    typer.Option(
        "--embedding",
        help="A named embedder of `embeddings` (S22) for a new kb; default: `embedding`",
    ),
]
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
    embedding: EmbeddingOption = None,
) -> None:
    """Chunk, embed and index files into a knowledge base (unchanged files are skipped)."""
    files = find_files(paths)
    if not files:
        _fail("no supported files found")
    with _engine(config, namespace) as engine:
        if embedding is not None:  # create the kb with it (an existing kb must match it)
            engine.create_kb(kb, embedding=embedding)
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
    embedding: EmbeddingOption = None,
) -> None:
    """Rebuild a knowledge base under the current settings (another model or chunker) into a new
    one, from the documents' stored text. The old kb stays; switch clients to the new name."""
    with _engine(config, namespace) as engine:
        n = engine.reindex(kb, into, embedding=embedding)
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


lineage_app = typer.Typer(help="The provenance ledger (lineage design)", no_args_is_help=True)
app.add_typer(lineage_app, name="lineage")


@lineage_app.command("backfill")
def lineage_backfill(
    kb: KbArg, config: ConfigOption = None, namespace: NamespaceOption = None
) -> None:
    """Record documents stored before lineage was kept (or with it off)."""
    with _engine(config, namespace) as engine:
        recorded = engine.lineage_backfill(kb)
        stats = engine.stats(kb)
    total = stats.documents if stats else 0
    typer.echo(f"recorded {recorded} of {total} documents")


ArtifactArg = Annotated[str, typer.Argument(help="An artifact id (from `lineage show`)")]


def _related(items: list[Related]) -> None:
    for r in items:
        a = r.artifact
        where = r.via.metadata.get("chunk_id")
        at = f"  as {where}" if where else ""
        typer.echo(f"{a.kind} {a.key} v{a.version} {a.status}  {a.id}{at}")


@lineage_app.command("show")
def lineage_show(
    kb: KbArg,
    doc_id: Annotated[str, typer.Argument(help="Document id")],
    config: ConfigOption = None,
    namespace: NamespaceOption = None,
) -> None:
    """Every recorded version of a document, newest first, with its chunks."""
    with _engine(config, namespace) as engine:
        history = engine.lineage_document(kb, doc_id)
    if not history:
        _fail(f"no lineage recorded for {doc_id!r} in {kb!r}")
    for version, chunks in history:
        ended = f" (ended {version.ended_at:%Y-%m-%d %H:%M:%S})" if version.ended_at else ""
        typer.echo(f"v{version.version} {version.status}{ended}  {version.id}")
        for r in chunks:
            text = (r.artifact.payload or "(text cleared)").replace("\n", " ")[:60]
            typer.echo(f"  {r.via.metadata.get('chunk_id')}  {r.artifact.id}  {text}")


@lineage_app.command("upstream")
def lineage_upstream(
    artifact_id: ArtifactArg, config: ConfigOption = None, namespace: NamespaceOption = None
) -> None:
    """What an artifact was derived from."""
    with _engine(config, namespace) as engine:
        _related(engine.lineage_upstream(artifact_id))


@lineage_app.command("downstream")
def lineage_downstream(
    artifact_id: ArtifactArg, config: ConfigOption = None, namespace: NamespaceOption = None
) -> None:
    """Everything derived from an artifact (what a change to it affects)."""
    with _engine(config, namespace) as engine:
        _related(engine.lineage_downstream(artifact_id))


@lineage_app.command("gc")
def lineage_gc(config: ConfigOption = None) -> None:
    """Clear the text of versions ended more than `lineage.keep_text_days` ago and delete
    traces older than `trace.retention_days` — in every namespace of the store."""
    settings = _settings(config)
    lineage = settings.lineage
    with _errors(), _store(settings.store, settings.database_url, lineage=lineage.enabled) as store:
        cleared, deleted = collect(
            store,
            keep_text_days=lineage.keep_text_days if lineage.enabled else None,
            retention_days=settings.trace.retention_days,
        )
    typer.echo(f"cleared {cleared} texts, deleted {deleted} traces")


trace_app = typer.Typer(help="Query traces (lineage design §6.3)", no_args_is_help=True)
app.add_typer(trace_app, name="trace")


@trace_app.command("show")
def trace_show(
    trace_id: Annotated[str, typer.Argument(help="A trace id (Answer.trace_id)")],
    config: ConfigOption = None,
    namespace: NamespaceOption = None,
) -> None:
    """What each stage of a search or answer used, and whether it is still current."""
    with _engine(config, namespace) as engine:
        rep = engine.trace_report(trace_id)
    r = rep.record
    typer.echo(f"trace {r.id}  {r.operation}  {r.created_at:%Y-%m-%d %H:%M:%S} UTC")
    typer.echo(f"kbs: {', '.join(r.kbs)}  query: {r.query if r.query is not None else '-'}")
    if r.operation == "ask":
        typer.echo(f"answer: {r.status} by {r.model_id or '-'}")
    for s in rep.sources:
        src, version = s.source, s.artifact
        state: str = s.state
        if s.state == "stale" and version is not None and version.ended_at is not None:
            state = f"stale since {version.ended_at:%Y-%m-%d %H:%M:%S}"
        typer.echo(f"{src.stage:<10} {src.rank or '-':>3}  {s.kb}/{src.chunk_id}  {state}")
        if s.state == "stale" and version is not None:
            was = version.payload
            typer.echo(f"    was: {was}" if was is not None else "    was: (text cleared)")


@trace_app.command("purge")
def trace_purge(
    yes: Annotated[bool, typer.Option("--yes", help="Confirm: this deletes every trace")] = False,
    config: ConfigOption = None,
    namespace: NamespaceOption = None,
) -> None:
    """Delete every trace of the namespace (its kbs stay: `drop-kb` removes those)."""
    if not yes:
        _fail("pass --yes to delete every trace of the namespace")
    with _engine(config, namespace) as engine:
        deleted = engine.purge_traces()
    typer.echo(f"deleted {deleted} traces")


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
        tokens = f"saved {answer.usage} (cached)" if answer.cached else f"tokens {answer.usage}"
        typer.echo(f"— {answer.model_id}, {tokens}")
    if answer.cached:
        typer.echo("(cached answer: the same question over the same sources was answered before)")


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
    kb_per_dir: Annotated[
        bool,
        typer.Option(
            "--kb-per-dir",
            help="A kb per corpus subdirectory, embedded by the named embedder of that name "
            "(`embeddings.<dir>`; default otherwise), searched together (S22)",
        ),
    ] = False,
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
        with _store(apart, first.database_url) as store:
            results = evaluate(
                loaded,
                corpus,
                golden,
                store,
                root=root,
                schema=schema,
                keep=keep,
                crosscheck=crosscheck,
                kb_per_dir=kb_per_dir,
            )
    typer.echo(table(results))
    for r in results:  # status, errors and warnings of each run, not only its numbers
        typer.echo("\n" + render_run(r.record, str(r.directory)))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([report(r) for r in results], indent=2, ensure_ascii=False), "utf-8"
        )
        typer.echo(f"report: {out}")


@app.command("eval-answers")
def eval_answers_command(
    config: Annotated[Path, typer.Option("--config", "-c", help="The YAML config that answers")],
    calibration: Annotated[
        Path,
        typer.Option(help="The lite-eval run that calibrated the grounded judge (its directory)"),
    ],
    min_agreement: Annotated[
        float,
        typer.Option(help="Refuse the judge if its calibration agreed less often than this"),
    ],
    corpus: Annotated[Path, typer.Option(help="Corpus directory")] = Path("eval/corpus"),
    golden: Annotated[Path, typer.Option(help="Golden questions (JSONL)")] = Path(
        "eval/golden.jsonl"
    ),
    limit: Annotated[int | None, typer.Option(help="Only the first N questions")] = None,
    schema: Annotated[
        str, typer.Option(help="Schema for the evaluation kb, apart from application data")
    ] = "lite_rag_eval",
    root: Annotated[
        Path, typer.Option(help="lite-eval evaluation root: generated dataset, suite and runs")
    ] = Path("eval/lite-eval"),
) -> None:
    """Answer the golden questions with `ask` and let the calibrated grounded judge score each
    answer (design §8, S13): the flag rate on real answers, and every flagged answer to read."""
    with _errors():
        require_extra("lite_eval", "eval")
    from lite_eval.report.terminal import render_run

    from lite_rag.eval_grounded import (
        calibrated_judge,
        flagged_answers,
        judge_answers,
        missed_facts,
        report,
    )

    try:
        settings = load_settings(config, file_wins=True, store={"schema": schema})
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    with _errors():
        calibrated_judge(calibration)  # a bad calibration fails before the store is touched
        apart = evaluation_store(settings.store, schema)
        migrate_store(apart, settings.database_url)
        with _store(apart, settings.database_url) as store:
            result = judge_answers(
                config.stem,
                settings,
                corpus,
                golden,
                store,
                root=root,
                calibration=calibration,
                min_agreement=min_agreement,
                schema=schema,
                limit=limit,
            )
    typer.echo(render_run(result.record, str(result.directory), result.scores))
    typer.echo(report(flagged_answers(result), missed_facts(result)), nl=False)
