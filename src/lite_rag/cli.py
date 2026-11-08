"""`literag` command line (design §11). Commands are added as their stage lands (design §12)."""

import json
import re
from pathlib import Path
from typing import Annotated

import typer

from lite_rag import __version__
from lite_rag.config import Settings, load_settings
from lite_rag.db import check_vector, connect, migrate
from lite_rag.errors import LiteRagError

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


def _masked(url: str | None) -> str | None:
    return re.sub(r"//([^:/@]+):[^@]+@", r"//\1:***@", url) if url else url


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
    try:
        with connect(settings.database_url) as conn:
            applied = migrate(conn, schema)
            check_vector(conn)
    except LiteRagError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    target = f"{_masked(settings.database_url)} schema {schema}"
    typer.echo(f"applied {', '.join(applied)} to {target}" if applied else f"{target}: up to date")
