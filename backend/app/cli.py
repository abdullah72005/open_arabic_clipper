"""Local operator commands using the same storage and health services as HTTP."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import typer

from app.core.settings import get_settings
from app.services.health import HealthService
from app.services.storage import StorageService

app = typer.Typer(no_args_is_help=True)


def _storage() -> StorageService:
    return StorageService(get_settings().storage_root)


@app.command()
def health() -> None:
    report = HealthService(_storage()).report()
    typer.echo(report.status.value)


@app.command()
def add(path: Path) -> None:
    if not path.is_file():
        raise typer.BadParameter("path must be a readable file")
    typer.echo(str(path.resolve()))


@app.command()
def status(source_id: UUID) -> None:
    typer.echo(str(source_id))


@app.command()
def retry(source_id: UUID) -> None:
    typer.echo(str(source_id))


@app.command()
def cleanup(older_than_seconds: int = 3600, limit: int = 100) -> None:
    removed = _storage().cleanup_temporary_files(
        older_than_seconds=older_than_seconds,
        limit=limit,
    )
    typer.echo(str(removed))
