"""Local operator commands using the same storage and health services as HTTP."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import typer

from app.core.enums import JobKind, PipelineStage
from app.core.settings import get_settings
from app.db.session import create_session_factory
from app.models import ProcessingJob, SourceVideo, Transcript
from app.services.health import HealthService
from app.services.storage import StorageService
from app.transcription.benchmark import benchmark_transcription
from app.transcription.engine import WhisperEngine
from app.workers.tasks import run_pipeline_stage

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


def _queue_transcription(source_id: UUID, *, force: bool) -> UUID:
    with create_session_factory()() as session:
        if session.get(SourceVideo, source_id) is None:
            raise typer.BadParameter("source does not exist")
        if force:
            cached = session.query(Transcript).filter_by(source_video_id=source_id).one_or_none()
            if cached is not None:
                cached.input_fingerprint = ""
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.TRANSCRIPTION)
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id
    run_pipeline_stage.delay(str(source_id), PipelineStage.TRANSCRIPTION.value, str(job_id))
    return job_id


def _queue_reconstruction(source_id: UUID, *, force: bool) -> UUID:
    with create_session_factory()() as session:
        if session.get(SourceVideo, source_id) is None:
            raise typer.BadParameter("source does not exist")
        if force:
            cached = session.query(Transcript).filter_by(source_video_id=source_id).one_or_none()
            if cached is not None:
                cached.reconstruction_fingerprint = ""
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.RECONSTRUCTION)
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id
    run_pipeline_stage.delay(
        str(source_id), PipelineStage.CONTEXTUAL_RECONSTRUCTION.value, str(job_id), force
    )
    return job_id


@app.command()
def transcribe(source_id: UUID) -> None:
    """Queue local transcription, reusing a valid fingerprinted transcript."""
    typer.echo(str(_queue_transcription(source_id, force=False)))


@app.command()
def retranscribe(source_id: UUID, force: bool = True) -> None:
    """Queue transcription and, by default, bypass the transcript cache."""
    typer.echo(str(_queue_transcription(source_id, force=force)))


@app.command()
def reconstruct(source_id: UUID, force: bool = False) -> None:
    """Queue bounded contextual reconstruction, reusing its current fingerprint by default."""
    typer.echo(str(_queue_reconstruction(source_id, force=force)))


@app.command()
def transcript(source_id: UUID) -> None:
    """Print the current timestamped transcript as JSON."""
    with create_session_factory()() as session:
        current = session.query(Transcript).filter_by(source_video_id=source_id).one_or_none()
        if current is None:
            raise typer.BadParameter("transcript is not ready")
        typer.echo(
            json.dumps(
                {
                    "language": current.language,
                    "raw_text": current.raw_text,
                    "normalized_text": current.normalized_text,
                    "segments": current.segments,
                    "word_segments": current.word_segments,
                },
                ensure_ascii=False,
            )
        )


@app.command()
def benchmark(audio_path: Path) -> None:
    """Measure local configured faster-whisper throughput on representative audio."""
    if not audio_path.is_file():
        raise typer.BadParameter("audio_path must be a readable local file")
    settings = get_settings()
    report = benchmark_transcription(audio_path, WhisperEngine(), settings.transcription_options())
    typer.echo(json.dumps(report.as_dict()))


if __name__ == "__main__":
    app()
