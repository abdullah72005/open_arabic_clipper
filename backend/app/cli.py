"""Local operator commands using the same storage and health services as HTTP."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import typer

from app.core.enums import CandidateDisposition, JobKind, PipelineStage, RefinementPriority
from app.core.settings import get_settings
from app.db.session import create_session_factory
from app.models import ClipCandidate, ProcessingJob, SourceVideo, Transcript
from app.refinement.handoff import build_stage4_handoff
from app.refinement.queue import (
    Stage35QueueError,
    list_refinements,
    queue_candidate_batch,
    queue_candidate_refinement,
    validate_candidate_for_refinement,
)
from app.runtime.heavy_model_lease import HeavyModelLeaseBusy, HeavyModelUnsafe
from app.runtime.memory import MemoryReadError, capture_memory
from app.services.health import HealthService
from app.services.storage import StorageCategory, StorageService
from app.transcription.benchmark import benchmark_transcription
from app.transcription.engine import WhisperEngine
from app.transcription.reconstruction.benchmark import (
    BenchmarkRunner,
    evaluate_completion_gate,
    load_benchmark_manifest,
    prompt_settings_fingerprint,
)
from app.transcription.reconstruction.capture import capture_hash, load_capture, save_capture
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import ProviderAvailability, ProviderHealth
from app.workers.tasks import run_pipeline_stage

app = typer.Typer(no_args_is_help=True)
_KNOWN_REGRESSION_MANIFEST_NAME = "stage-2-7/known-regression-v1.json"


def _storage() -> StorageService:
    return StorageService(get_settings().storage_root)


@app.command("diagnose-memory")
def diagnose_memory(
    json_output: bool = typer.Option(False, "--json/--text", help="Emit machine-readable JSON."),
) -> None:
    """Print labeled host/container memory facts. Read-only."""

    try:
        snapshot = capture_memory()
    except MemoryReadError as error:
        typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False))
        raise typer.Exit(code=1) from error
    if json_output:
        payload = asdict(snapshot)
        payload["effective_capacity"] = snapshot.effective_capacity
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    typer.echo(
        f"linux_total={snapshot.linux_total / 1024**3:.2f} GiB "
        f"linux_available={snapshot.linux_available / 1024**3:.2f} GiB"
    )
    typer.echo(
        f"swap_total={snapshot.swap_total / 1024**3:.2f} GiB "
        f"swap_free={snapshot.swap_free / 1024**3:.2f} GiB"
    )
    limit = (
        f"{snapshot.cgroup_limit / 1024**3:.2f} GiB"
        if snapshot.cgroup_limit is not None
        else "unlimited"
    )
    typer.echo(f"cgroup_limit={limit} cgroup_peak={snapshot.cgroup_peak}")
    typer.echo(
        f"process_rss={snapshot.process_rss / 1024**3:.3f} GiB "
        f"effective_capacity={snapshot.effective_capacity / 1024**3:.2f} GiB"
    )


@app.command()
def health() -> None:
    report = HealthService(_storage()).report()
    typer.echo(report.status.value)


@app.command("reconstruction-health")
def reconstruction_health() -> None:
    """Verify configured reconstruction endpoint and exact model availability."""

    settings = get_settings()
    provider = settings.reconstruction_provider_instance()
    report = (
        provider.health()
        if provider is not None
        else ProviderHealth(
            ProviderAvailability.MISCONFIGURED,
            settings.reconstruction_provider,
            settings.reconstruction_provider_model,
            None,
            "local Qwen reconstruction is disabled by default; set "
            "CLIPFACTORY_LOCAL_QWEN_ENABLED=true to enable local providers",
        )
    )
    typer.echo(
        json.dumps(
            {
                "availability": report.availability.value,
                "provider": report.provider,
                "model": report.model,
                "digest": report.model_digest,
                "detail": report.detail,
            }
        )
    )
    if report.availability is not ProviderAvailability.AVAILABLE:
        raise typer.Exit(1)


@app.command("recover-heavy-model")
def recover_heavy_model() -> None:
    """Clear unsafe heavy-model state only after the model is no longer resident."""

    settings = get_settings()
    factory = settings.heavy_model_lease_factory()
    if not factory.unsafe_recorded():
        typer.echo(json.dumps({"status": "CLEAR", "detail": "no unsafe heavy-model state"}))
        return
    provider = settings.reconstruction_provider_instance()
    resident = provider.is_model_resident() if provider is not None else True
    if resident:
        typer.echo(
            json.dumps(
                {
                    "status": "UNSAFE",
                    "detail": factory.unsafe_reason(),
                    "model": settings.reconstruction_provider_model,
                }
            )
        )
        raise typer.Exit(1)
    factory.recover()
    typer.echo(json.dumps({"status": "RECOVERED", "detail": "model confirmed not resident"}))


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
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.TRANSCRIPTION)
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id
    run_pipeline_stage.delay(str(source_id), PipelineStage.TRANSCRIPTION.value, str(job_id), force)
    return job_id


def _queue_reconstruction(source_id: UUID, *, force: bool) -> UUID:
    with create_session_factory()() as session:
        if session.get(SourceVideo, source_id) is None:
            raise typer.BadParameter("source does not exist")
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
def retranscribe(source_id: UUID, force: bool = typer.Option(True, "--force/--no-force")) -> None:
    """Queue transcription and, by default, bypass the transcript cache."""
    typer.echo(str(_queue_transcription(source_id, force=force)))


@app.command()
def reconstruct(source_id: UUID, force: bool = typer.Option(False, "--force/--no-force")) -> None:
    """Queue bounded contextual reconstruction, reusing its current fingerprint by default."""
    typer.echo(str(_queue_reconstruction(source_id, force=force)))


def _queue_candidate_analysis(source_id: UUID, *, force: bool) -> UUID:
    with create_session_factory()() as session:
        if session.get(SourceVideo, source_id) is None:
            raise typer.BadParameter("source does not exist")
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.CANDIDATE_ANALYSIS)
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id
    run_pipeline_stage.delay(
        str(source_id), PipelineStage.CANDIDATE_ANALYSIS.value, str(job_id), force
    )
    return job_id


@app.command("candidate-analysis")
def candidate_analysis(
    source_id: UUID, force: bool = typer.Option(False, "--force/--no-force")
) -> None:
    """Queue bounded Stage 3 candidate analysis for a source."""

    typer.echo(str(_queue_candidate_analysis(source_id, force=force)))


@app.command("candidates")
def candidates(
    source_id: UUID,
    limit: int = typer.Option(20, min=1, max=200),
    include_rejected: bool = typer.Option(False, "--include-rejected/--accepted-only"),
) -> None:
    """Print bounded current candidate summaries for local inspection."""

    with create_session_factory()() as session:
        statement = (
            session.query(ClipCandidate)
            .filter(ClipCandidate.source_video_id == source_id)
            .filter(ClipCandidate.is_current.is_(True))
            .order_by(ClipCandidate.clip_score.desc())
        )
        if not include_rejected:
            statement = statement.filter(
                ClipCandidate.disposition.in_(
                    [
                        CandidateDisposition.CANDIDATE,
                        CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
                    ]
                )
            )
        rows = statement.limit(limit).all()
        typer.echo(
            json.dumps(
                [
                    {
                        "id": str(row.id),
                        "candidate_key": row.candidate_key,
                        "disposition": row.disposition.value,
                        "start_time": row.start_time,
                        "end_time": row.end_time,
                        "segment_indexes": [
                            row.start_segment_index,
                            row.end_segment_index,
                        ],
                        "clip_score": row.clip_score,
                        "transcript_confidence": row.transcript_confidence,
                        "primary_content_type": row.primary_content_type.value,
                        "refinement_reasons": row.refinement_reasons,
                        "rights_risk": row.rights_risk.value,
                        "originality_risk": row.originality_risk.value,
                        "excerpt": row.transcript_excerpt[:200],
                    }
                    for row in rows
                ],
                ensure_ascii=False,
            )
        )


@app.command("candidate-refine")
def candidate_refine(
    candidate_id: UUID,
    priority: str = typer.Option("CANDIDATE", "--priority"),
    force: bool = typer.Option(False, "--force/--no-force"),
) -> None:
    """Queue one explicit candidate-scoped Stage 3.5 refinement."""

    try:
        parsed = RefinementPriority(priority.upper())
    except ValueError as error:
        raise typer.BadParameter("priority must be CANDIDATE or FINAL_CLIP") from error
    with create_session_factory()() as session:
        try:
            candidate = validate_candidate_for_refinement(session, candidate_id)
        except Stage35QueueError as error:
            raise typer.BadParameter(str(error)) from error
        outcome = queue_candidate_refinement(session, _storage(), candidate, parsed, force=force)
    typer.echo(
        json.dumps(
            {
                "refinement_id": str(outcome.refinement_id),
                "job_id": str(outcome.job_id) if outcome.job_id else None,
                "status": outcome.status,
                "queued": outcome.queued,
                "cached": outcome.cached,
                "active": outcome.active,
            }
        )
    )


@app.command("candidate-refine-batch")
def candidate_refine_batch(
    source_id: UUID,
    limit: int = typer.Option(5, min=1, max=10),
    force: bool = typer.Option(False, "--force/--no-force"),
) -> None:
    """Queue a bounded, score-ordered candidate-grade refinement batch."""

    with create_session_factory()() as session:
        try:
            outcomes = queue_candidate_batch(
                session, _storage(), source_id, limit=limit, force=force
            )
        except Stage35QueueError as error:
            raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            [
                {
                    "refinement_id": str(outcome.refinement_id),
                    "job_id": str(outcome.job_id) if outcome.job_id else None,
                    "queued": outcome.queued,
                    "cached": outcome.cached,
                    "active": outcome.active,
                }
                for outcome in outcomes
            ]
        )
    )


@app.command("candidate-refinements")
def candidate_refinements(candidate_id: UUID) -> None:
    """Print both quality-level refinements for a candidate."""

    with create_session_factory()() as session:
        rows = list_refinements(session, candidate_id)
        typer.echo(
            json.dumps(
                [
                    {
                        "id": str(row.id),
                        "priority": row.priority.value,
                        "status": row.status.value,
                        "quality_level": row.quality_level,
                        "confidence": row.confidence,
                        "refined_start": row.refined_start,
                        "refined_end": row.refined_end,
                        "final_transcript": row.final_transcript[:500],
                        "cache_eligible": row.cache_eligible,
                    }
                    for row in rows
                ],
                ensure_ascii=False,
            )
        )


@app.command("candidate-handoff")
def candidate_handoff(candidate_id: UUID) -> None:
    """Print the typed read-only Stage 4 handoff for a candidate."""

    with create_session_factory()() as session:
        handoff = build_stage4_handoff(session, candidate_id)
        if handoff is None:
            raise typer.BadParameter("candidate does not exist")
    typer.echo(json.dumps(handoff, ensure_ascii=False, default=str))


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
                    "reconstruction_status": current.reconstruction_status.value,
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


@app.command("benchmark-reconstruction")
def benchmark_reconstruction(
    manifest_name: str,
    model: str | None = typer.Option(None, "--model"),
    allow_known_regression_set: bool = typer.Option(False, "--allow-known-regression-set"),
    capture_asr: bool = typer.Option(False, "--capture-asr", help="Capture immutable ASR only."),
    from_capture: str | None = typer.Option(
        None, "--from-capture", help="Replay a stored capture without a transcriber."
    ),
) -> None:
    """Run a private, authorized reconstruction benchmark through production stages."""

    settings = get_settings()
    storage = _storage()
    if allow_known_regression_set and manifest_name != _KNOWN_REGRESSION_MANIFEST_NAME:
        raise typer.BadParameter(
            "--allow-known-regression-set is reserved for the Chernobyl diagnostic manifest"
        )
    manifest_path = storage.resolve(StorageCategory.BENCHMARKS, manifest_name)
    manifest = load_benchmark_manifest(
        manifest_path,
        allow_known_regression_set=allow_known_regression_set,
        known_regression_manifest_path=storage.resolve(
            StorageCategory.BENCHMARKS, _KNOWN_REGRESSION_MANIFEST_NAME
        ),
    )
    provider = settings.reconstruction_provider_instance(model=model)
    health = provider.health() if provider is not None else None
    whisper_options = asdict(settings.transcription_options())
    fingerprint = prompt_settings_fingerprint(
        provider=settings.reconstruction_provider,
        model=health.model if health is not None else None,
        digest=health.model_digest if health is not None else None,
        whisper_options=whisper_options,
    )
    runner = BenchmarkRunner(
        storage=storage,
        whisper_engine=WhisperEngine() if from_capture is None else None,
        corrector=settings.contextual_corrector(),
        reconstructor=ContextualReconstructor(provider),
        transcription_options=settings.transcription_options(),
        provider_health=health,
        prompt_settings_fingerprint=fingerprint,
    )
    lease_factory = settings.heavy_model_lease_factory()
    if capture_asr:
        try:
            with lease_factory.acquire(purpose="benchmark-asr") as _heavy_lease:
                capture = runner.capture_asr(manifest)
        except HeavyModelLeaseBusy as error:
            typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False))
            raise typer.Exit(code=1) from error
        except HeavyModelUnsafe as error:
            typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False))
            raise typer.Exit(code=1) from error
        path = save_capture(storage, capture, name=capture.capture_id)
        typer.echo(
            json.dumps(
                {
                    "capture_id": capture.capture_id,
                    "capture_hash": capture_hash(capture),
                    "path": str(path),
                    "status": "CAPTURED",
                },
                ensure_ascii=False,
            )
        )
        return
    try:
        with lease_factory.acquire(purpose="benchmark") as _heavy_lease:
            report = runner.run(
                manifest, capture=load_capture(storage, from_capture) if from_capture else None
            )
    except HeavyModelLeaseBusy as error:
        typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False))
        raise typer.Exit(code=1) from error
    except HeavyModelUnsafe as error:
        typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False))
        raise typer.Exit(code=1) from error
    passed, reasons = evaluate_completion_gate(
        report, expected_prompt_settings_fingerprint=fingerprint
    )
    typer.echo(
        json.dumps(
            {
                "model": report.model_identifier,
                "model_digest": report.model_digest,
                "provider_available": report.provider_available,
                "human_labels_complete": report.human_labels_complete,
                "model_feasible": report.model_feasible,
                "semantic_correct_stage25": report.semantic_correct_stage25,
                "semantic_correct_stage27": report.semantic_correct_stage27,
                "improved": report.improved,
                "unchanged_correct": report.unchanged_correct,
                "unchanged_wrong": report.unchanged_wrong,
                "regressed": report.regressed,
                "hallucinated": report.hallucinated,
                "changed_wrong": report.changed_wrong,
                "unresolved": report.unresolved,
                "unreviewed": report.unreviewed,
                "exact_improved": report.exact_improved,
                "exact_unchanged_correct": report.exact_unchanged_correct,
                "exact_unchanged_wrong": report.exact_unchanged_wrong,
                "exact_regressed": report.exact_regressed,
                "exact_changed_wrong": report.exact_changed_wrong,
                "reconstruction_wall_seconds": report.reconstruction_wall_seconds,
                "average_serialized_prompt_bytes": report.average_serialized_prompt_bytes,
                "max_serialized_prompt_bytes": report.max_serialized_prompt_bytes,
                "average_estimated_input_tokens": report.average_estimated_input_tokens,
                "max_estimated_input_tokens": report.max_estimated_input_tokens,
                "unload_confirmed": report.unload_confirmed,
                "unload_warning": report.unload_warning,
                "source_audio_seconds": report.source_audio_seconds,
                "wall_clock_seconds": report.wall_clock_seconds,
                "peak_ram_bytes": report.peak_ram_bytes,
                "peak_vram_bytes": report.peak_vram_bytes,
                "comparison_path": str(report.comparison_path) if report.comparison_path else None,
                "report_path": str(report.report_path) if report.report_path else None,
                "worksheet_path": str(report.worksheet_path) if report.worksheet_path else None,
                "passed": passed,
                "reasons": reasons,
                "status": "READY FOR STAGE 3" if passed else "STAGE 2.7 MUST CONTINUE",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    app()
