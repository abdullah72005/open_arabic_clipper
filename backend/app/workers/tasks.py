"""Durable Celery wrappers for pipeline work."""

from datetime import datetime, timezone
from typing import Final
from uuid import UUID

from celery import Task  # type: ignore[import-untyped]
from sqlalchemy.orm import Session

from app.candidates.executor import CandidateAnalysisExecutor
from app.core.enums import JobKind, JobStatus, PipelineStage
from app.core.settings import get_settings
from app.db.session import create_session_factory
from app.media.audio import AudioExtractor
from app.media.ffprobe import FFprobe
from app.models import ProcessingJob
from app.pipeline.executor import StageCancelled, StageExecutor
from app.pipeline.runner import PipelineRunner
from app.pipeline.stages import (
    AudioAnalysisExecutor,
    AudioExtractionExecutor,
    ContextualReconstructionExecutor,
    IngestExecutor,
    ProbeExecutor,
    TranscriptionExecutor,
    TranscriptNormalizationExecutor,
)
from app.refinement.executor import build_candidate_refinement_executor
from app.services.storage import StorageService
from app.transcription.engine import WhisperEngine
from app.workers.celery_app import celery_app

_executors: dict[PipelineStage, StageExecutor] = {}
_last_heartbeat: datetime | None = None
MAX_RETRIES: Final = 3
_NEXT_STAGE: Final = {
    PipelineStage.INGEST: PipelineStage.PROBE,
    PipelineStage.PROBE: PipelineStage.AUDIO_EXTRACTION,
    PipelineStage.AUDIO_EXTRACTION: PipelineStage.TRANSCRIPTION,
    PipelineStage.TRANSCRIPTION: PipelineStage.TRANSCRIPT_NORMALIZATION,
    PipelineStage.TRANSCRIPT_NORMALIZATION: PipelineStage.CONTEXTUAL_RECONSTRUCTION,
    PipelineStage.CONTEXTUAL_RECONSTRUCTION: PipelineStage.AUDIO_ANALYSIS,
    PipelineStage.AUDIO_ANALYSIS: PipelineStage.CANDIDATE_ANALYSIS,
}


def register_stage_executor(stage: PipelineStage, executor: StageExecutor) -> None:
    """Register concrete work without coupling orchestration to media services."""
    _executors[stage] = executor


def _stage_executors(session: Session) -> dict[PipelineStage, StageExecutor]:
    """Build worker-local Stage 2 executors while retaining test/Stage 1 registrations."""
    settings = get_settings()
    storage = StorageService(settings.storage_root)
    lease_factory = settings.heavy_model_lease_factory()
    defaults: dict[PipelineStage, StageExecutor] = {
        PipelineStage.INGEST: IngestExecutor(),
        PipelineStage.PROBE: ProbeExecutor(FFprobe(binary=settings.ffprobe_binary)),
        PipelineStage.AUDIO_EXTRACTION: AudioExtractionExecutor(
            AudioExtractor(session=session, storage=storage, ffmpeg_binary=settings.ffmpeg_binary)
        ),
        PipelineStage.TRANSCRIPTION: TranscriptionExecutor(
            session=session,
            engine=WhisperEngine(),
            options=settings.transcription_options(),
            storage=storage,
            lease_factory=lease_factory,
        ),
        PipelineStage.TRANSCRIPT_NORMALIZATION: TranscriptNormalizationExecutor(
            session=session, corrector=settings.contextual_corrector()
        ),
        PipelineStage.CONTEXTUAL_RECONSTRUCTION: ContextualReconstructionExecutor(
            session=session,
            reconstructor=settings.contextual_reconstructor(),
            lease_factory=lease_factory,
        ),
        PipelineStage.AUDIO_ANALYSIS: AudioAnalysisExecutor(
            session=session, storage=storage, ffmpeg_binary=settings.ffmpeg_binary
        ),
        PipelineStage.CANDIDATE_ANALYSIS: CandidateAnalysisExecutor(
            session=session,
            config=settings.stage3_config(),
            provider=settings.candidate_semantic_provider(),
            mode=settings.semantic_provider_mode(),
            lease_factory=lease_factory,
            admission=settings.gemini_admission_controller(),
        ),
    }
    return {**defaults, **_executors}


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True, autoretry_for=(), name="clipfactory.run_pipeline_stage"
)
def run_pipeline_stage(
    self: Task, source_id: str, stage: str, job_id: str | None = None, force: bool = False
) -> dict[str, str | bool | None]:
    """Run one durable stage; retry only exceptions explicitly marked retryable."""
    parsed_stage = PipelineStage(stage)
    parsed_job_id = UUID(job_id) if job_id else None
    session = create_session_factory()()
    try:
        if parsed_job_id is None and parsed_stage is PipelineStage.INGEST:
            job = ProcessingJob(source_video_id=UUID(source_id), kind=JobKind.INGEST)
            session.add(job)
            session.commit()
            parsed_job_id = job.id
        if parsed_job_id is not None:
            pending_job = session.get(ProcessingJob, parsed_job_id)
            if pending_job is not None and pending_job.status is JobStatus.CANCELLED:
                # A job cancelled while queued must never run nor schedule the
                # next stage.
                return {"job_id": str(parsed_job_id), "skipped": False, "cancelled": True}
        runner = PipelineRunner(session, _stage_executors(session))
        try:
            result = runner.run(UUID(source_id), parsed_stage, job_id=parsed_job_id, force=force)
        except Exception as error:
            if getattr(error, "retryable", False):
                if parsed_job_id is not None:
                    retry_job = session.get(ProcessingJob, parsed_job_id)
                    if retry_job is not None:
                        retry_job.retry_count += 1
                        session.commit()
                raise self.retry(
                    args=[source_id, stage, str(parsed_job_id) if parsed_job_id else None, force],
                    exc=error,
                    max_retries=MAX_RETRIES,
                ) from error
            raise
    finally:
        session.close()
    if next_stage := _NEXT_STAGE.get(parsed_stage):
        run_pipeline_stage.delay(source_id, next_stage.value)
    return {
        "run_id": str(result.run_id),
        "job_id": str(result.job_id) if result.job_id else None,
        "skipped": result.skipped,
    }


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True, autoretry_for=(), name="clipfactory.run_candidate_refinement"
)
def run_candidate_refinement(
    self: Task,
    candidate_id: str,
    priority: str,
    refinement_id: str,
    job_id: str | None = None,
    force: bool = False,
) -> dict[str, str | bool | None]:
    """Run one explicit candidate-scoped Stage 3.5 refinement.

    This is an extension of the existing job system, not a pipeline stage: it
    creates no ``PipelineRun`` and never touches the whole-source stage chain.
    """
    from uuid import UUID as _UUID

    parsed_refinement = _UUID(refinement_id)
    parsed_job = _UUID(job_id) if job_id else None
    session = create_session_factory()()
    try:
        settings = get_settings()
        storage = StorageService(settings.storage_root)
        if parsed_job is not None:
            job = session.get(ProcessingJob, parsed_job)
            if job is not None and job.status is JobStatus.CANCELLED:
                return {"refinement_id": str(parsed_refinement), "cancelled": True}
            if job is not None:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(timezone.utc)
                session.commit()
        executor = build_candidate_refinement_executor(session, storage, settings)
        executor.set_active_job(parsed_job)
        try:
            executor.execute(parsed_refinement, force=force)
        except StageCancelled:
            if parsed_job is not None:
                cancelled_job = session.get(ProcessingJob, parsed_job)
                if cancelled_job is not None and cancelled_job.status is not JobStatus.CANCELLED:
                    cancelled_job.status = JobStatus.CANCELLED
                    cancelled_job.completed_at = datetime.now(timezone.utc)
                    session.commit()
            return {"refinement_id": str(parsed_refinement), "cancelled": True}
        except Exception as error:
            if parsed_job is not None:
                failed_job = session.get(ProcessingJob, parsed_job)
                if failed_job is not None and failed_job.status is not JobStatus.CANCELLED:
                    failed_job.status = JobStatus.FAILED
                    failed_job.completed_at = datetime.now(timezone.utc)
                    failed_job.error_message = type(error).__name__[:2048]
                    session.commit()
            if getattr(error, "retryable", False):
                raise self.retry(
                    args=[candidate_id, priority, refinement_id, job_id, force],
                    exc=error,
                    max_retries=MAX_RETRIES,
                ) from error
            raise
        if parsed_job is not None:
            finished_job = session.get(ProcessingJob, parsed_job)
            if finished_job is not None and finished_job.status is not JobStatus.CANCELLED:
                finished_job.status = JobStatus.SUCCEEDED
                finished_job.completed_at = datetime.now(timezone.utc)
                session.commit()
        return {
            "refinement_id": str(parsed_refinement),
            "job_id": str(parsed_job) if parsed_job else None,
            "skipped": False,
        }
    finally:
        session.close()


@celery_app.task(name="clipfactory.worker_heartbeat")  # type: ignore[untyped-decorator]
def worker_heartbeat() -> dict[str, str]:
    """Expose latest worker liveness timestamp for health checks."""
    global _last_heartbeat
    _last_heartbeat = datetime.now(timezone.utc)
    return {"recorded_at": _last_heartbeat.isoformat()}


def last_heartbeat() -> datetime | None:
    """Return heartbeat seen by this worker process."""
    return _last_heartbeat
