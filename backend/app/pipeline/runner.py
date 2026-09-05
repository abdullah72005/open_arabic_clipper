"""Idempotent durable pipeline stage runner."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import JobKind, JobStatus, PipelineRunStatus, PipelineStage
from app.models import PipelineRun, ProcessingJob, SourceVideo
from app.pipeline.executor import StageExecutor


class StageExecutionError(RuntimeError):
    """A stage failure whose retry policy is safe for workers to inspect."""

    retryable = False


class RetryableStageError(StageExecutionError):
    """A temporary stage failure eligible for Celery retry."""

    retryable = True


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of one attempted stage execution."""

    run_id: UUID
    job_id: UUID | None
    skipped: bool = False


class PipelineRunner:
    """Persist stage and job transitions around an injected stage executor."""

    def __init__(self, session: Session, executors: Mapping[PipelineStage, StageExecutor]) -> None:
        self._session = session
        self._executors = executors

    def run(
        self,
        source_id: UUID,
        stage: PipelineStage,
        *,
        job_id: UUID | None = None,
    ) -> PipelineResult:
        source = self._require_source(source_id)
        run = self._latest_run(source.id, stage)
        if run is not None and run.status is PipelineRunStatus.SUCCEEDED:
            return PipelineResult(run.id, job_id, skipped=True)

        job = self._load_or_create_job(source.id, stage, job_id)
        now = datetime.now(timezone.utc)
        if run is None:
            run = PipelineRun(source_video_id=source.id, stage=stage)
            self._session.add(run)
        elif run.status in {PipelineRunStatus.FAILED, PipelineRunStatus.CANCELLED}:
            run.attempt += 1
        run.status = PipelineRunStatus.RUNNING
        run.error_message = None
        run.started_at = now
        run.completed_at = None
        if job is not None:
            job.status = JobStatus.RUNNING
            job.error_code = None
            job.error_message = None
            job.started_at = now
            job.completed_at = None
        self._session.commit()

        executor = self._executors.get(stage)
        if executor is None:
            error = StageExecutionError(f"no executor registered for stage {stage.value}")
            self._persist_failure(run, job, error)
            raise error
        try:
            executor.execute(source)
        except Exception as error:
            self._persist_failure(run, job, error)
            raise

        completed_at = datetime.now(timezone.utc)
        run.status = PipelineRunStatus.SUCCEEDED
        run.completed_at = completed_at
        if job is not None:
            job.status = JobStatus.SUCCEEDED
            job.completed_at = completed_at
        source.lifecycle_state = _next_stage(stage)
        self._session.commit()
        return PipelineResult(run.id, job.id if job is not None else None)

    def retry(self, job_id: UUID) -> PipelineResult:
        job = self._session.get(ProcessingJob, job_id)
        if job is None:
            raise LookupError(f"processing job {job_id} does not exist")
        if job.status not in {JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.QUEUED}:
            raise ValueError(f"processing job {job_id} is not retryable from {job.status.value}")
        job.retry_count += 1
        self._session.commit()
        return self.run(job.source_video_id, _stage_for_job_kind(job.kind), job_id=job.id)

    def _require_source(self, source_id: UUID) -> SourceVideo:
        source = self._session.get(SourceVideo, source_id)
        if source is None:
            raise LookupError(f"source video {source_id} does not exist")
        return source

    def _latest_run(self, source_id: UUID, stage: PipelineStage) -> PipelineRun | None:
        return self._session.scalar(
            select(PipelineRun)
            .where(PipelineRun.source_video_id == source_id, PipelineRun.stage == stage)
            .order_by(PipelineRun.created_at.desc())
        )

    def _load_or_create_job(
        self, source_id: UUID, stage: PipelineStage, job_id: UUID | None
    ) -> ProcessingJob | None:
        if job_id is not None:
            job = self._session.get(ProcessingJob, job_id)
            if job is None or job.source_video_id != source_id:
                raise LookupError(f"processing job {job_id} does not belong to source {source_id}")
            return job
        job = ProcessingJob(source_video_id=source_id, kind=_job_kind_for_stage(stage))
        self._session.add(job)
        return job

    def _persist_failure(
        self, run: PipelineRun, job: ProcessingJob | None, error: Exception
    ) -> None:
        completed_at = datetime.now(timezone.utc)
        message = str(error)
        run.status = PipelineRunStatus.FAILED
        run.error_message = message
        run.completed_at = completed_at
        if job is not None:
            job.status = JobStatus.FAILED
            job.error_code = type(error).__name__
            job.error_message = message
            job.completed_at = completed_at
        self._session.commit()


def _stage_for_job_kind(kind: JobKind) -> PipelineStage:
    if kind is JobKind.INGEST:
        return PipelineStage.INGEST
    if kind is JobKind.PROBE:
        return PipelineStage.PROBE
    if kind is JobKind.TRANSCRIPTION:
        return PipelineStage.TRANSCRIPTION
    if kind is JobKind.RECONSTRUCTION:
        return PipelineStage.CONTEXTUAL_RECONSTRUCTION
    raise ValueError(f"no pipeline stage is defined for job kind {kind.value}")


def _job_kind_for_stage(stage: PipelineStage) -> JobKind:
    if stage is PipelineStage.PROBE:
        return JobKind.PROBE
    if stage is PipelineStage.TRANSCRIPTION:
        return JobKind.TRANSCRIPTION
    if stage is PipelineStage.CONTEXTUAL_RECONSTRUCTION:
        return JobKind.RECONSTRUCTION
    return JobKind.INGEST


def _next_stage(stage: PipelineStage) -> PipelineStage:
    if stage is PipelineStage.INGEST:
        return PipelineStage.PROBE
    if stage is PipelineStage.PROBE:
        return PipelineStage.AUDIO_EXTRACTION
    if stage is PipelineStage.AUDIO_EXTRACTION:
        return PipelineStage.TRANSCRIPTION
    if stage is PipelineStage.TRANSCRIPTION:
        return PipelineStage.TRANSCRIPT_NORMALIZATION
    if stage is PipelineStage.TRANSCRIPT_NORMALIZATION:
        return PipelineStage.CONTEXTUAL_RECONSTRUCTION
    if stage is PipelineStage.CONTEXTUAL_RECONSTRUCTION:
        return PipelineStage.AUDIO_ANALYSIS
    if stage is PipelineStage.AUDIO_ANALYSIS:
        return PipelineStage.READY_FOR_ANALYSIS
    return PipelineStage.READY_FOR_ANALYSIS
