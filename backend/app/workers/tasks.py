"""Durable Celery wrappers for pipeline work."""

from datetime import datetime, timezone
from typing import Final
from uuid import UUID

from celery import Task  # type: ignore[import-not-found]

from app.core.enums import JobKind, PipelineStage
from app.db.session import create_session_factory
from app.models import ProcessingJob
from app.pipeline.executor import StageExecutor
from app.pipeline.runner import PipelineRunner
from app.workers.celery_app import celery_app

_executors: dict[PipelineStage, StageExecutor] = {}
_last_heartbeat: datetime | None = None
MAX_RETRIES: Final = 3


def register_stage_executor(stage: PipelineStage, executor: StageExecutor) -> None:
    """Register concrete work without coupling orchestration to media services."""
    _executors[stage] = executor


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True, autoretry_for=(), name="clipfactory.run_pipeline_stage"
)
def run_pipeline_stage(
    self: Task, source_id: str, stage: str, job_id: str | None = None
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
        runner = PipelineRunner(session, _executors)
        try:
            result = runner.run(UUID(source_id), parsed_stage, job_id=parsed_job_id)
        except Exception as error:
            if getattr(error, "retryable", False):
                if parsed_job_id is not None:
                    retry_job = session.get(ProcessingJob, parsed_job_id)
                    if retry_job is not None:
                        retry_job.retry_count += 1
                        session.commit()
                raise self.retry(
                    args=[source_id, stage, str(parsed_job_id) if parsed_job_id else None],
                    exc=error,
                    max_retries=MAX_RETRIES,
                ) from error
            raise
    finally:
        session.close()
    return {
        "run_id": str(result.run_id),
        "job_id": str(result.job_id) if result.job_id else None,
        "skipped": result.skipped,
    }


@celery_app.task(name="clipfactory.worker_heartbeat")  # type: ignore[untyped-decorator]
def worker_heartbeat() -> dict[str, str]:
    """Expose latest worker liveness timestamp for health checks."""
    global _last_heartbeat
    _last_heartbeat = datetime.now(timezone.utc)
    return {"recorded_at": _last_heartbeat.isoformat()}


def last_heartbeat() -> datetime | None:
    """Return heartbeat seen by this worker process."""
    return _last_heartbeat
