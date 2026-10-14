"""Stage 6 materialization queueing, prerequisite validation, idempotency.

Prerequisites: a current retained candidate, a current live-effective selected
Stage 4 plan, a current live-effective executable Stage 5.0 render contract
bound to that selection/plan/FINAL_CLIP, and a current ready Stage 5.1
composition bound to the contract. Queueing extends the existing Celery /
``ProcessingJob`` system with the ``MATERIALIZATION`` job kind; it adds no
``PipelineStage``, no ``PipelineRun``, and no ``_NEXT_STAGE`` entry, and never
touches the source lifecycle.

At most one active Stage 6 job exists per candidate/refinement scope: an atomic
compare-and-swap on ``processing_jobs.task_id`` (the durable dispatch token)
guarantees a single live run.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.enums import JobKind, JobStatus
from app.core.settings import Settings
from app.materialization.service import MaterializationPrerequisites, resolve_prerequisites
from app.models import ProcessingJob

_ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}


class MaterializationQueueError(ValueError):
    """A Stage 6 queue request was stale, rejected, or missing a prerequisite."""


@dataclass(frozen=True)
class MaterializationQueueOutcome:
    job_id: uuid.UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool
    skipped_reason: str | None = None


def validate_candidate_for_materialization(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    profile_key: str | None = None,
) -> MaterializationPrerequisites:
    """Validate every current, live-effective Stage 6 prerequisite."""

    return resolve_prerequisites(session, candidate_id, profile_key=profile_key)


def _active_job(session: Session, refinement_id: uuid.UUID) -> ProcessingJob | None:
    jobs = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.candidate_refinement_id == refinement_id,
            ProcessingJob.kind == JobKind.MATERIALIZATION,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).all()
    return jobs[0] if jobs else None


def _claim_job(session: Session, job_id: uuid.UUID) -> bool:
    result = session.execute(
        update(ProcessingJob)
        .where(
            ProcessingJob.id == job_id,
            ProcessingJob.status == JobStatus.QUEUED,
            ProcessingJob.task_id.is_(None),
        )
        .values(task_id=str(job_id))
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def queue_materialization(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    priority: str = "CANDIDATE",
    profile_key: str | None = None,
    supporting_requested: str = "OFF",
    force: bool = False,
    settings: Settings | None = None,
) -> MaterializationQueueOutcome:
    """Queue (or reuse) one bounded Stage 6 materialization run for a candidate."""

    prereq = validate_candidate_for_materialization(session, candidate_id, profile_key=profile_key)
    refinement_id = prereq.final_clip.id

    active = _active_job(session, refinement_id)
    if active is not None:
        return MaterializationQueueOutcome(
            job_id=active.id,
            status=active.status.value,
            queued=False,
            cached=False,
            active=True,
        )

    job = ProcessingJob(
        source_video_id=prereq.source.id,
        kind=JobKind.MATERIALIZATION,
        candidate_refinement_id=refinement_id,
    )
    session.add(job)
    session.flush()
    if not _claim_job(session, job.id):
        session.rollback()
        active = _active_job(session, refinement_id)
        if active is not None:
            return MaterializationQueueOutcome(
                job_id=active.id,
                status=active.status.value,
                queued=False,
                cached=False,
                active=True,
            )
        raise MaterializationQueueError("could not claim a Stage 6 job; retry the request")
    session.commit()
    session.refresh(job)
    _dispatch(
        str(prereq.candidate.id),
        str(job.id),
        priority=priority,
        profile_key=profile_key,
        supporting_requested=supporting_requested,
        force=force,
    )
    return MaterializationQueueOutcome(
        job_id=job.id,
        status=job.status.value,
        queued=True,
        cached=False,
        active=False,
    )


def get_job(session: Session, job_id: uuid.UUID | str) -> ProcessingJob | None:
    try:
        parsed = job_id if isinstance(job_id, uuid.UUID) else uuid.UUID(str(job_id))
    except (TypeError, ValueError):
        return None
    return session.get(ProcessingJob, parsed)


def _dispatch(
    candidate_id: str,
    job_id: str,
    *,
    priority: str,
    profile_key: str | None,
    supporting_requested: str,
    force: bool,
) -> None:
    from app.workers.tasks import run_materialization_job

    run_materialization_job.delay(
        candidate_id,
        job_id,
        priority,
        profile_key,
        supporting_requested,
        force,
    )


__all__ = [
    "MaterializationQueueError",
    "MaterializationQueueOutcome",
    "get_job",
    "queue_materialization",
    "validate_candidate_for_materialization",
]
