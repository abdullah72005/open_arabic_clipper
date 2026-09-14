"""Stage 4.0 queueing, prerequisite validation, and idempotency."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import CandidateDisposition, JobKind, JobStatus
from app.models import (
    ClipCandidate,
    ProcessingJob,
    TransformationEligibilityAnalysis,
    TransformationStrategyCandidate,
)
from app.transformation.inputs import resolve_effective_refinement
from app.transformation.policy import DEFAULT_CONFIG

_VALID_DISPOSITIONS = {
    CandidateDisposition.CANDIDATE,
    CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
}
_ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}
_CACHEABLE_STATUS = {"COMPLETE"}


class TransformationQueueError(ValueError):
    """A Stage 4.0 queue request was stale, rejected, or missing a prerequisite."""


@dataclass(frozen=True)
class TransformationQueueOutcome:
    analysis_id: uuid.UUID
    job_id: uuid.UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool
    skipped_reason: str | None = None


def validate_candidate_for_transformation(
    session: Session, candidate_id: uuid.UUID | str
) -> ClipCandidate:
    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        raise TransformationQueueError("candidate does not exist")
    if not candidate.is_current:
        raise TransformationQueueError("candidate is stale and cannot be analyzed")
    if candidate.disposition not in _VALID_DISPOSITIONS:
        raise TransformationQueueError("candidate was not retained for transformation")
    if resolve_effective_refinement(session, candidate) is None:
        raise TransformationQueueError(
            "candidate has no completed, usable Stage 3.5 refinement prerequisite"
        )
    return candidate


def get_or_create_analysis(
    session: Session, candidate: ClipCandidate
) -> TransformationEligibilityAnalysis:
    existing = session.scalar(
        select(TransformationEligibilityAnalysis).where(
            TransformationEligibilityAnalysis.clip_candidate_id == candidate.id
        )
    )
    if existing is not None:
        return existing
    analysis = TransformationEligibilityAnalysis(
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
    )
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    return analysis


def _active_job(
    session: Session, analysis: TransformationEligibilityAnalysis
) -> ProcessingJob | None:
    jobs = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.transformation_analysis_id == analysis.id,
            ProcessingJob.kind == JobKind.TRANSFORMATION_ELIGIBILITY,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).all()
    return jobs[0] if jobs else None


def _matches_cache(
    session: Session, candidate: ClipCandidate, analysis: TransformationEligibilityAnalysis
) -> bool:
    if not analysis.cache_eligible or not analysis.input_fingerprint:
        return False
    if analysis.execution_status.value not in _CACHEABLE_STATUS:
        return False
    from app.transformation.executor import TransformationEligibilityExecutor

    executor = TransformationEligibilityExecutor(session=session, config=DEFAULT_CONFIG)
    try:
        current = executor.input_fingerprint(candidate)
    except Exception:
        return False
    return bool(current) and current == analysis.input_fingerprint


def queue_transformation_analysis(
    session: Session,
    candidate: ClipCandidate,
    *,
    force: bool = False,
) -> TransformationQueueOutcome:
    analysis = get_or_create_analysis(session, candidate)
    active = _active_job(session, analysis)
    if active is not None:
        return TransformationQueueOutcome(
            analysis_id=analysis.id,
            job_id=active.id,
            status=analysis.execution_status.value,
            queued=False,
            cached=False,
            active=True,
        )
    if not force and _matches_cache(session, candidate, analysis):
        return TransformationQueueOutcome(
            analysis_id=analysis.id,
            job_id=None,
            status=analysis.execution_status.value,
            queued=False,
            cached=True,
            active=False,
        )
    job = ProcessingJob(
        source_video_id=candidate.source_video_id,
        kind=JobKind.TRANSFORMATION_ELIGIBILITY,
        transformation_analysis_id=analysis.id,
    )
    session.add(job)
    session.flush()
    analysis.active_job_id = str(job.id)
    session.commit()
    session.refresh(job)
    _dispatch(analysis.id, job.id, force)
    return TransformationQueueOutcome(
        analysis_id=analysis.id,
        job_id=job.id,
        status=analysis.execution_status.value,
        queued=True,
        cached=False,
        active=False,
    )


def get_analysis(
    session: Session, analysis_id: uuid.UUID | str
) -> TransformationEligibilityAnalysis | None:
    return session.get(TransformationEligibilityAnalysis, _as_uuid(analysis_id))


def get_analysis_for_candidate(
    session: Session, candidate_id: uuid.UUID | str
) -> TransformationEligibilityAnalysis | None:
    return session.scalar(
        select(TransformationEligibilityAnalysis).where(
            TransformationEligibilityAnalysis.clip_candidate_id == _as_uuid(candidate_id)
        )
    )


def list_strategies(
    session: Session, analysis_id: uuid.UUID | str
) -> list[TransformationStrategyCandidate]:
    return list(
        session.scalars(
            select(TransformationStrategyCandidate)
            .where(TransformationStrategyCandidate.analysis_id == _as_uuid(analysis_id))
            .order_by(
                TransformationStrategyCandidate.is_current.desc(),
                TransformationStrategyCandidate.rank.asc(),
                TransformationStrategyCandidate.strategy_type.asc(),
            )
        ).all()
    )


def _dispatch(analysis_id: uuid.UUID, job_id: uuid.UUID, force: bool) -> None:
    from app.workers.tasks import run_transformation_analysis

    run_transformation_analysis.delay(str(analysis_id), str(job_id), force)


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as error:
        raise TransformationQueueError("invalid identifier") from error
