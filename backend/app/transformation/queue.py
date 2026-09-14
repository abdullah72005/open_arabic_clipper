"""Stage 4.0 queueing, prerequisite validation, and idempotency.

Cache validation uses the same settings-derived Stage 4.0 configuration,
provider mode, and provider runtime identity that execution uses, so an
adaptive/local-only cached analysis is reused instead of re-queued. Analysis
creation and active-job claiming are concurrency-safe: a unique-constraint race
on the analysis is recovered, and an atomic compare-and-swap on
``active_job_id`` guarantees at most one active Stage 4.0 job per candidate.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import CandidateDisposition, JobKind, JobStatus
from app.core.settings import get_settings
from app.models import (
    ClipCandidate,
    ProcessingJob,
    TransformationEligibilityAnalysis,
    TransformationStrategyCandidate,
)
from app.transformation.executor import build_transformation_executor
from app.transformation.inputs import resolve_effective_refinement

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


def _find_analysis(
    session: Session, candidate_id: uuid.UUID
) -> TransformationEligibilityAnalysis | None:
    return session.scalar(
        select(TransformationEligibilityAnalysis).where(
            TransformationEligibilityAnalysis.clip_candidate_id == candidate_id
        )
    )


def get_or_create_analysis(
    session: Session, candidate: ClipCandidate
) -> TransformationEligibilityAnalysis:
    """Return the current analysis, creating it once under concurrency.

    A uniqueness race on ``clip_candidate_id`` is recovered inside a savepoint
    so one of two simultaneous callers never raises and both observe the same
    row.
    """

    existing = _find_analysis(session, candidate.id)
    if existing is not None:
        return existing
    candidate_id = candidate.id
    source_video_id = candidate.source_video_id
    analysis = TransformationEligibilityAnalysis(
        source_video_id=source_video_id,
        clip_candidate_id=candidate_id,
    )
    try:
        with session.begin_nested():
            session.add(analysis)
            session.flush()
    except IntegrityError:
        existing = _find_analysis(session, candidate_id)
        if existing is None:
            raise
        return existing
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


def _active_outcome(
    analysis: TransformationEligibilityAnalysis, job: ProcessingJob
) -> TransformationQueueOutcome:
    return TransformationQueueOutcome(
        analysis_id=analysis.id,
        job_id=job.id,
        status=analysis.execution_status.value,
        queued=False,
        cached=False,
        active=True,
    )


def _claim_analysis(session: Session, analysis_id: uuid.UUID, job_id: uuid.UUID) -> bool:
    """Atomically claim an unowned analysis for one job.

    A single conditional ``UPDATE ... WHERE active_job_id IS NULL`` is a
    compare-and-swap on both PostgreSQL and SQLite: the losing writer
    re-evaluates the predicate after the winner commits and observes zero rows.
    """

    result = session.execute(
        update(TransformationEligibilityAnalysis)
        .where(
            TransformationEligibilityAnalysis.id == analysis_id,
            TransformationEligibilityAnalysis.active_job_id.is_(None),
        )
        .values(active_job_id=str(job_id))
    )
    return int(result.rowcount) == 1


def _clear_stale_claim(session: Session, analysis: TransformationEligibilityAnalysis) -> None:
    stale = analysis.active_job_id
    if stale is None:
        return
    session.execute(
        update(TransformationEligibilityAnalysis)
        .where(
            TransformationEligibilityAnalysis.id == analysis.id,
            TransformationEligibilityAnalysis.active_job_id == stale,
        )
        .values(active_job_id=None)
    )
    session.commit()
    session.refresh(analysis)


def _matches_cache(
    session: Session,
    candidate: ClipCandidate,
    analysis: TransformationEligibilityAnalysis,
    settings: object,
) -> bool:
    if not analysis.cache_eligible or not analysis.input_fingerprint:
        return False
    if analysis.execution_status.value not in _CACHEABLE_STATUS:
        return False
    try:
        executor = build_transformation_executor(session, settings)
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
    settings = get_settings()
    analysis = get_or_create_analysis(session, candidate)
    for _attempt in range(2):
        active = _active_job(session, analysis)
        if active is not None:
            return _active_outcome(analysis, active)
        if analysis.active_job_id is not None:
            _clear_stale_claim(session, analysis)
            continue
        if not force and _matches_cache(session, candidate, analysis, settings):
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
        if _claim_analysis(session, analysis.id, job.id):
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
        # Another caller claimed while this one was between read and write.
        session.rollback()
        refreshed = session.get(TransformationEligibilityAnalysis, analysis.id)
        if refreshed is None:
            raise TransformationQueueError("transformation analysis vanished during queueing")
        analysis = refreshed
    active = _active_job(session, analysis)
    if active is not None:
        return _active_outcome(analysis, active)
    raise TransformationQueueError("could not claim a Stage 4.0 job; retry the request")


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
