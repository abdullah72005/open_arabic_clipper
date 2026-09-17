"""Stage 4.2 queueing, prerequisite validation, and idempotency.

Prerequisites: a current retained candidate and a current, non-stale, complete
Stage 4.1 plan set with at least one current plan. Cache validation uses the
same settings-derived Stage 4.2 configuration, provider mode, and stable
configured provider identity that execution uses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import JobKind, JobStatus
from app.core.settings import get_settings
from app.models import (
    ClipCandidate,
    ProcessingJob,
    TransformationGovernanceResult,
    TransformationGovernanceSet,
    TransformationPlanSet,
)
from app.transformation.governance.executor import (
    JOB_CLAIM_STALE_SECONDS,
    build_transformation_governance_executor,
)
from app.transformation.governance.inputs import (
    GovernanceInputError,
)
from app.transformation.governance.inputs import (
    validate_candidate_for_governance as _validate_inputs,
)

_ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}
_CACHEABLE_STATUS = {"COMPLETE", "PROVIDER_DEGRADED"}


class GovernanceQueueError(ValueError):
    """A Stage 4.2 queue request was stale, rejected, or missing a prerequisite."""


def validate_candidate_for_governance(
    session: Session, candidate_id: uuid.UUID | str
) -> tuple[ClipCandidate, TransformationPlanSet]:
    """Queue-gate prerequisite validation with a bounded queue error."""

    try:
        return _validate_inputs(session, candidate_id)
    except GovernanceInputError as error:
        raise GovernanceQueueError(str(error)) from error


@dataclass(frozen=True)
class GovernanceQueueOutcome:
    governance_set_id: uuid.UUID
    job_id: uuid.UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool
    skipped_reason: str | None = None


def _find_governance_set(
    session: Session, candidate_id: uuid.UUID
) -> TransformationGovernanceSet | None:
    return session.scalar(
        select(TransformationGovernanceSet).where(
            TransformationGovernanceSet.clip_candidate_id == candidate_id
        )
    )


def get_or_create_governance_set(
    session: Session, candidate: ClipCandidate, plan_set: TransformationPlanSet
) -> TransformationGovernanceSet:
    """Return the current governance set, creating it once under concurrency."""

    existing = _find_governance_set(session, candidate.id)
    if existing is not None:
        return existing
    governance_set = TransformationGovernanceSet(
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        transformation_plan_set_id=plan_set.id,
        transformation_analysis_id=plan_set.transformation_analysis_id,
        refinement_id=plan_set.refinement_id,
        refinement_priority=plan_set.refinement_priority,
        refinement_quality_level=plan_set.refinement_quality_level,
    )
    try:
        with session.begin_nested():
            session.add(governance_set)
            session.flush()
    except IntegrityError:
        existing = _find_governance_set(session, candidate.id)
        if existing is None:
            raise
        return existing
    session.commit()
    session.refresh(governance_set)
    return governance_set


def _active_job(
    session: Session, governance_set: TransformationGovernanceSet
) -> ProcessingJob | None:
    jobs = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.transformation_governance_set_id == governance_set.id,
            ProcessingJob.kind == JobKind.TRANSFORMATION_GOVERNANCE,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).all()
    return jobs[0] if jobs else None


def _active_outcome(
    governance_set: TransformationGovernanceSet,
    job: ProcessingJob,
    *,
    skipped_reason: str | None = None,
) -> GovernanceQueueOutcome:
    return GovernanceQueueOutcome(
        governance_set_id=governance_set.id,
        job_id=job.id,
        status=governance_set.execution_status.value,
        queued=False,
        cached=False,
        active=True,
        skipped_reason=skipped_reason,
    )


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _job_heartbeat_stale(job: ProcessingJob, *, now: datetime | None = None) -> bool:
    """Whether a RUNNING job's liveness is genuinely abandoned.

    Matches the executor's stale-reclaim rule: an old ``started_at`` alone is not
    enough; the heartbeat (or ``started_at`` when no heartbeat exists) must be
    older than the shared stale window. A live/queued job is never stale.
    """

    if job.status is not JobStatus.RUNNING:
        return False
    current = now or datetime.now(timezone.utc)
    threshold = current - timedelta(seconds=JOB_CLAIM_STALE_SECONDS)
    heartbeat = _as_utc(job.heartbeat_at)
    if heartbeat is not None:
        return heartbeat < threshold
    started = _as_utc(job.started_at)
    if started is not None:
        return started < threshold
    return False


def _claim_governance_set(
    session: Session, governance_set_id: uuid.UUID, job_id: uuid.UUID
) -> bool:
    result = session.execute(
        update(TransformationGovernanceSet)
        .where(
            TransformationGovernanceSet.id == governance_set_id,
            TransformationGovernanceSet.active_job_id.is_(None),
        )
        .values(active_job_id=str(job_id))
    )
    return int(result.rowcount) == 1


def _clear_stale_claim(session: Session, governance_set: TransformationGovernanceSet) -> None:
    stale = governance_set.active_job_id
    if stale is None:
        return
    session.execute(
        update(TransformationGovernanceSet)
        .where(
            TransformationGovernanceSet.id == governance_set.id,
            TransformationGovernanceSet.active_job_id == stale,
        )
        .values(active_job_id=None)
    )
    session.commit()
    session.refresh(governance_set)


def _matches_cache(
    session: Session, governance_set: TransformationGovernanceSet, settings: object
) -> bool:
    if not governance_set.cache_eligible or not governance_set.input_fingerprint:
        return False
    if governance_set.execution_status.value not in _CACHEABLE_STATUS:
        return False
    try:
        executor = build_transformation_governance_executor(session, settings)
        current = executor.input_fingerprint(governance_set)
    except Exception:
        return False
    return bool(current) and current == governance_set.input_fingerprint


def queue_transformation_governance(
    session: Session,
    candidate: ClipCandidate,
    plan_set: TransformationPlanSet,
    *,
    force: bool = False,
) -> GovernanceQueueOutcome:
    settings = get_settings()
    governance_set = get_or_create_governance_set(session, candidate, plan_set)
    for _attempt in range(2):
        active = _active_job(session, governance_set)
        if active is not None:
            if _job_heartbeat_stale(active):
                # Recover a genuinely abandoned RUNNING job by re-dispatching the
                # same job id: the executor's atomic claim performs the
                # claim-version-bumping stale reclaim, so the old worker can
                # neither persist nor finalize a newer result, and a live job is
                # never reclaimed.
                _dispatch(governance_set.id, active.id, force)
                return _active_outcome(
                    governance_set, active, skipped_reason="RECOVERED_STALE_RUNNING"
                )
            return _active_outcome(governance_set, active)
        if governance_set.active_job_id is not None:
            _clear_stale_claim(session, governance_set)
            continue
        if not force and _matches_cache(session, governance_set, settings):
            return GovernanceQueueOutcome(
                governance_set_id=governance_set.id,
                job_id=None,
                status=governance_set.execution_status.value,
                queued=False,
                cached=True,
                active=False,
            )
        job = ProcessingJob(
            source_video_id=candidate.source_video_id,
            kind=JobKind.TRANSFORMATION_GOVERNANCE,
            transformation_governance_set_id=governance_set.id,
        )
        session.add(job)
        session.flush()
        if _claim_governance_set(session, governance_set.id, job.id):
            session.commit()
            session.refresh(job)
            _dispatch(governance_set.id, job.id, force)
            return GovernanceQueueOutcome(
                governance_set_id=governance_set.id,
                job_id=job.id,
                status=governance_set.execution_status.value,
                queued=True,
                cached=False,
                active=False,
            )
        session.rollback()
        refreshed = session.get(TransformationGovernanceSet, governance_set.id)
        if refreshed is None:
            raise GovernanceQueueError("governance set vanished during queueing")
        governance_set = refreshed
    active = _active_job(session, governance_set)
    if active is not None:
        return _active_outcome(governance_set, active)
    raise GovernanceQueueError("could not claim a Stage 4.2 job; retry the request")


def get_governance_set(
    session: Session, governance_set_id: uuid.UUID | str
) -> TransformationGovernanceSet | None:
    return session.get(TransformationGovernanceSet, _as_uuid(governance_set_id))


def get_governance_set_for_candidate(
    session: Session, candidate_id: uuid.UUID | str
) -> TransformationGovernanceSet | None:
    return session.scalar(
        select(TransformationGovernanceSet).where(
            TransformationGovernanceSet.clip_candidate_id == _as_uuid(candidate_id)
        )
    )


def list_results(
    session: Session, governance_set_id: uuid.UUID | str
) -> list[TransformationGovernanceResult]:
    return list(
        session.scalars(
            select(TransformationGovernanceResult)
            .where(TransformationGovernanceResult.governance_set_id == _as_uuid(governance_set_id))
            .order_by(TransformationGovernanceResult.created_at.asc())
        ).all()
    )


def get_result_for_plan(
    session: Session,
    governance_set_id: uuid.UUID | str,
    transformation_plan_id: uuid.UUID | str,
) -> TransformationGovernanceResult | None:
    return session.scalar(
        select(TransformationGovernanceResult).where(
            TransformationGovernanceResult.governance_set_id == _as_uuid(governance_set_id),
            TransformationGovernanceResult.transformation_plan_id
            == _as_uuid(transformation_plan_id),
        )
    )


def _dispatch(governance_set_id: uuid.UUID, job_id: uuid.UUID, force: bool) -> None:
    from app.workers.tasks import run_transformation_governance

    run_transformation_governance.delay(str(governance_set_id), str(job_id), force)


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as error:
        raise GovernanceQueueError("invalid identifier") from error


__all__ = [
    "GovernanceQueueError",
    "GovernanceQueueOutcome",
    "get_governance_set",
    "get_governance_set_for_candidate",
    "get_or_create_governance_set",
    "get_result_for_plan",
    "list_results",
    "queue_transformation_governance",
    "validate_candidate_for_governance",
]
