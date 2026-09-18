"""Stage 4.1 queueing, prerequisite validation, and idempotency.

Prerequisites: a current retained candidate, a usable Stage 3.5 refinement, and
a current, non-stale Stage 4.0 analysis with at least one current recommended
strategy. Cache validation uses the same settings-derived Stage 4.1
configuration, provider mode, and stable configured provider identity that
execution uses. Plan-set creation and active-job claiming are concurrency-safe.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import CandidateDisposition, JobKind, JobStatus, StrategyDisposition
from app.core.settings import get_settings
from app.models import (
    ClipCandidate,
    ProcessingJob,
    TransformationEligibilityAnalysis,
    TransformationPlan,
    TransformationPlanSet,
)
from app.transformation.handoff import build_stage4_1_handoff
from app.transformation.inputs import resolve_effective_refinement
from app.transformation.planning.executor import build_transformation_planning_executor
from app.transformation.queue import list_strategies

_VALID_DISPOSITIONS = {
    CandidateDisposition.CANDIDATE,
    CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
}
_ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}
_CACHEABLE_STATUS = {"COMPLETE", "PROVIDER_DEGRADED"}


class PlanningQueueError(ValueError):
    """A Stage 4.1 queue request was stale, rejected, or missing a prerequisite."""


@dataclass(frozen=True)
class PlanningQueueOutcome:
    plan_set_id: uuid.UUID
    job_id: uuid.UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool
    skipped_reason: str | None = None


def validate_candidate_for_planning(
    session: Session, candidate_id: uuid.UUID | str
) -> tuple[ClipCandidate, TransformationEligibilityAnalysis]:
    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        raise PlanningQueueError("candidate does not exist")
    if not candidate.is_current:
        raise PlanningQueueError("candidate is stale and cannot be planned")
    if candidate.disposition not in _VALID_DISPOSITIONS:
        raise PlanningQueueError("candidate was not retained for transformation")
    if resolve_effective_refinement(session, candidate) is None:
        raise PlanningQueueError("candidate has no usable Stage 3.5 refinement prerequisite")
    handoff = build_stage4_1_handoff(session, candidate.id)
    if handoff is None:
        raise PlanningQueueError("candidate does not exist")
    if handoff.get("stale"):
        raise PlanningQueueError("Stage 4.0 analysis is stale; re-run Stage 4.0 first")
    if not handoff.get("ready_for_stage4_1"):
        raise PlanningQueueError(
            "Stage 4.0 analysis has no current recommended strategy or is not complete"
        )
    analysis = session.get(TransformationEligibilityAnalysis, _as_uuid(str(handoff["analysis_id"])))
    if analysis is None:
        raise PlanningQueueError("Stage 4.0 analysis is missing")
    return candidate, analysis


def _find_plan_set(session: Session, candidate_id: uuid.UUID) -> TransformationPlanSet | None:
    return session.scalar(
        select(TransformationPlanSet).where(TransformationPlanSet.clip_candidate_id == candidate_id)
    )


def get_or_create_plan_set(
    session: Session, candidate: ClipCandidate, analysis: TransformationEligibilityAnalysis
) -> TransformationPlanSet:
    """Return the current plan set, creating it once under concurrency."""

    existing = _find_plan_set(session, candidate.id)
    if existing is not None:
        return existing
    plan_set = TransformationPlanSet(
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        transformation_analysis_id=analysis.id,
    )
    try:
        with session.begin_nested():
            session.add(plan_set)
            session.flush()
    except IntegrityError:
        existing = _find_plan_set(session, candidate.id)
        if existing is None:
            raise
        return existing
    session.commit()
    session.refresh(plan_set)
    return plan_set


def _active_job(session: Session, plan_set: TransformationPlanSet) -> ProcessingJob | None:
    jobs = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.transformation_plan_set_id == plan_set.id,
            ProcessingJob.kind == JobKind.TRANSFORMATION_PLANNING,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).all()
    return jobs[0] if jobs else None


def _active_outcome(plan_set: TransformationPlanSet, job: ProcessingJob) -> PlanningQueueOutcome:
    return PlanningQueueOutcome(
        plan_set_id=plan_set.id,
        job_id=job.id,
        status=plan_set.execution_status.value,
        queued=False,
        cached=False,
        active=True,
    )


def _claim_plan_set(session: Session, plan_set_id: uuid.UUID, job_id: uuid.UUID) -> bool:
    result = session.execute(
        update(TransformationPlanSet)
        .where(
            TransformationPlanSet.id == plan_set_id,
            TransformationPlanSet.active_job_id.is_(None),
        )
        .values(active_job_id=str(job_id))
    )
    return int(result.rowcount) == 1


def _clear_stale_claim(session: Session, plan_set: TransformationPlanSet) -> None:
    stale = plan_set.active_job_id
    if stale is None:
        return
    session.execute(
        update(TransformationPlanSet)
        .where(
            TransformationPlanSet.id == plan_set.id,
            TransformationPlanSet.active_job_id == stale,
        )
        .values(active_job_id=None)
    )
    session.commit()
    session.refresh(plan_set)


def _matches_cache(
    session: Session,
    plan_set: TransformationPlanSet,
    settings: object,
) -> bool:
    if not plan_set.cache_eligible or not plan_set.input_fingerprint:
        return False
    if plan_set.execution_status.value not in _CACHEABLE_STATUS:
        return False
    try:
        executor = build_transformation_planning_executor(session, settings)
        current = executor.input_fingerprint(plan_set)
    except Exception:
        return False
    return bool(current) and current == plan_set.input_fingerprint


def queue_transformation_planning(
    session: Session,
    candidate: ClipCandidate,
    analysis: TransformationEligibilityAnalysis,
    *,
    force: bool = False,
) -> PlanningQueueOutcome:
    settings = get_settings()
    plan_set = get_or_create_plan_set(session, candidate, analysis)
    for _attempt in range(2):
        active = _active_job(session, plan_set)
        if active is not None:
            return _active_outcome(plan_set, active)
        if plan_set.active_job_id is not None:
            _clear_stale_claim(session, plan_set)
            continue
        if not force and _matches_cache(session, plan_set, settings):
            return PlanningQueueOutcome(
                plan_set_id=plan_set.id,
                job_id=None,
                status=plan_set.execution_status.value,
                queued=False,
                cached=True,
                active=False,
            )
        job = ProcessingJob(
            source_video_id=candidate.source_video_id,
            kind=JobKind.TRANSFORMATION_PLANNING,
            transformation_plan_set_id=plan_set.id,
        )
        session.add(job)
        session.flush()
        if _claim_plan_set(session, plan_set.id, job.id):
            session.commit()
            session.refresh(job)
            _dispatch(plan_set.id, job.id, force)
            return PlanningQueueOutcome(
                plan_set_id=plan_set.id,
                job_id=job.id,
                status=plan_set.execution_status.value,
                queued=True,
                cached=False,
                active=False,
            )
        session.rollback()
        refreshed = session.get(TransformationPlanSet, plan_set.id)
        if refreshed is None:
            raise PlanningQueueError("plan set vanished during queueing")
        plan_set = refreshed
    active = _active_job(session, plan_set)
    if active is not None:
        return _active_outcome(plan_set, active)
    raise PlanningQueueError("could not claim a Stage 4.1 job; retry the request")


def get_plan_set(session: Session, plan_set_id: uuid.UUID | str) -> TransformationPlanSet | None:
    return session.get(TransformationPlanSet, _as_uuid(plan_set_id))


def get_plan_set_for_candidate(
    session: Session, candidate_id: uuid.UUID | str
) -> TransformationPlanSet | None:
    return session.scalar(
        select(TransformationPlanSet).where(
            TransformationPlanSet.clip_candidate_id == _as_uuid(candidate_id)
        )
    )


def list_plans(session: Session, plan_set_id: uuid.UUID | str) -> list[TransformationPlan]:
    return list(
        session.scalars(
            select(TransformationPlan)
            .where(TransformationPlan.plan_set_id == _as_uuid(plan_set_id))
            .order_by(
                TransformationPlan.is_current.desc(),
                TransformationPlan.generation_rank.asc(),
            )
        ).all()
    )


def has_current_recommended_strategy(
    session: Session, analysis: TransformationEligibilityAnalysis
) -> bool:
    return any(
        row.is_current and row.disposition is StrategyDisposition.RECOMMENDED
        for row in list_strategies(session, analysis.id)
    )


def _dispatch(plan_set_id: uuid.UUID, job_id: uuid.UUID, force: bool) -> None:
    from app.workers.tasks import run_transformation_planning

    run_transformation_planning.delay(str(plan_set_id), str(job_id), force)


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as error:
        raise PlanningQueueError("invalid identifier") from error
