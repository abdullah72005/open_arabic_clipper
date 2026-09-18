"""Stage 5.1 visual-composition queueing, prerequisite validation, idempotency.

Prerequisites: a current retained candidate plus a current, live-effective,
executable Stage 5.0 render contract. Cache reuse re-fingerprints live rows
without probing geometry (stat only). Plan-row creation and active-job claiming
are concurrency-safe: a unique-constraint race on the durable envelope row is
recovered inside a savepoint and an atomic compare-and-swap on ``active_job_id``
guarantees at most one active Stage 5.1 job per envelope.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.composition.policy import (
    FINGERPRINT_VERSION,
    SCHEMA_VERSION,
    VISUAL_COMPOSITION_POLICY_VERSION,
    Stage51Config,
    VisualCompositionExecutionStatus,
    VisualCompositionStatus,
)
from app.composition.service import (
    VisualCompositionView,
    get_current_visual_composition,
    read_visual_composition,
)
from app.composition.types import DisplayGeometry
from app.core.enums import CandidateDisposition, JobKind, JobStatus
from app.core.settings import Settings, get_settings
from app.models import ClipCandidate, ProcessingJob
from app.models.visual_composition_plan import VisualCompositionPlan as VisualCompositionPlanRow
from app.render.policy import EXECUTABLE_STATUSES
from app.render.service import get_current_render_contract, read_render_contract

_VALID_DISPOSITIONS = {
    CandidateDisposition.CANDIDATE,
    CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
}
_ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}


class CompositionQueueError(ValueError):
    """A Stage 5.1 queue request was stale, rejected, or missing a prerequisite."""


@dataclass(frozen=True)
class CompositionQueueOutcome:
    plan_id: uuid.UUID
    job_id: uuid.UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool
    skipped_reason: str | None = None


class _NoProbe:
    """Fail closed if queue-time freshness ever tries to touch the filesystem."""

    def probe(self, path: Path) -> DisplayGeometry:
        raise CompositionQueueError("queue-time freshness must never re-probe media")


def validate_candidate_for_composition(
    session: Session, candidate_id: uuid.UUID | str
) -> ClipCandidate:
    """Validate the retained candidate and its current executable render contract."""

    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        raise CompositionQueueError("candidate does not exist")
    if not candidate.is_current:
        raise CompositionQueueError("candidate is stale and cannot be composed")
    if candidate.disposition not in _VALID_DISPOSITIONS:
        raise CompositionQueueError("candidate was not retained for composition")
    row = get_current_render_contract(session, candidate.id)
    if row is None:
        raise CompositionQueueError("candidate has no Stage 5.0 render contract")
    view = read_render_contract(session, candidate.id)
    if view is None or not view.effective:
        raise CompositionQueueError("Stage 5.0 render contract is not current")
    if not row.contract_ready or row.status.value not in EXECUTABLE_STATUSES:
        raise CompositionQueueError("Stage 5.0 render contract is not executable")
    return candidate


def get_or_create_plan_row(session: Session, candidate: ClipCandidate) -> VisualCompositionPlanRow:
    """Return the current plan row or create one durable envelope under concurrency."""

    existing = get_current_visual_composition(session, candidate.id)
    if existing is not None:
        return existing
    row = VisualCompositionPlanRow(
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        status=VisualCompositionStatus.BLOCKED,
        execution_status=VisualCompositionExecutionStatus.QUEUED,
        plan_ready=False,
        is_current=True,
        input_fingerprint="",
        policy_version=VISUAL_COMPOSITION_POLICY_VERSION,
        schema_version=SCHEMA_VERSION,
        fingerprint_version=FINGERPRINT_VERSION,
    )
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
    except IntegrityError:
        concurrent = get_current_visual_composition(session, candidate.id)
        if concurrent is None:
            concurrent = _row_by_empty_fingerprint(session, candidate.id)
        if concurrent is None:
            raise
        return concurrent
    session.commit()
    session.refresh(row)
    return row


def _row_by_empty_fingerprint(
    session: Session, candidate_id: uuid.UUID
) -> VisualCompositionPlanRow | None:
    return session.scalars(
        select(VisualCompositionPlanRow)
        .where(
            VisualCompositionPlanRow.clip_candidate_id == candidate_id,
            VisualCompositionPlanRow.input_fingerprint == "",
        )
        .order_by(VisualCompositionPlanRow.created_at.desc())
    ).first()


def _active_job(session: Session, row: VisualCompositionPlanRow) -> ProcessingJob | None:
    jobs = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.visual_composition_plan_id == row.id,
            ProcessingJob.kind == JobKind.VISUAL_COMPOSITION,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).all()
    return jobs[0] if jobs else None


def _active_outcome(row: VisualCompositionPlanRow, job: ProcessingJob) -> CompositionQueueOutcome:
    return CompositionQueueOutcome(
        plan_id=row.id,
        job_id=job.id,
        status=row.execution_status.value,
        queued=False,
        cached=False,
        active=True,
    )


def _claim_row(session: Session, row_id: uuid.UUID, job_id: uuid.UUID) -> bool:
    result = session.execute(
        update(VisualCompositionPlanRow)
        .where(
            VisualCompositionPlanRow.id == row_id,
            VisualCompositionPlanRow.active_job_id.is_(None),
        )
        .values(active_job_id=job_id)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def _clear_stale_claim(session: Session, row: VisualCompositionPlanRow) -> None:
    stale = row.active_job_id
    if stale is None:
        return
    session.execute(
        update(VisualCompositionPlanRow)
        .where(
            VisualCompositionPlanRow.id == row.id,
            VisualCompositionPlanRow.active_job_id == stale,
        )
        .values(active_job_id=None)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    session.refresh(row)


def _matches_cache(
    session: Session,
    candidate: ClipCandidate,
    config: Stage51Config,
) -> bool:
    view: VisualCompositionView | None = read_visual_composition(
        session, candidate.id, display_probe=_NoProbe(), config=config
    )
    if view is None:
        return False
    return bool(view.effective) and bool(view.row.cache_eligible)


def queue_visual_composition(
    session: Session,
    candidate: ClipCandidate,
    *,
    force: bool = False,
    settings: Settings | None = None,
) -> CompositionQueueOutcome:
    """Queue (or reuse) one deterministic Stage 5.1 visual-composition run."""

    resolved = settings or get_settings()
    config = resolved.stage51_config()
    row = get_or_create_plan_row(session, candidate)
    for _attempt in range(2):
        active = _active_job(session, row)
        if active is not None:
            return _active_outcome(row, active)
        if row.active_job_id is not None:
            _clear_stale_claim(session, row)
            continue
        if not force and _matches_cache(session, candidate, config):
            return CompositionQueueOutcome(
                plan_id=row.id,
                job_id=None,
                status=row.execution_status.value,
                queued=False,
                cached=True,
                active=False,
            )
        job = ProcessingJob(
            source_video_id=candidate.source_video_id,
            kind=JobKind.VISUAL_COMPOSITION,
            visual_composition_plan_id=row.id,
        )
        session.add(job)
        session.flush()
        if _claim_row(session, row.id, job.id):
            session.commit()
            session.refresh(job)
            _dispatch(row.id, job.id, force)
            return CompositionQueueOutcome(
                plan_id=row.id,
                job_id=job.id,
                status=row.execution_status.value,
                queued=True,
                cached=False,
                active=False,
            )
        session.rollback()
        refreshed = session.get(VisualCompositionPlanRow, row.id)
        if refreshed is None:
            raise CompositionQueueError("visual-composition plan row vanished during queueing")
        row = refreshed
    active = _active_job(session, row)
    if active is not None:
        return _active_outcome(row, active)
    raise CompositionQueueError("could not claim a Stage 5.1 job; retry the request")


def get_plan_row(session: Session, plan_id: uuid.UUID | str) -> VisualCompositionPlanRow | None:
    return session.get(VisualCompositionPlanRow, _as_uuid(plan_id))


def _dispatch(plan_id: uuid.UUID, job_id: uuid.UUID, force: bool) -> None:
    from app.workers.tasks import run_visual_composition

    run_visual_composition.delay(str(plan_id), str(job_id), force)


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as error:
        raise CompositionQueueError("invalid identifier") from error


__all__ = [
    "CompositionQueueError",
    "CompositionQueueOutcome",
    "get_or_create_plan_row",
    "get_plan_row",
    "queue_visual_composition",
    "validate_candidate_for_composition",
]
