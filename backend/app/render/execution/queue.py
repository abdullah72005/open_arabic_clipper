"""Stage 5.2 render-execution queueing and idempotency.

Prerequisites: a current retained candidate, a current live-effective executable
Stage 5.0 render contract, and a current ready Stage 5.1 visual-composition
plan. The request is validated and frozen at queue time; concurrent identical
requests converge through a scoped partial unique index and an atomic
``active_job_id`` compare-and-swap.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import JobKind, JobStatus, RenderExecutionLifecycle
from app.core.settings import Settings, get_settings
from app.models import ClipCandidate, ProcessingJob
from app.models.render_execution import RenderExecution
from app.render.execution.policy import (
    CORE_SOURCE_VALIDATION,
    DEFAULT_DELIVERY_PROFILE_KEY,
    SUPPORTED_ARTIFACT_PURPOSES,
    delivery_profile_for,
)
from app.render.execution.service import (
    RenderExecutionError,
    build_render_spec,
    get_current_render_execution,
    persist_envelope,
    request_input_fingerprint,
    validate_candidate_for_render,
    with_execution_id,
)
from app.services.storage import StorageService, StorageValidationError

_ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}


class RenderQueueError(ValueError):
    """A Stage 5.2 queue request was stale, rejected, or missing a prerequisite."""


@dataclass(frozen=True)
class RenderQueueOutcome:
    render_execution_id: uuid.UUID
    job_id: uuid.UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool
    artifact_purpose: str = CORE_SOURCE_VALIDATION
    delivery_profile_key: str = DEFAULT_DELIVERY_PROFILE_KEY


def queue_render_execution(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    delivery_profile_key: str = DEFAULT_DELIVERY_PROFILE_KEY,
    artifact_purpose: str = CORE_SOURCE_VALIDATION,
    force: bool = False,
    settings: Settings | None = None,
    dispatcher: object | None = None,
) -> RenderQueueOutcome:
    resolved = settings or get_settings()
    if not resolved.render_execution_enabled:
        raise RenderQueueError("RENDER_DISABLED")
    if artifact_purpose not in SUPPORTED_ARTIFACT_PURPOSES:
        raise RenderQueueError("UNSUPPORTED_ARTIFACT_PURPOSE")
    if delivery_profile_for(delivery_profile_key) is None:
        raise RenderQueueError("UNSUPPORTED_DELIVERY_PROFILE")
    try:
        prerequisites = validate_candidate_for_render(session, candidate_id, settings=resolved)
    except RenderExecutionError as error:
        raise RenderQueueError(error.reason_code) from error
    candidate = prerequisites.candidate
    storage = StorageService(resolved.storage_root)
    try:
        spec = build_render_spec(
            session,
            candidate,
            artifact_purpose=artifact_purpose,
            delivery_profile_key=delivery_profile_key,
            settings=resolved,
            storage=storage,
        )
    except RenderExecutionError as error:
        raise RenderQueueError(error.reason_code) from error
    except (StorageValidationError, OSError):
        raise RenderQueueError("SOURCE_MEDIA_UNMANAGED") from None
    fingerprint = request_input_fingerprint(spec, resolved.stage52_config())

    row = _get_or_create_envelope(
        session,
        candidate,
        spec=spec,
        input_fingerprint=fingerprint,
        delivery_profile_key=delivery_profile_key,
    )
    for _attempt in range(2):
        active = _active_job(session, row)
        if active is not None:
            return _active_outcome(row, active)
        if row.active_job_id is not None:
            _clear_stale_claim(session, row)
            continue
        if not force and row.lifecycle is RenderExecutionLifecycle.COMPLETE and row.cache_eligible:
            return RenderQueueOutcome(
                render_execution_id=row.id,
                job_id=None,
                status=row.lifecycle.value,
                queued=False,
                cached=True,
                active=False,
                delivery_profile_key=delivery_profile_key,
            )
        job = ProcessingJob(
            source_video_id=candidate.source_video_id,
            kind=JobKind.RENDER_EXECUTION,
            render_execution_id=row.id,
        )
        session.add(job)
        session.flush()
        if _claim_row(session, row.id, job.id):
            session.commit()
            session.refresh(job)
            _dispatch(row.id, job.id, force, dispatcher)
            return RenderQueueOutcome(
                render_execution_id=row.id,
                job_id=job.id,
                status=row.lifecycle.value,
                queued=True,
                cached=False,
                active=False,
                delivery_profile_key=delivery_profile_key,
            )
        session.rollback()
        refreshed = session.get(RenderExecution, row.id)
        if refreshed is None:
            raise RenderQueueError("render execution vanished during queueing")
        row = refreshed
    active = _active_job(session, row)
    if active is not None:
        return _active_outcome(row, active)
    raise RenderQueueError("could not claim a render job; retry the request")


def _get_or_create_envelope(
    session: Session,
    candidate: ClipCandidate,
    *,
    spec: object,
    input_fingerprint: str,
    delivery_profile_key: str,
) -> RenderExecution:
    existing = session.scalars(
        select(RenderExecution)
        .where(RenderExecution.clip_candidate_id == candidate.id)
        .where(RenderExecution.input_fingerprint == input_fingerprint)
    ).first()
    if existing is not None:
        return existing
    current = get_current_render_execution(
        session, candidate.id, delivery_profile_key=delivery_profile_key
    )
    if current is not None:
        current.is_current = False
        session.flush()
    from app.render.execution.types import RenderSpec

    assert isinstance(spec, RenderSpec)
    try:
        with session.begin_nested():
            row = persist_envelope(
                session,
                candidate,
                spec=spec,
                input_fingerprint=input_fingerprint,
                delivery_profile_key=delivery_profile_key,
            )
            session.flush()
    except IntegrityError:
        concurrent = session.scalars(
            select(RenderExecution)
            .where(RenderExecution.clip_candidate_id == candidate.id)
            .where(RenderExecution.input_fingerprint == input_fingerprint)
        ).first()
        if concurrent is None:
            raise
        return concurrent
    session.commit()
    session.refresh(row)
    return row


def _active_job(session: Session, row: RenderExecution) -> ProcessingJob | None:
    jobs = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.render_execution_id == row.id,
            ProcessingJob.kind == JobKind.RENDER_EXECUTION,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).all()
    return jobs[0] if jobs else None


def _active_outcome(row: RenderExecution, job: ProcessingJob) -> RenderQueueOutcome:
    return RenderQueueOutcome(
        render_execution_id=row.id,
        job_id=job.id,
        status=row.lifecycle.value,
        queued=False,
        cached=False,
        active=True,
        delivery_profile_key=row.delivery_profile_key,
    )


def _claim_row(session: Session, row_id: uuid.UUID, job_id: uuid.UUID) -> bool:
    result = session.execute(
        update(RenderExecution)
        .where(RenderExecution.id == row_id, RenderExecution.active_job_id.is_(None))
        .values(active_job_id=job_id)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def _clear_stale_claim(session: Session, row: RenderExecution) -> None:
    stale = row.active_job_id
    if stale is None:
        return
    session.execute(
        update(RenderExecution)
        .where(RenderExecution.id == row.id, RenderExecution.active_job_id == stale)
        .values(active_job_id=None)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    session.refresh(row)


def get_render_execution_row(
    session: Session, render_execution_id: uuid.UUID | str
) -> RenderExecution | None:
    try:
        parsed = (
            render_execution_id
            if isinstance(render_execution_id, uuid.UUID)
            else uuid.UUID(str(render_execution_id))
        )
    except (TypeError, ValueError):
        return None
    return session.get(RenderExecution, parsed)


def _dispatch(
    render_execution_id: uuid.UUID,
    job_id: uuid.UUID,
    force: bool,
    dispatcher: object | None,
) -> None:
    if dispatcher is not None:
        dispatch = getattr(dispatcher, "dispatch", None)
        if callable(dispatch):
            dispatch(render_execution_id, job_id, force)
            return
        if callable(dispatcher):
            dispatcher(render_execution_id, job_id, force)
            return
    from app.workers.tasks import run_render_execution

    run_render_execution.delay(str(render_execution_id), str(job_id), force)


__all__ = [
    "RenderQueueError",
    "RenderQueueOutcome",
    "get_render_execution_row",
    "queue_render_execution",
    "with_execution_id",
]
