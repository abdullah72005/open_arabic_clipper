"""Stage 6 materialization queueing, prerequisite validation, idempotency.

Prerequisites: a current retained candidate, a current live-effective selected
Stage 4 plan, a current live-effective executable Stage 5.0 render contract
bound to that selection/plan/FINAL_CLIP, and a current ready Stage 5.1
composition bound to the contract. Queueing extends the existing Celery /
``ProcessingJob`` system with the ``MATERIALIZATION`` job kind; it adds no
``PipelineStage``, no ``PipelineRun``, and no ``_NEXT_STAGE`` entry, and never
touches the source lifecycle.

The durable *envelope* (``clip_materializations`` row) is created before any
provider work; at most one active Stage 6 job owns it via an atomic compare-and-
swap on ``clip_materializations.active_job_id``. The first request for a given
input generation locks the candidate, assembles inputs, finds-or-creates the
envelope, and either reuses the active job (recovering a failed dispatch) or
creates one new job. Concurrent requests therefore share one scoped envelope and
one job.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import JobKind, JobStatus, MaterializationLifecycle
from app.core.settings import Settings
from app.materialization.repository import (
    bind_active_job,
    get_or_create_envelope,
)
from app.materialization.service import (
    MaterializationPrerequisites,
    assemble_inputs,
    resolve_prerequisites,
)
from app.models import ClipCandidate, ClipMaterialization, ProcessingJob

_ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}
DISPATCH_FAILED_MARKER = "MATERIALIZATION_DISPATCH_FAILED"


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
    materialization_id: uuid.UUID | None = None
    render_execution_id: uuid.UUID | None = None


def validate_candidate_for_materialization(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    profile_key: str | None = None,
) -> MaterializationPrerequisites:
    """Validate every current, live-effective Stage 6 prerequisite."""

    return resolve_prerequisites(session, candidate_id, profile_key=profile_key)


def _lock_candidate(session: Session, candidate_id: uuid.UUID | str) -> ClipCandidate | None:
    try:
        parsed = (
            candidate_id if isinstance(candidate_id, uuid.UUID) else uuid.UUID(str(candidate_id))
        )
    except (TypeError, ValueError):
        return None
    return session.get(ClipCandidate, parsed, with_for_update=True)


def _active_job(session: Session, materialization_id: uuid.UUID) -> ProcessingJob | None:
    jobs = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.materialization_id == materialization_id,
            ProcessingJob.kind == JobKind.MATERIALIZATION,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).all()
    return jobs[0] if jobs else None


def _is_cached(row: ClipMaterialization) -> bool:
    return bool(row.cache_eligible) and row.lifecycle is MaterializationLifecycle.COMPLETE


def _outcome(
    job: ProcessingJob | None,
    envelope: ClipMaterialization,
    *,
    queued: bool,
    cached: bool = False,
    active: bool = False,
    skipped_reason: str | None = None,
) -> MaterializationQueueOutcome:
    return MaterializationQueueOutcome(
        job_id=job.id if job is not None else None,
        status=job.status.value if job is not None else envelope.lifecycle.value,
        queued=queued,
        cached=cached,
        active=active,
        skipped_reason=skipped_reason,
        materialization_id=envelope.id,
        render_execution_id=envelope.render_execution_id,
    )


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

    candidate = _lock_candidate(session, candidate_id)
    if candidate is None:
        raise MaterializationQueueError("candidate does not exist")

    prereq = validate_candidate_for_materialization(session, candidate_id, profile_key=profile_key)
    inputs = assemble_inputs(
        session,
        candidate_id,
        priority=priority,
        profile_key=profile_key,
        supporting_requested=supporting_requested,
    )
    request_identity = {
        "priority": priority,
        "profile_key": inputs.profile.profile_key,
        "supporting_requested": supporting_requested,
    }
    envelope = get_or_create_envelope(session, inputs, request_identity)

    active = _active_job(session, envelope.id)
    if active is not None:
        session.commit()
        _recover_or_dispatch(
            session,
            candidate_id,
            active,
            priority=priority,
            profile_key=profile_key,
            supporting_requested=supporting_requested,
            force=force,
        )
        return _outcome(active, envelope, queued=False, active=True)

    if not force and _is_cached(envelope):
        session.commit()
        return _outcome(None, envelope, queued=False, cached=True)

    # A completed-enough pre-render state is awaiting its render/finalization;
    # the post-render callback owns the next job (finalization). Re-requesting a
    # dispatch here is recovery, handled in Task 5; for now surface it truthfully.
    if envelope.lifecycle is MaterializationLifecycle.AWAITING_RENDER:
        session.commit()
        return _outcome(None, envelope, queued=False, skipped_reason="AWAITING_FINALIZATION")

    # Clear a stale binding left by a terminal job before CAS-ing a new one.
    if envelope.active_job_id is not None:
        envelope.active_job_id = None
        session.flush()

    job = ProcessingJob(
        source_video_id=prereq.source.id,
        kind=JobKind.MATERIALIZATION,
        candidate_refinement_id=prereq.final_clip.id,
        materialization_id=envelope.id,
    )
    session.add(job)
    session.flush()

    if not bind_active_job(session, envelope.id, job.id):
        # A concurrent request already bound a job: reuse the shared generation.
        session.rollback()
        shared_envelope = session.get(ClipMaterialization, envelope.id)
        assert shared_envelope is not None
        shared_active = _active_job(session, shared_envelope.id)
        return _outcome(
            shared_active,
            shared_envelope,
            queued=False,
            active=shared_active is not None,
        )

    payload = dict(active_envelope_payload(envelope))
    payload.update(request_identity)
    envelope.request_payload = payload
    session.commit()
    session.refresh(job)
    session.refresh(envelope)

    _dispatch_or_mark(
        session,
        candidate_id,
        job,
        envelope,
        priority=priority,
        profile_key=profile_key,
        supporting_requested=supporting_requested,
        force=force,
    )
    return _outcome(job, envelope, queued=True)


def active_envelope_payload(envelope: ClipMaterialization) -> dict[str, object]:
    return dict(envelope.request_payload or {})


def _recover_or_dispatch(
    session: Session,
    candidate_id: uuid.UUID | str,
    job: ProcessingJob,
    *,
    priority: str,
    profile_key: str | None,
    supporting_requested: str,
    force: bool,
) -> None:
    """Redispatch a pending job whose broker dispatch previously failed.

    Only a QUEUED job carrying the dispatch-failed marker is redispatched with
    its original request. A RUNNING job is left alone; a QUEUED job without the
    marker was already delivered, so the request merely returns its identity.
    Duplicate redispatches (crash after success but before clearing the marker)
    are harmless because the worker claim is atomic.
    """

    if job.status is not JobStatus.QUEUED:
        return
    if job.error_code != DISPATCH_FAILED_MARKER:
        return
    _dispatch_or_mark(
        session,
        candidate_id,
        job,
        session.get(ClipMaterialization, job.materialization_id) or _empty_envelope(job),
        priority=priority,
        profile_key=profile_key,
        supporting_requested=supporting_requested,
        force=force,
    )


def _empty_envelope(job: ProcessingJob) -> ClipMaterialization:
    raise MaterializationQueueError("materialization envelope is missing for an active job")


def _dispatch_or_mark(
    session: Session,
    candidate_id: uuid.UUID | str,
    job: ProcessingJob,
    envelope: ClipMaterialization,
    *,
    priority: str,
    profile_key: str | None,
    supporting_requested: str,
    force: bool,
) -> None:
    try:
        _dispatch(
            str(candidate_id),
            str(job.id),
            priority=priority,
            profile_key=profile_key,
            supporting_requested=supporting_requested,
            force=force,
        )
    except Exception:
        _mark_dispatch_failed(session, job, envelope)
        return
    _clear_dispatch_marker(session, job)


def _mark_dispatch_failed(
    session: Session, job: ProcessingJob, envelope: ClipMaterialization
) -> None:
    job.error_code = DISPATCH_FAILED_MARKER
    reasons = list(envelope.reason_codes or [])
    if DISPATCH_FAILED_MARKER not in reasons:
        reasons.append(DISPATCH_FAILED_MARKER)
    envelope.reason_codes = reasons
    try:
        session.commit()
    except Exception:
        session.rollback()


def _clear_dispatch_marker(session: Session, job: ProcessingJob) -> None:
    if job.error_code == DISPATCH_FAILED_MARKER:
        job.error_code = None
        try:
            session.commit()
        except Exception:
            session.rollback()


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
    "DISPATCH_FAILED_MARKER",
    "MaterializationQueueError",
    "MaterializationQueueOutcome",
    "get_job",
    "queue_materialization",
    "validate_candidate_for_materialization",
]
