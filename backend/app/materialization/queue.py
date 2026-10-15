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
from app.core.settings import Settings, get_settings
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
from app.models.render_execution import RenderExecution

_ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}
DISPATCH_FAILED_MARKER = "MATERIALIZATION_DISPATCH_FAILED"
RENDER_DISPATCH_FAILED_MARKER = "RENDER_DISPATCH_FAILED"

#: Durable job phases resolved from the job's bound envelope, never from the
#: transient Celery task name or caller arguments.
PHASE_PREPARE = "PREPARE"
PHASE_FINALIZE = "FINALIZE"

_RENDER_TERMINAL = {"COMPLETE", "FAILED", "BLOCKED", "CANCELLED"}

#: Terminal render states that have authoritative final evidence to assess (or a
#: justified repair to allocate). A CANCELLED render is never finalized.
_RENDER_FINALIZABLE = {"COMPLETE", "FAILED", "BLOCKED"}


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
    phase: str | None = None


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
    phase: str | None = None,
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
        phase=phase,
    )


def resolve_job_phase(
    session: Session, job: ProcessingJob, envelope: ClipMaterialization | None
) -> str:
    """Resolve the durable phase of one MATERIALIZATION job from its envelope.

    A job is a FINALIZE job only when its envelope is AWAITING_RENDER with a
    terminal bound render; otherwise it is PREPARE work. The Celery task name and
    transient caller arguments are never the phase identity.
    """

    if envelope is None and job.materialization_id is not None:
        envelope = session.get(
            ClipMaterialization, job.materialization_id, populate_existing=True
        )
    if envelope is None:
        return PHASE_PREPARE
    if envelope.lifecycle is not MaterializationLifecycle.AWAITING_RENDER:
        return PHASE_PREPARE
    if envelope.render_execution_id is None:
        return PHASE_PREPARE
    render = session.get(RenderExecution, envelope.render_execution_id, populate_existing=True)
    if render is None:
        return PHASE_PREPARE
    if render.lifecycle.value in _RENDER_FINALIZABLE:
        return PHASE_FINALIZE
    return PHASE_PREPARE


def resolve_job_options(
    session: Session, job: ProcessingJob, envelope: ClipMaterialization | None
) -> dict[str, object]:
    """Resolve candidate/source/request options from the job's bound envelope."""

    if envelope is None and job.materialization_id is not None:
        envelope = session.get(ClipMaterialization, job.materialization_id, populate_existing=True)
    request = (
        envelope.request_payload
        if envelope is not None and isinstance(envelope.request_payload, dict)
        else {}
    )
    return {
        "candidate_id": str(envelope.clip_candidate_id) if envelope is not None else None,
        "priority": str(request.get("priority") or "CANDIDATE"),
        "profile_key": request.get("requested_profile_key", request.get("profile_key")),
        "supporting_requested": str(request.get("supporting_requested") or "OFF"),
    }


def queue_materialization(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    priority: str = "CANDIDATE",
    profile_key: str | None = None,
    supporting_requested: str = "OFF",
    force: bool = False,
    settings: Settings | None = None,
    render_dispatcher: object | None = None,
) -> MaterializationQueueOutcome:
    """Queue (or reuse) one bounded Stage 6 materialization run for a candidate."""

    resolved_settings = settings or get_settings()
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
        "requested_profile_key": profile_key,
        "supporting_requested": supporting_requested,
    }
    envelope = get_or_create_envelope(session, inputs, request_identity)

    active = _active_job(session, envelope.id)
    if active is not None:
        _recover_active_job(
            session,
            candidate_id,
            active,
            envelope,
            settings=resolved_settings,
            dispatcher=render_dispatcher,
            priority=priority,
            profile_key=profile_key,
            supporting_requested=supporting_requested,
            force=force,
        )
        session.commit()
        return _outcome(
            active,
            envelope,
            queued=False,
            active=True,
            phase=resolve_job_phase(session, active, envelope),
        )

    if not force and _is_cached(envelope):
        session.commit()
        return _outcome(None, envelope, queued=False, cached=True)

    # One authoritative durable phase router: an AWAITING_RENDER envelope is
    # either recovering its exact render handoff, reporting its active render, or
    # queuing/recovering its exact finalizer — never blindly returning a stale
    # "AWAITING_FINALIZATION" and never re-running accepted providers.
    if envelope.lifecycle is MaterializationLifecycle.AWAITING_RENDER:
        return _route_awaiting_render(
            session,
            candidate_id,
            envelope,
            settings=resolved_settings,
            dispatcher=render_dispatcher,
        )

    # G: a cancelled/semantically-blocked/materialized-terminal generation is
    # preserved truthfully; nothing silently regenerates to force output.
    if envelope.lifecycle in {
        MaterializationLifecycle.BLOCKED,
        MaterializationLifecycle.CANCELLED,
    }:
        session.commit()
        return _outcome(None, envelope, queued=False, skipped_reason=envelope.lifecycle.value)
    if envelope.lifecycle is MaterializationLifecycle.COMPLETE and not force:
        session.commit()
        return _outcome(
            None,
            envelope,
            queued=False,
            skipped_reason=(
                envelope.semantic_outcome.value if envelope.semantic_outcome else "COMPLETE"
            ),
        )

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
    payload.setdefault("requested_profile_key", profile_key)
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


def _recover_active_job(
    session: Session,
    candidate_id: uuid.UUID | str,
    job: ProcessingJob,
    envelope: ClipMaterialization,
    *,
    settings: Settings,
    dispatcher: object | None,
    priority: str,
    profile_key: str | None,
    supporting_requested: str,
    force: bool,
) -> None:
    """Redispatch a pending job whose broker delivery previously failed.

    The durable phase is resolved from the job's bound envelope, so a queued
    FINALIZE job is redispatched to finalization and never to preparation. A
    RUNNING job is left alone; a QUEUED job without the marker was already
    delivered, so the request merely returns its identity.
    """

    if job.status is not JobStatus.QUEUED:
        return
    if job.error_code != DISPATCH_FAILED_MARKER:
        return
    if resolve_job_phase(session, job, envelope) == PHASE_FINALIZE:
        from app.materialization.finalization import _deliver_finalization_job

        _deliver_finalization_job(session, job, settings=settings, dispatcher=dispatcher)
        return
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


def _route_awaiting_render(
    session: Session,
    candidate_id: uuid.UUID | str,
    envelope: ClipMaterialization,
    *,
    settings: Settings,
    dispatcher: object | None,
) -> MaterializationQueueOutcome:
    """Resolve the durable phase of an AWAITING_RENDER envelope.

    B: no durable bound render -> recover the exact render handoff (accepted
    providers are never repeated). C: an active bound render -> return its active
    identity and recover a marked broker-delivery failure. D: an exact terminal
    bound render -> queue/recover one finalizer. G: a cancelled render -> preserve
    the truthful outcome. A historical/superseded bound render never finalizes.
    """

    from app.services.storage import StorageService

    render_id = envelope.render_execution_id
    if render_id is not None:
        render = session.get(RenderExecution, render_id, populate_existing=True)
        if render is not None and envelope.id == render.materialization_id:
            state = render.lifecycle.value
            if state not in _RENDER_TERMINAL and render.is_current:
                _recover_render_job(session, render, dispatcher=dispatcher)
                session.commit()
                return _outcome(
                    None, envelope, queued=False, active=True, skipped_reason="AWAITING_RENDER"
                )
            if state == "CANCELLED":
                session.commit()
                return _outcome(None, envelope, queued=False, skipped_reason="RENDER_CANCELLED")
            if state in _RENDER_FINALIZABLE and render.is_current:
                from app.materialization.finalization import request_finalization_job

                job_id = request_finalization_job(
                    session, render.id, settings=settings, dispatcher=dispatcher
                )
                session.commit()
                job = session.get(ProcessingJob, job_id) if job_id is not None else None
                if job is None:
                    return _outcome(
                        None, envelope, queued=False, skipped_reason="FINALIZATION_UNAVAILABLE"
                    )
                return _outcome(
                    job, envelope, queued=False, active=True, phase=PHASE_FINALIZE
                )
            if not render.is_current:
                # Historical/superseded render: require explicit revalidation.
                session.commit()
                return _outcome(
                    None, envelope, queued=False, skipped_reason="RENDER_SUPERSEDED"
                )

    # B: the envelope committed a renderable result but never durably bound a
    # render. Recover the exact handoff; accepted providers are checkpointed.
    from app.materialization.finalization import dispatch_transformed_render

    try:
        dispatch_transformed_render(
            session,
            envelope,
            candidate_id,
            settings=settings,
            storage=StorageService(settings.storage_root),
            dispatcher=dispatcher,
        )
    except Exception:
        session.rollback()
        return _outcome(
            None, envelope, queued=False, skipped_reason="RENDER_HANDOFF_RECOVERY_FAILED"
        )
    session.commit()
    return _outcome(
        None, envelope, queued=False, active=True, skipped_reason="RENDER_HANDOFF_RECOVERED"
    )


def _recover_render_job(
    session: Session, render: RenderExecution, *, dispatcher: object | None
) -> ProcessingJob | None:
    """Redispatch a bound render job whose broker delivery previously failed."""

    from app.render.execution.queue import deliver_render_job

    job = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.render_execution_id == render.id,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).first()
    if job is None:
        return None
    if job.error_code == RENDER_DISPATCH_FAILED_MARKER:
        try:
            deliver_render_job(render.id, job.id, False, dispatcher)
            job.error_code = None
        except Exception:
            job.error_code = RENDER_DISPATCH_FAILED_MARKER
        session.commit()
    return job


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
    "PHASE_FINALIZE",
    "PHASE_PREPARE",
    "RENDER_DISPATCH_FAILED_MARKER",
    "MaterializationQueueError",
    "MaterializationQueueOutcome",
    "get_job",
    "queue_materialization",
    "resolve_job_options",
    "resolve_job_phase",
    "validate_candidate_for_materialization",
]
