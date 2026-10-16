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


#: Completed-cache verification dispositions. A COMPLETE+cache_eligible envelope
#: is only a cache *candidate*: it must still verify its exact bound render,
#: binding, live scope, artifact integrity, and committed final assessment before
#: it may be claimed as cached.
CACHE_VALID = "VALID"
CACHE_RECOVERABLE = "RECOVERABLE"
#: The exact completed render remains valid but the finalizer's committed
#: assessment is missing/invalid: recoverable by one finalizer over that render.
CACHE_ASSESSMENT_RECOVERABLE = "ASSESSMENT_RECOVERABLE"
CACHE_TERMINAL = "TERMINAL"

#: Shared-readiness reasons that mean the completed render itself is valid and only
#: the finalizer's committed assessment is missing/invalid. A completed cache hit
#: carrying only these reasons is recoverable by one finalizer over the exact
#: render (reusing accepted components: no provider repeat, no unnecessary render).
_ASSESSMENT_RECOVERY_REASONS = frozenset(
    {
        "MISSING_FINAL_ASSESSMENT",
        "ASSESSMENT_POLICY_MISMATCH",
        "ASSESSMENT_RENDER_BINDING_MISMATCH",
        "ASSESSMENT_FINGERPRINT_MISMATCH",
        "HUMAN_REVIEW_NOT_REQUIRED",
        "HUMAN_REVIEW_FLAG_CLEARED",
    }
)


def _bound_completed_render(
    session: Session, envelope: ClipMaterialization
) -> RenderExecution | None:
    """Return the exact effective bound transformed render, or ``None``.

    The bound render must be the envelope's exact reverse-bound revision, a
    ``TRANSFORMED_SHORT`` artifact, ``COMPLETE``, current, and its own
    cache-eligible flag set. Any deviation returns ``None`` (never a substitute
    row).
    """

    from app.render.execution.policy import TRANSFORMED_SHORT

    if envelope.render_execution_id is None:
        return None
    render = session.get(RenderExecution, envelope.render_execution_id, populate_existing=True)
    if render is None:
        return None
    if render.materialization_id != envelope.id:
        return None
    if render.artifact_purpose.value != TRANSFORMED_SHORT:
        return None
    if render.lifecycle.value != "COMPLETE":
        return None
    if not render.is_current or not render.cache_eligible:
        return None
    return render


def _completed_cache_state(
    session: Session, envelope: ClipMaterialization, *, settings: Settings
) -> str:
    """Classify a COMPLETE envelope as a verified cache, recoverable, or terminal.

    ``CACHE_VALID`` requires the exact effective bound render, the complete
    generation/output/repair binding identity, a currently-fresh render scope, an
    artifact that actually exists with the expected digest, AND the same
    committed-assessment/live-input requirements the shared readiness view
    enforces. ``CACHE_ASSESSMENT_RECOVERABLE`` means the completed render is itself
    valid but the committed final assessment is missing/invalid: one finalizer over
    that exact render can restore it. ``CACHE_RECOVERABLE`` means the completed
    result is no longer a trustworthy cache claim and must be revalidated/
    technically recovered. ``CACHE_TERMINAL`` means the envelope is not a cache
    candidate at all and the normal COMPLETE skip applies.
    """

    if not (envelope.cache_eligible and envelope.lifecycle is MaterializationLifecycle.COMPLETE):
        return CACHE_TERMINAL
    render = _bound_completed_render(session, envelope)
    if render is None:
        return CACHE_RECOVERABLE
    from app.materialization.finalization import _artifact_identity, _validate_render_binding

    if _validate_render_binding(envelope, render) is not None:
        return CACHE_RECOVERABLE
    from app.render.execution.service import _live_request_freshness

    if _live_request_freshness(session, render, settings) != "CURRENT":
        return CACHE_RECOVERABLE
    from app.services.storage import StorageService

    storage = StorageService(settings.storage_root)
    if isinstance(_artifact_identity(storage, render), str):
        return CACHE_RECOVERABLE

    # A completed cache hit must satisfy the SAME committed assessment, live-input,
    # render-binding, QC, and artifact requirements as the shared readiness view.
    # Missing or mismatched assessment evidence is never a verified cache hit. The
    # authoritative read view is reused rather than a second permissive
    # approximation, so ordinary POST and resume converge on one verdict.
    from app.materialization.readiness import evaluate_readiness

    view = evaluate_readiness(session, storage, envelope)
    if view.review_ready:
        return CACHE_VALID
    if view.reasons and all(reason in _ASSESSMENT_RECOVERY_REASONS for reason in view.reasons):
        # The exact completed render is valid; only the committed final assessment
        # is missing/invalid. Recover one finalizer using that render.
        return CACHE_ASSESSMENT_RECOVERABLE
    return CACHE_RECOVERABLE


def _recover_completed_generation(
    session: Session,
    candidate_id: uuid.UUID | str,
    envelope: ClipMaterialization,
    *,
    settings: Settings,
    dispatcher: object | None,
    keep_render: bool,
) -> MaterializationQueueOutcome:
    """Recover a completed generation through the authoritative durable router.

    One transition is shared by normal POST and id-based resume: an AWAITING_RENDER
    (pending-finalization) generation never carries a stale review requirement or a
    stale reviewable semantic outcome; only a verified current final assessment may
    restore review readiness. Historical assessments, artifacts, accepted
    components, budgets, and repair usage remain preserved.

    ``keep_render`` preserves the exact bound render (a valid completed render whose
    assessment is missing is finalized directly); otherwise the forward binding is
    cleared so the handoff boundary reactivates an equivalent current revision or
    queues a bounded technical re-render (never a semantic regeneration).
    """

    envelope.lifecycle = MaterializationLifecycle.AWAITING_RENDER
    if not keep_render:
        envelope.render_execution_id = None
    envelope.cache_eligible = False
    envelope.human_review_required = False
    envelope.publication_ready = False
    envelope.semantic_outcome = None
    session.flush()
    return _route_awaiting_render(
        session, candidate_id, envelope, settings=settings, dispatcher=dispatcher
    )


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

    if not force:
        cache_state = _completed_cache_state(session, envelope, settings=resolved_settings)
        if cache_state == CACHE_VALID:
            session.commit()
            return _outcome(None, envelope, queued=False, cached=True)
        if cache_state == CACHE_RECOVERABLE:
            # A completed result whose bound render, binding, scope, or artifact no
            # longer verifies is never returned as cached: the accepted semantic
            # components, provider budgets, repair usage, and historical artifacts
            # are preserved, and the exact transform is technically recovered
            # through the render reactivation/re-handoff boundary (never a semantic
            # regeneration). Clearing the forward binding forces the handoff
            # boundary to reactivate an equivalent current revision or queue a
            # bounded technical re-render, rather than finalizing an unverifiable
            # render directly.
            return _recover_completed_generation(
                session,
                candidate_id,
                envelope,
                settings=resolved_settings,
                dispatcher=render_dispatcher,
                keep_render=False,
            )
        if cache_state == CACHE_ASSESSMENT_RECOVERABLE:
            # The exact completed render remains valid; only the finalizer's
            # committed assessment is missing/invalid. Recover one finalizer over
            # that render, reusing accepted components (no provider repeat and no
            # unnecessary re-render).
            return _recover_completed_generation(
                session,
                candidate_id,
                envelope,
                settings=resolved_settings,
                dispatcher=render_dispatcher,
                keep_render=True,
            )

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
    the truthful outcome. A historical/superseded but otherwise exact bound render
    is recovered through the handoff boundary (reactivating an equivalent render or
    queueing a technical re-render), never finalized directly.
    """

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
                return _request_bound_finalizer(
                    session, envelope, render, settings=settings, dispatcher=dispatcher
                )
            # A terminal bound render that is no longer current/effective is
            # recovered through the exact handoff below rather than being treated
            # as finalizable: reactivating an equivalent historical render (same
            # generation, verified artifact) or queueing a technical re-render.

    return _recover_render_handoff(
        session, candidate_id, envelope, settings=settings, dispatcher=dispatcher
    )


def _request_bound_finalizer(
    session: Session,
    envelope: ClipMaterialization,
    render: RenderExecution,
    *,
    settings: Settings,
    dispatcher: object | None,
) -> MaterializationQueueOutcome:
    """Queue/recover one finalizer for an exact terminal current bound render."""

    from app.materialization.finalization import request_finalization_job

    job_id = request_finalization_job(session, render.id, settings=settings, dispatcher=dispatcher)
    session.commit()
    job = session.get(ProcessingJob, job_id) if job_id is not None else None
    if job is None:
        return _outcome(None, envelope, queued=False, skipped_reason="FINALIZATION_UNAVAILABLE")
    return _outcome(job, envelope, queued=False, active=True, phase=PHASE_FINALIZE)


def _recover_render_handoff(
    session: Session,
    candidate_id: uuid.UUID | str,
    envelope: ClipMaterialization,
    *,
    settings: Settings,
    dispatcher: object | None,
) -> MaterializationQueueOutcome:
    """Recover the exact transformed-render handoff through the render boundary.

    The accepted components and output fingerprint are preserved; the existing
    render queue reactivates an equivalent historical revision when its artifact
    still verifies, or queues a bounded technical re-render otherwise. A recovered
    terminal current render is then handed to the finalizer.
    """

    from app.materialization.finalization import dispatch_transformed_render
    from app.services.storage import StorageService

    try:
        render_id = dispatch_transformed_render(
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
    if render_id is None:
        return _outcome(
            None, envelope, queued=False, skipped_reason="RENDER_HANDOFF_RECOVERY_FAILED"
        )
    render = session.get(RenderExecution, render_id, populate_existing=True)
    if render is None:
        return _outcome(
            None, envelope, queued=False, skipped_reason="RENDER_HANDOFF_RECOVERY_FAILED"
        )
    # A freshly queued technical re-render owns the work: never finalize a still
    # active render. Only a recovered terminal current render is handed to the
    # finalizer.
    if _active_render_job(session, render.id) is not None:
        return _outcome(
            None, envelope, queued=False, active=True, skipped_reason="RENDER_HANDOFF_RECOVERED"
        )
    if render.is_current and render.lifecycle.value in _RENDER_FINALIZABLE:
        return _request_bound_finalizer(
            session, envelope, render, settings=settings, dispatcher=dispatcher
        )
    return _outcome(
        None, envelope, queued=False, active=True, skipped_reason="RENDER_HANDOFF_RECOVERED"
    )


def _active_render_job(session: Session, render_id: uuid.UUID) -> ProcessingJob | None:
    """Return an active RENDER_EXECUTION job for this render, or ``None``."""

    return session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.render_execution_id == render_id,
            ProcessingJob.kind == JobKind.RENDER_EXECUTION,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).first()


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


def cancel_materialization(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    profile_key: str | None = None,
) -> uuid.UUID:
    """Cooperatively cancel the current Stage 6 generation for a candidate.

    Resolves the requested/default profile exactly as POST does and delegates the
    authoritative cancellation to :func:`cancel_materialization_by_id`, so the
    candidate-scoped and id-scoped entry points share one implementation.
    """

    candidate = _lock_candidate(session, candidate_id)
    if candidate is None:
        raise MaterializationQueueError("candidate does not exist")
    from app.materialization.profiles import profile_from_row
    from app.materialization.service import _current_profile

    resolved_profile_key = profile_from_row(_current_profile(session, profile_key)).profile_key
    envelope = session.scalars(
        select(ClipMaterialization)
        .where(
            ClipMaterialization.clip_candidate_id == candidate.id,
            ClipMaterialization.profile_key == resolved_profile_key,
            ClipMaterialization.is_current.is_(True),
        )
        .order_by(ClipMaterialization.created_at.desc())
    ).first()
    if envelope is None:
        raise MaterializationQueueError("materialization does not exist")
    cancelled = _cancel_envelope(session, envelope.id)
    return uuid.UUID(str(cancelled.id))


def _load_envelope_by_id(
    session: Session, materialization_id: uuid.UUID | str
) -> ClipMaterialization:
    try:
        parsed = (
            materialization_id
            if isinstance(materialization_id, uuid.UUID)
            else uuid.UUID(str(materialization_id))
        )
    except (TypeError, ValueError):
        raise MaterializationQueueError("materialization does not exist") from None
    envelope = session.get(ClipMaterialization, parsed, populate_existing=True)
    if envelope is None:
        raise MaterializationQueueError("materialization does not exist")
    return envelope


def cancel_materialization_by_id(
    session: Session, materialization_id: uuid.UUID | str
) -> ClipMaterialization:
    """Cooperatively cancel one *exact* materialization id (generation authority).

    Under the candidate-first lock order, operates on the requested envelope and
    requires it to be the exact current generation; a historical/stale id is
    refused with a truthful reason and makes no mutation or delivery. A repeated
    cancellation of the same cancelled current row is idempotent.
    """

    loaded = _load_envelope_by_id(session, materialization_id)
    candidate = _lock_candidate(session, loaded.clip_candidate_id)
    if candidate is None:
        raise MaterializationQueueError("candidate does not exist")
    # Re-read under the lock: the operation must target the requested envelope,
    # never resolve a historical id into a newer current envelope.
    envelope = session.get(ClipMaterialization, loaded.id, populate_existing=True)
    if envelope is None:
        raise MaterializationQueueError("materialization does not exist")
    return _cancel_envelope(session, envelope.id)


def _cancel_envelope(session: Session, materialization_id: uuid.UUID) -> ClipMaterialization:
    """Cancel one exact current envelope; idempotent for an already-cancelled row."""

    envelope = session.get(
        ClipMaterialization, materialization_id, populate_existing=True, with_for_update=True
    )
    if envelope is None:
        raise MaterializationQueueError("materialization does not exist")
    if not envelope.is_current:
        raise MaterializationQueueError("MATERIALIZATION_NOT_CURRENT")
    if envelope.lifecycle is MaterializationLifecycle.CANCELLED:
        # Idempotent: the same cancelled current generation is already terminal.
        session.commit()
        return envelope

    for job in session.scalars(
        select(ProcessingJob).where(
            ProcessingJob.materialization_id == envelope.id,
            ProcessingJob.kind == JobKind.MATERIALIZATION,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
    ).all():
        job.status = JobStatus.CANCELLED

    if envelope.render_execution_id is not None:
        render = session.get(
            RenderExecution, envelope.render_execution_id, populate_existing=True
        )
        if render is not None:
            render_job = session.scalars(
                select(ProcessingJob).where(
                    ProcessingJob.render_execution_id == render.id,
                    ProcessingJob.kind == JobKind.RENDER_EXECUTION,
                    ProcessingJob.status.in_(_ACTIVE_STATUSES),
                )
            ).first()
            if render_job is not None:
                render_job.status = JobStatus.CANCELLED
                session.flush()
                from app.render.execution.service import cancel_execution

                try:
                    cancel_execution(session, render.id, job_id=render_job.id)
                except Exception:
                    session.rollback()
                    raise

    envelope.lifecycle = MaterializationLifecycle.CANCELLED
    envelope.active_job_id = None
    envelope.human_review_required = False
    envelope.publication_ready = False
    envelope.cache_eligible = False
    session.commit()
    return envelope


def resume_materialization_by_id(
    session: Session,
    materialization_id: uuid.UUID | str,
    *,
    settings: Settings | None = None,
    dispatcher: object | None = None,
) -> MaterializationQueueOutcome:
    """Resume one *exact* materialization id using its persisted options.

    The requested envelope's exact persisted options and matching checkpoints are
    used; a historical/stale id is refused. A verified completed result returns the
    verified cache. An explicit resume of a cancelled current generation creates a
    new fenced job while preserving accepted components, budgets, and repair usage;
    old jobs stay terminal and no uncertain provider attempt is spent.
    """

    resolved_settings = settings or get_settings()
    loaded = _load_envelope_by_id(session, materialization_id)
    candidate = _lock_candidate(session, loaded.clip_candidate_id)
    if candidate is None:
        raise MaterializationQueueError("candidate does not exist")
    envelope = session.get(ClipMaterialization, loaded.id, populate_existing=True)
    if envelope is None:
        raise MaterializationQueueError("materialization does not exist")
    if not envelope.is_current:
        raise MaterializationQueueError("MATERIALIZATION_NOT_CURRENT")

    # Reconstruct the exact persisted request options and revalidate the live
    # prerequisites plus the recomputed input fingerprint BEFORE any active
    # recovery, cache acceptance, job creation, binding change, or dispatch.
    # ``is_current`` alone is insufficient: upstream mutation (a superseded
    # FINAL_CLIP, selected plan, render contract, or composition) can leave this
    # row current while its persisted generation no longer matches live inputs,
    # and a cancelled generation must never be resumed into stale work. A genuine
    # match preserves current-generation checkpoint reuse.
    from app.materialization.finalization import _revalidate_live_inputs

    if _revalidate_live_inputs(session, envelope) is None:
        raise MaterializationQueueError("UPSTREAM_REVALIDATION_REQUIRED")

    request = envelope.request_payload if isinstance(envelope.request_payload, dict) else {}
    priority = str(request.get("priority") or envelope.priority.value)
    profile_key = request.get("requested_profile_key", request.get("profile_key"))
    supporting_requested = str(request.get("supporting_requested") or "OFF")

    active = _active_job(session, envelope.id)
    if active is not None:
        _recover_active_job(
            session,
            candidate.id,
            active,
            envelope,
            settings=resolved_settings,
            dispatcher=dispatcher,
            priority=priority,
            profile_key=profile_key,
            supporting_requested=supporting_requested,
            force=True,
        )
        session.commit()
        return _outcome(
            active,
            envelope,
            queued=False,
            active=True,
            phase=resolve_job_phase(session, active, envelope),
        )

    if envelope.lifecycle is MaterializationLifecycle.COMPLETE:
        cache_state = _completed_cache_state(session, envelope, settings=resolved_settings)
        if cache_state == CACHE_VALID:
            session.commit()
            return _outcome(None, envelope, queued=False, cached=True)
        if cache_state == CACHE_RECOVERABLE:
            return _recover_completed_generation(
                session,
                candidate.id,
                envelope,
                settings=resolved_settings,
                dispatcher=dispatcher,
                keep_render=False,
            )
        if cache_state == CACHE_ASSESSMENT_RECOVERABLE:
            return _recover_completed_generation(
                session,
                candidate.id,
                envelope,
                settings=resolved_settings,
                dispatcher=dispatcher,
                keep_render=True,
            )
        session.commit()
        return _outcome(
            None,
            envelope,
            queued=False,
            skipped_reason=(
                envelope.semantic_outcome.value if envelope.semantic_outcome else "COMPLETE"
            ),
        )

    if envelope.lifecycle is MaterializationLifecycle.AWAITING_RENDER:
        return _route_awaiting_render(
            session, candidate.id, envelope, settings=resolved_settings, dispatcher=dispatcher
        )

    if envelope.lifecycle is MaterializationLifecycle.BLOCKED:
        session.commit()
        return _outcome(None, envelope, queued=False, skipped_reason="BLOCKED")

    # QUEUED/RUNNING/FAILED/CANCELLED: an explicit resume may create a new fenced
    # job for this exact envelope, preserving accepted components, budgets, repair
    # usage, and unresolved uncertainty (the gateway refuses to re-spend uncertain
    # attempts). Cancelled jobs are never revived.
    return _resume_envelope_job(
        session,
        candidate.id,
        envelope,
        settings=resolved_settings,
        dispatcher=dispatcher,
        priority=priority,
        profile_key=profile_key,
        supporting_requested=supporting_requested,
    )


def resume_materialization(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    profile_key: str | None = None,
    settings: Settings | None = None,
    dispatcher: object | None = None,
) -> MaterializationQueueOutcome:
    """Candidate-scoped compatibility wrapper over the authoritative id resume.

    Resolves the requested/default profile exactly as POST does, then delegates to
    :func:`resume_materialization_by_id` so the candidate/CLI entry points share one
    implementation and never silently create a replacement generation.
    """

    candidate = _lock_candidate(session, candidate_id)
    if candidate is None:
        raise MaterializationQueueError("candidate does not exist")
    from app.materialization.profiles import profile_from_row
    from app.materialization.service import _current_profile

    resolved_profile_key = profile_from_row(_current_profile(session, profile_key)).profile_key
    envelope = session.scalars(
        select(ClipMaterialization)
        .where(
            ClipMaterialization.clip_candidate_id == candidate.id,
            ClipMaterialization.profile_key == resolved_profile_key,
            ClipMaterialization.is_current.is_(True),
        )
        .order_by(ClipMaterialization.created_at.desc())
    ).first()
    if envelope is None:
        raise MaterializationQueueError("materialization does not exist")
    return resume_materialization_by_id(
        session, envelope.id, settings=settings, dispatcher=dispatcher
    )


def _resume_envelope_job(
    session: Session,
    candidate_id: uuid.UUID,
    envelope: ClipMaterialization,
    *,
    settings: Settings,
    dispatcher: object | None,
    priority: str,
    profile_key: object,
    supporting_requested: str,
) -> MaterializationQueueOutcome:
    """Create/recover one fenced MATERIALIZATION job for an exact envelope."""

    if envelope.active_job_id is not None:
        envelope.active_job_id = None
        session.flush()

    job = ProcessingJob(
        source_video_id=envelope.source_video_id,
        kind=JobKind.MATERIALIZATION,
        candidate_refinement_id=envelope.refinement_id,
        materialization_id=envelope.id,
    )
    session.add(job)
    session.flush()

    if not bind_active_job(session, envelope.id, job.id):
        session.rollback()
        shared = session.get(ClipMaterialization, envelope.id, populate_existing=True)
        assert shared is not None
        shared_active = _active_job(session, shared.id)
        return _outcome(
            shared_active,
            shared,
            queued=False,
            active=shared_active is not None,
        )

    if envelope.lifecycle is not MaterializationLifecycle.AWAITING_RENDER:
        envelope.lifecycle = MaterializationLifecycle.QUEUED
    payload = dict(active_envelope_payload(envelope))
    payload.setdefault("priority", priority)
    payload.setdefault("supporting_requested", supporting_requested)
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
        profile_key=profile_key if isinstance(profile_key, str) else None,
        supporting_requested=supporting_requested,
        force=True,
    )
    return _outcome(job, envelope, queued=True)



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
    "cancel_materialization",
    "cancel_materialization_by_id",
    "get_job",
    "queue_materialization",
    "resolve_job_options",
    "resolve_job_phase",
    "resume_materialization",
    "resume_materialization_by_id",
    "validate_candidate_for_materialization",
]
