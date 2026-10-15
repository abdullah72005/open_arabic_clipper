"""Stage 6 render dispatch and truthful finalization.

The prepare job commits accepted components, the immutable output fingerprint,
and lifecycle ``AWAITING_RENDER``. This module binds a ``TRANSFORMED_SHORT``
render execution to that *exact* materialization id/output fingerprint through
the existing render queue, and, once the render reaches a terminal state,
finalizes the materialization by *re-assessing* the persisted semantic evidence
together with the actual render/QC evidence.

The finalizer never promotes a materialization to review readiness on the mere
completion of a technical render. It re-derives live upstream effectiveness,
rebuilds an explicit :class:`~app.materialization.acceptance.AssessmentContext`
from the versioned, dependency-bound pre-render snapshot plus the stored accepted
component/slot/claim/narration/timeline evidence and the actual render/QC
measurements, then runs the existing categorical assessment engine. A stored
``PASS`` label can never override a missing or contradictory measurement, a
completed render never erases a persisted semantic failure, and a stale/missing
snapshot is never deserialized into favorable default gates.

Every handled disposition (complete/deferred/superseded/cancelled) is committed
through one guarded transaction that finishes the exact owned job and releases
its binding where ownership permits; a lost owner never mutates another worker's
claim. Finalization is idempotent: an identical evidence replay makes zero
provider/render calls, preserves the assessment identity, and still completes
the *current* job under its exact owner fence.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import (
    JobStatus,
    MaterializationLifecycle,
    MaterializationOutcome,
    MaterializationPriority,
)
from app.core.settings import Settings
from app.materialization.acceptance import AssessmentContext, assess_materialization
from app.materialization.components import script_dependency_inputs
from app.materialization.fingerprints import input_fingerprint, stable_digest
from app.materialization.inputs import MaterializationInputs
from app.materialization.policy import (
    REASON_HERO_DAMAGED,
    REASON_INVALID_GROUNDING_REFERENCE,
    REASON_MANUAL_TRANSCRIPT_REVIEW,
    REASON_SCRIPT_AUDIO_MISMATCH,
    REASON_UNREADABLE_REQUIRED_COPY,
    REASON_UNSUITABLE_ASSET_RIGHTS,
    REASON_UNSUPPORTED_ESSENTIAL_CLAIM,
)
from app.materialization.profiles import semantic_context_matches
from app.materialization.repository import (
    PRE_RENDER_CONTEXT_METRIC,
    PRE_RENDER_CONTEXT_VERSION,
    MaterializationOwner,
    allocate_repair,
    finish_owner_job,
    guarded_update,
    owner_is_current,
)
from app.materialization.types import QualityFinding
from app.models import ClipMaterialization, RenderContract
from app.models.render_execution import RenderExecution
from app.render.execution.policy import (
    AUDIO_PEAK_REPAIR_NEEDED,
    TRANSFORMED_SHORT,
)
from app.services.storage import StorageService

FINALIZATION_COMPLETE = "COMPLETE"
FINALIZATION_REPAIR_QUEUED = "REPAIR_QUEUED"
FINALIZATION_DEFERRED = "DEFERRED"
FINALIZATION_SUPERSEDED = "SUPERSEDED"
FINALIZATION_CANCELLED = "CANCELLED"

REASON_RENDER_NOT_COMPLETE = "RENDER_NOT_COMPLETE"
REASON_RENDER_BINDING_INVALID = "RENDER_BINDING_INVALID"
REASON_RENDER_BINDING_STALE = "RENDER_BINDING_STALE"
REASON_MISSING_LOUDNESS_EVIDENCE = "MISSING_LOUDNESS_EVIDENCE"
REASON_ARTIFACT_MISSING = "ARTIFACT_MISSING"
REASON_ARTIFACT_HASH_MISMATCH = "ARTIFACT_HASH_MISMATCH"
REASON_UPSTREAM_REVALIDATION_REQUIRED = "UPSTREAM_REVALIDATION_REQUIRED"
REASON_PEAK_REPAIR_UNREPAIRABLE = "PEAK_REPAIR_UNREPAIRABLE"
REASON_SCRIPT_DEPENDENCY_MISMATCH = "SCRIPT_DEPENDENCY_MISMATCH"
REASON_RENDER_IDENTITY_INCOMPLETE = "RENDER_IDENTITY_INCOMPLETE"
REASON_EVIDENCE_CHANGED_DURING_PUBLICATION = "EVIDENCE_CHANGED_DURING_PUBLICATION"
ASSESSMENT_POLICY_VERSION = "stage6-assessment-v2"

# Reviewable outcomes: actual media exists and all critical gates are known and
# satisfied. Anything else is terminal-but-not-review-ready or deferred.
_REVIEWABLE_OUTCOMES = {
    MaterializationOutcome.READY_FOR_HUMAN_REVIEW.value,
    MaterializationOutcome.MATERIALIZED_WITH_CAUTION.value,
}
_DEFERRED_OUTCOMES = {
    MaterializationOutcome.DEFERRED_EVIDENCE.value,
    MaterializationOutcome.UPSTREAM_REVALIDATION_REQUIRED.value,
    MaterializationOutcome.DEFERRED_PROVIDER.value,
}

_REQUIRED_CONTEXT_KEYS = frozenset(AssessmentContext().as_dict().keys())
_BOOL_CONTEXT_KEYS = frozenset(
    {
        "required_slots_fulfilled",
        "required_narration_required",
        "narration_accepted",
        "tts_available",
        "provider_deferred",
        "unsupported_essential",
        "hero_protected",
        "asset_rights_ok",
        "script_audio_match",
        "readable_copy",
        "final_clip_usable",
        "critical_evidence_current",
        "critical_fidelity_unknown",
        "render_complete",
    }
)


@dataclass(frozen=True)
class FinalizationResult:
    """Truthful Stage 6 finalization outcome (never a publication permission)."""

    outcome: str
    reason_codes: tuple[str, ...] = ()
    assessment_fingerprint: str = ""
    render_execution_id: str | None = None
    reused: bool = False
    semantic_outcome: str | None = None
    findings: tuple[dict[str, object], ...] = field(default_factory=tuple)


def dispatch_transformed_render(
    session: Session,
    materialization: ClipMaterialization,
    candidate_id: uuid.UUID | str,
    *,
    settings: Settings,
    storage: StorageService,
    delivery_profile_key: str | None = None,
    dispatcher: object | None = None,
) -> uuid.UUID | None:
    """Bind and enqueue a TRANSFORMED_SHORT render for this exact materialization.

    Persists the exact render binding on the envelope before returning. A broker
    failure leaves the durable request in place for the next POST/resume to
    redispatch; it never silently drops the request.
    """

    from app.render.execution.policy import DEFAULT_DELIVERY_PROFILE_KEY
    from app.render.execution.queue import deliver_render_job, queue_render_execution

    if materialization.lifecycle is not MaterializationLifecycle.AWAITING_RENDER:
        return None
    if not materialization.output_fingerprint:
        return None
    delivery = DEFAULT_DELIVERY_PROFILE_KEY

    # Persist ALL bindings (the render envelope's reverse materialization binding
    # and this envelope's forward render binding) *before* broker delivery. The
    # queue commits the render row; capture its dispatch instead of delivering so
    # the forward binding is durable first, then deliver.
    captured: dict[str, uuid.UUID] = {}

    class _Capture:
        def dispatch(self, render_id: uuid.UUID, job_id: uuid.UUID, force: bool) -> None:
            captured["render_id"] = render_id
            captured["job_id"] = job_id

    outcome = queue_render_execution(
        session,
        candidate_id,
        artifact_purpose=TRANSFORMED_SHORT,
        delivery_profile_key=delivery,
        materialization_id=materialization.id,
        expected_output_fingerprint=materialization.output_fingerprint,
        settings=settings,
        dispatcher=_Capture(),
    )
    render_execution_id = uuid.UUID(str(outcome.render_execution_id))
    materialization.render_execution_id = render_execution_id
    session.commit()

    if outcome.job_id is not None:
        try:
            deliver_render_job(
                render_execution_id, uuid.UUID(str(outcome.job_id)), False, dispatcher
            )
        except Exception:
            _mark_render_dispatch_failed(session, outcome.job_id)
    return render_execution_id


def _mark_render_dispatch_failed(session: Session, job_id: object) -> None:
    from app.models import ProcessingJob

    job = session.get(ProcessingJob, job_id)
    if job is None:
        return
    job.error_code = "RENDER_DISPATCH_FAILED"
    try:
        session.commit()
    except Exception:
        session.rollback()


def request_finalization_job(
    session: Session,
    render_execution_id: uuid.UUID | str,
    *,
    settings: Settings,
    dispatcher: object | None = None,
) -> uuid.UUID | None:
    """Request a MATERIALIZATION job to finalize an AWAITING_RENDER envelope.

    Called after a transformed render reaches a terminal state. Coalesces with an
    existing active MATERIALIZATION job for the same envelope (duplicate callbacks
    are harmless). A durable ``MATERIALIZATION_DISPATCH_FAILED`` marker lets the
    next POST/resume recover a lost dispatch; GET never calls this.
    """

    from app.core.enums import JobKind
    from app.materialization.repository import bind_active_job
    from app.models import ProcessingJob

    parsed = (
        render_execution_id
        if isinstance(render_execution_id, uuid.UUID)
        else _as_uuid(render_execution_id)
    )
    if parsed is None:
        return None
    render = session.get(RenderExecution, parsed, populate_existing=True)
    if render is None or render.materialization_id is None:
        return None
    if render.artifact_purpose.value != TRANSFORMED_SHORT:
        return None
    if render.lifecycle.value not in {"COMPLETE", "FAILED", "BLOCKED"}:
        # Only a terminal render has authoritative final evidence.
        return None
    if not render.is_current:
        # Ignore historical/superseded callbacks.
        return None
    envelope = session.get(ClipMaterialization, render.materialization_id, populate_existing=True)
    if envelope is None or not envelope.is_current:
        return None
    if envelope.lifecycle is not MaterializationLifecycle.AWAITING_RENDER:
        return None
    # The callback render must be the envelope's exact bound revision.
    if envelope.render_execution_id is None or envelope.render_execution_id != render.id:
        return None
    if not _render_binding_matches(session, envelope, render):
        return None

    existing = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.materialization_id == envelope.id,
            ProcessingJob.kind == JobKind.MATERIALIZATION,
            ProcessingJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).first()
    if existing is not None:
        return uuid.UUID(str(existing.id))

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
        return None
    session.commit()
    session.refresh(job)
    _deliver_finalization_job(session, job, settings=settings, dispatcher=dispatcher)
    return uuid.UUID(str(job.id))


def _render_binding_matches(
    session: Session, envelope: ClipMaterialization, render: RenderExecution
) -> bool:
    return _validate_render_binding(envelope, render) is None


def _deliver_finalization_job(
    session: Session,
    job: Any,
    *,
    settings: Settings,
    dispatcher: object | None,
) -> None:
    from app.materialization.queue import DISPATCH_FAILED_MARKER

    envelope = session.get(ClipMaterialization, job.materialization_id)
    candidate_id = str(envelope.clip_candidate_id) if envelope is not None else ""
    try:
        if dispatcher is not None:
            call = getattr(dispatcher, "dispatch_finalization", None) or getattr(
                dispatcher, "dispatch", None
            )
            if callable(call):
                call(candidate_id, str(job.id))
            else:
                raise TypeError("dispatcher is not callable")
        else:
            from app.workers.tasks import run_materialization_finalization

            run_materialization_finalization.delay(candidate_id, str(job.id))
        job.error_code = None
    except Exception:
        job.error_code = DISPATCH_FAILED_MARKER
    try:
        session.commit()
    except Exception:
        session.rollback()


# ---------------------------------------------------------------------------
# Guarded finalization disposition
# ---------------------------------------------------------------------------


def _disposition(
    session: Session,
    owner: MaterializationOwner,
    *,
    outcome: str,
    reason_codes: tuple[str, ...] = (),
    job_status: JobStatus = JobStatus.SUCCEEDED,
    envelope_values: Mapping[str, object] | None = None,
    mutate_envelope: bool = True,
    assessment_fingerprint: str = "",
    render_execution_id: str | None = None,
    semantic_outcome: str | None = None,
    findings: tuple[dict[str, object], ...] = (),
    reused: bool = False,
) -> FinalizationResult:
    """Commit one handled finalization disposition and close the owned job.

    A handled semantic defer/rejection is not an unhandled execution failure: the
    owned job always becomes terminal. Envelope state is only mutated while this
    worker still holds the exact ownership token; a lost owner never mutates
    another worker's claim, but it also never leaves its own job RUNNING.
    """

    session.expire_all()
    if mutate_envelope and owner_is_current(session, owner):
        values: dict[str, object] = dict(envelope_values or {})
        values.setdefault("active_job_id", None)
        row = session.get(ClipMaterialization, owner.materialization_id, populate_existing=True)
        existing = list(row.reason_codes or []) if row is not None else []
        merged = list(dict.fromkeys((*existing, *(str(code) for code in reason_codes))))
        values["reason_codes"] = merged
        if not guarded_update(session, owner, values):
            # Envelope ownership was lost between the check and the write (for
            # example a concurrent supersession). The envelope is now owned by a
            # newer generation; never mutate it, but still terminalize *this*
            # execution's exact job below.
            session.rollback()
    # Always attempt to terminalize the owned RUNNING job. A handled disposition
    # (including a lost envelope) never leaves this execution's job RUNNING.
    if not finish_owner_job(session, owner, job_status):
        session.rollback()
        return FinalizationResult(FINALIZATION_SUPERSEDED, ("OWNERSHIP_LOST",))
    session.commit()
    return FinalizationResult(
        outcome,
        reason_codes=tuple(str(code) for code in reason_codes),
        assessment_fingerprint=assessment_fingerprint,
        render_execution_id=render_execution_id,
        reused=reused,
        semantic_outcome=semantic_outcome,
        findings=findings,
    )


def _ownership_lost(session: Session, owner: MaterializationOwner) -> FinalizationResult:
    """Terminalize this execution's exact owned job after losing the envelope.

    A superseded/demoted envelope may no longer belong to this execution, but the
    RUNNING job this worker claimed still does (until a newer owner reclaims it).
    Closing it under its exact job fence never modifies a newer owner's job and
    never leaves our execution's job stranded RUNNING.
    """

    if finish_owner_job(session, owner, JobStatus.SUCCEEDED):
        session.commit()
    else:
        session.rollback()
    return FinalizationResult(FINALIZATION_SUPERSEDED, ("OWNERSHIP_LOST",))


def _deferred_evidence_values() -> dict[str, object]:
    """Envelope values for a terminal-but-not-review-ready missing-evidence defer.

    A COMPLETE render whose required evidence is absent/incomplete is recorded as
    ``DEFERRED_EVIDENCE``: it never requests human review, is never cache-eligible,
    and never becomes publication-ready.
    """

    return {
        "lifecycle": MaterializationLifecycle.COMPLETE,
        "semantic_outcome": MaterializationOutcome.DEFERRED_EVIDENCE,
        "human_review_required": False,
        "cache_eligible": False,
        "publication_ready": False,
    }


def finalize_materialization(
    session: Session,
    owner: MaterializationOwner,
    *,
    settings: Settings,
    storage: StorageService,
) -> FinalizationResult:
    """Finalize one materialization from its exact stored render evidence."""

    session.expire_all()
    envelope = session.get(ClipMaterialization, owner.materialization_id, populate_existing=True)
    if envelope is None:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_SUPERSEDED,
            reason_codes=("MATERIALIZATION_MISSING",),
        )
    if envelope.lifecycle is MaterializationLifecycle.CANCELLED:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_CANCELLED,
            reason_codes=("MATERIALIZATION_CANCELLED",),
            job_status=JobStatus.CANCELLED,
        )
    if not owner_is_current(session, owner):
        return _ownership_lost(session, owner)

    render_id = envelope.render_execution_id
    if render_id is None:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_RENDER_NOT_COMPLETE,),
        )
    render = session.get(RenderExecution, render_id, populate_existing=True)
    if render is None:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_RENDER_NOT_COMPLETE,),
        )
    if render.lifecycle.value == "CANCELLED":
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_CANCELLED,
            reason_codes=("RENDER_CANCELLED",),
            job_status=JobStatus.CANCELLED,
        )
    if render.lifecycle.value not in {"COMPLETE", "FAILED", "BLOCKED"}:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_RENDER_NOT_COMPLETE,),
        )
    if not render.is_current:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_SUPERSEDED,
            reason_codes=("RENDER_NOT_CURRENT",),
        )

    binding = _validate_render_binding(envelope, render)
    if binding is not None:
        return _disposition(
            session,
            owner,
            outcome=binding.outcome,
            reason_codes=binding.reason_codes,
            job_status=(
                JobStatus.CANCELLED
                if binding.outcome == FINALIZATION_CANCELLED
                else JobStatus.SUCCEEDED
            ),
        )

    # Live currentness: the exact persisted request options must still assemble to
    # the same immutable input generation. Upstream can change during rendering;
    # a stale input may never authorize review readiness or cache reuse.
    inputs = _revalidate_live_inputs(session, envelope)
    if inputs is None:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_SUPERSEDED,
            reason_codes=(REASON_UPSTREAM_REVALIDATION_REQUIRED,),
            envelope_values={
                "lifecycle": MaterializationLifecycle.BLOCKED,
                "semantic_outcome": MaterializationOutcome.UPSTREAM_REVALIDATION_REQUIRED,
                "cache_eligible": False,
                "human_review_required": False,
                "publication_ready": False,
            },
        )

    if render.lifecycle.value != "COMPLETE":
        # A terminal FAILED/BLOCKED render is finalized truthfully: the exact
        # callback identity is already verified above, and a justified
        # reason-specific repair may be allocated and dispatched. A render with
        # no completed artifact and no justified repair defers (or rejects) and
        # is never promoted to review readiness.
        return _finalize_terminal_failure(
            session,
            owner,
            envelope,
            render,
            inputs,
            settings=settings,
            storage=storage,
        )

    measured = render.qc_result.get("measured") if isinstance(render.qc_result, dict) else {}
    measured = measured if isinstance(measured, dict) else {}
    integrated = measured.get("final_integrated_lufs")
    true_peak = measured.get("final_true_peak_dbtp")
    if render.qc_status is None:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_MISSING_LOUDNESS_EVIDENCE,),
            envelope_values=_deferred_evidence_values(),
            semantic_outcome=MaterializationOutcome.DEFERRED_EVIDENCE.value,
        )
    if not _finite(integrated) or not _finite(true_peak):
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_MISSING_LOUDNESS_EVIDENCE,),
            envelope_values=_deferred_evidence_values(),
            semantic_outcome=MaterializationOutcome.DEFERRED_EVIDENCE.value,
        )
    # A COMPLETE render must carry its exact, complete serialized identity: the
    # output/runtime/QC fingerprints, the artifact binding, and the measured final
    # audio. Missing fields are never accepted through defaults. (A pre-QC
    # FAILED/BLOCKED render is handled earlier and is not subject to this gate.)
    identity_error = _complete_render_identity_error(render, integrated, true_peak)
    if identity_error is not None:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(identity_error,),
            envelope_values=_deferred_evidence_values(),
            semantic_outcome=MaterializationOutcome.DEFERRED_EVIDENCE.value,
        )

    # Hash the artifact OUTSIDE any lock. The short publication transaction below
    # re-verifies this exact identity and the live dependencies before publishing.
    artifact = _artifact_identity(storage, render)
    if isinstance(artifact, str):
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(artifact,),
            envelope_values=_deferred_evidence_values(),
            semantic_outcome=MaterializationOutcome.DEFERRED_EVIDENCE.value,
        )

    config = settings.stage6_config()
    context = _build_assessment_context(
        session,
        envelope,
        render,
        inputs=inputs,
        integrated=float(cast(float, integrated)),
        true_peak=float(cast(float, true_peak)),
        config=config,
    )
    assessment = assess_materialization(
        context, output_duration=_output_duration(envelope), config=config
    )

    fingerprint = _assessment_fingerprint(envelope, render, measured, assessment)
    findings = _merge_findings(envelope, render, assessment)
    finding_reasons = [
        code
        for finding in findings
        if finding.get("level") == "UNMET"
        for code in _reason_tuple(finding)
    ]
    reasons = tuple(
        dict.fromkeys(
            (
                *list(envelope.reason_codes or []),
                *list(assessment.reason_codes),
                *(str(code) for code in finding_reasons),
            )
        )
    )
    metrics = dict(envelope.metrics or {})
    metrics["assessment_fingerprint"] = fingerprint
    metrics["final_integrated_lufs"] = integrated
    metrics["final_true_peak_dbtp"] = true_peak
    metrics["render_output_fingerprint"] = render.output_fingerprint
    metrics["assessment_policy_version"] = ASSESSMENT_POLICY_VERSION

    reviewable = assessment.outcome in _REVIEWABLE_OUTCOMES
    envelope_values = {
        "lifecycle": MaterializationLifecycle.COMPLETE,
        "semantic_outcome": MaterializationOutcome(assessment.outcome),
        "human_review_required": bool(assessment.human_review_required) if reviewable else False,
        "publication_ready": False,
        "cache_eligible": reviewable,
        "quality_findings": findings,
        "metrics": metrics,
    }
    return _publish_assessment(
        session,
        owner,
        render_id=render.id,
        artifact=artifact,
        assessment=assessment,
        fingerprint=fingerprint,
        findings=tuple(findings),
        reasons=reasons,
        envelope_values=envelope_values,
        storage=storage,
        inputs=inputs,
        config=config,
        integrated=float(cast(float, integrated)),
        true_peak=float(cast(float, true_peak)),
    )


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character.isalnum() for character in value)
    )


def _complete_render_identity_error(
    render: RenderExecution, integrated: object, true_peak: object
) -> str | None:
    """Require the exact successful-render identity on the COMPLETE branch.

    Output/runtime/QC fingerprint identity, the artifact reference binding, and
    finite measured final audio are independently required: the callback identity
    of a terminal failed/blocked render must never stand in for a successful
    assessment's complete evidence. Missing/malformed fields are refused rather
    than defaulted.
    """

    if render.qc_status is None:
        return REASON_MISSING_LOUDNESS_EVIDENCE
    for value in (render.output_fingerprint, render.runtime_fingerprint, render.qc_fingerprint):
        if not _valid_hash(value):
            return REASON_RENDER_IDENTITY_INCOMPLETE
    reference = render.artifact_reference if isinstance(render.artifact_reference, dict) else {}
    if not str(reference.get("relative_path") or ""):
        return REASON_ARTIFACT_MISSING
    if not _valid_hash(str(reference.get("sha256") or "")):
        return REASON_ARTIFACT_MISSING
    if not _finite(integrated) or not _finite(true_peak):
        return REASON_MISSING_LOUDNESS_EVIDENCE
    return None


@dataclass(frozen=True)
class _ArtifactIdentity:
    relative_path: str
    sha256: str
    size_bytes: int
    mtime_ns: int


def _artifact_identity(storage: StorageService, render: RenderExecution) -> _ArtifactIdentity | str:
    """Verify the artifact outside any lock; return its identity or a reason code."""

    reference = render.artifact_reference if isinstance(render.artifact_reference, dict) else {}
    relative = reference.get("relative_path")
    expected_hash = str(reference.get("sha256") or "")
    if not isinstance(relative, str) or not relative or not expected_hash:
        return REASON_ARTIFACT_MISSING
    path = (storage.storage_root / relative).resolve()
    try:
        path.relative_to(storage.storage_root)
    except ValueError:
        return REASON_ARTIFACT_MISSING
    if not path.is_file():
        return REASON_ARTIFACT_MISSING
    digest = _sha256(path)
    if digest != expected_hash.lower():
        return REASON_ARTIFACT_HASH_MISMATCH
    stat = path.stat()
    return _ArtifactIdentity(
        relative_path=relative,
        sha256=digest,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _acquire_publication_locks(
    session: Session, envelope: ClipMaterialization, candidate_id: object
) -> bool:
    """Acquire the frozen lock order: candidate -> contract -> plan -> clip -> visual.

    Extended, consistently, with the mutable selection/profile rows the finalizer
    must observe. These are acquired *before* any materialization/render/job lock,
    so the order matches queueing (which locks the candidate first) and no
    candidate/upstream lock is ever taken after an envelope/job lock.
    """

    from app.models import ClipCandidate
    from app.models.candidate_refinement import CandidateRefinement
    from app.models.channel_production_profile import ChannelProductionProfile
    from app.models.transformation_plan import TransformationPlan
    from app.models.transformation_selection import TransformationPlanSelection
    from app.models.visual_composition_plan import VisualCompositionPlan

    locked = session.scalars(
        select(ClipCandidate)
        .where(ClipCandidate.id == candidate_id, ClipCandidate.is_current.is_(True))
        .with_for_update()
    ).first()
    if locked is None:
        return False
    if envelope.selection_id is not None:
        session.execute(
            select(TransformationPlanSelection.id)
            .where(TransformationPlanSelection.id == envelope.selection_id)
            .with_for_update()
        ).first()
    contract = None
    if envelope.render_contract_id is not None:
        contract = session.get(RenderContract, envelope.render_contract_id, with_for_update=True)
    if contract is not None:
        if contract.selected_plan_id is not None:
            session.execute(
                select(TransformationPlan.id)
                .where(TransformationPlan.id == contract.selected_plan_id)
                .with_for_update()
            ).first()
        if contract.final_refinement_id is not None:
            session.execute(
                select(CandidateRefinement.id)
                .where(CandidateRefinement.id == contract.final_refinement_id)
                .with_for_update()
            ).first()
    elif envelope.selected_plan_id is not None:
        session.execute(
            select(TransformationPlan.id)
            .where(TransformationPlan.id == envelope.selected_plan_id)
            .with_for_update()
        ).first()
    if envelope.refinement_id is not None:
        session.execute(
            select(CandidateRefinement.id)
            .where(CandidateRefinement.id == envelope.refinement_id)
            .with_for_update()
        ).first()
    if envelope.visual_composition_plan_id is not None:
        session.execute(
            select(VisualCompositionPlan.id)
            .where(VisualCompositionPlan.id == envelope.visual_composition_plan_id)
            .with_for_update()
        ).first()
    if envelope.channel_production_profile_id is not None:
        session.execute(
            select(ChannelProductionProfile.id)
            .where(ChannelProductionProfile.id == envelope.channel_production_profile_id)
            .with_for_update()
        ).first()
    return True


def _artifact_still_matches(
    storage: StorageService, artifact: _ArtifactIdentity
) -> bool:
    """Verify the exact hashed artifact identity (size + mtime, not a re-hash) under the lock."""

    path = (storage.storage_root / artifact.relative_path).resolve()
    try:
        path.relative_to(storage.storage_root)
    except ValueError:
        return False
    if not path.is_file():
        return False
    stat = path.stat()
    return bool(stat.st_size == artifact.size_bytes and stat.st_mtime_ns == artifact.mtime_ns)


def _publish_assessment(
    session: Session,
    owner: MaterializationOwner,
    *,
    render_id: uuid.UUID,
    artifact: _ArtifactIdentity,
    assessment: Any,
    fingerprint: str,
    findings: tuple[dict[str, object], ...],
    reasons: tuple[str, ...],
    envelope_values: Mapping[str, object],
    storage: StorageService,
    inputs: MaterializationInputs,
    config: Any,
    integrated: float,
    true_peak: float,
) -> FinalizationResult:
    """Publish the final assessment under a short fenced, locked transaction.

    The lock-free preparation above hashed the artifact and computed the
    assessment. Here we end any prior snapshot, acquire the architecture's lock
    order, re-run the live upstream revalidation / render-binding / artifact
    checks *under* the locks, recompute the assessment from the freshly-locked
    evidence, and finish the envelope/job/release together. A prerequisite
    mutation, render demotion, cancellation, or output/repair change committed
    after the initial revalidation can never authorize a stale review readiness,
    and an assessment computed against other evidence is never published.
    """

    session.rollback()  # end any prior transaction; take a fresh snapshot

    # The candidate/upstream locks must be acquired FIRST: queueing takes the
    # candidate lock before the envelope, so taking the envelope lock first would
    # invert the frozen order and permit a deadlock cycle.
    envelope = session.get(ClipMaterialization, owner.materialization_id, populate_existing=True)
    if envelope is None:
        return FinalizationResult(FINALIZATION_SUPERSEDED, ("MATERIALIZATION_MISSING",))
    if not _acquire_publication_locks(session, envelope, envelope.clip_candidate_id):
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_SUPERSEDED,
            reason_codes=(REASON_UPSTREAM_REVALIDATION_REQUIRED,),
            envelope_values={
                "lifecycle": MaterializationLifecycle.BLOCKED,
                "semantic_outcome": MaterializationOutcome.UPSTREAM_REVALIDATION_REQUIRED,
                "cache_eligible": False,
                "human_review_required": False,
                "publication_ready": False,
            },
        )

    # Now lock materialization -> render; the job row is locked last by
    # ``finish_owner_job``/``_disposition``. Fresh, populated reads only.
    envelope = session.get(
        ClipMaterialization, owner.materialization_id, populate_existing=True, with_for_update=True
    )
    if envelope is None:
        return _ownership_lost(session, owner)
    render = session.get(
        RenderExecution, render_id, populate_existing=True, with_for_update=True
    )
    if render is None or render.lifecycle.value != "COMPLETE":
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_RENDER_NOT_COMPLETE,),
        )
    # Revalidate exact current render scope: the still-bound, still-current render
    # revision. A demoted/superseded render is never published as current.
    if envelope.render_execution_id != render.id or not render.is_current:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_SUPERSEDED,
            reason_codes=("RENDER_NOT_CURRENT",),
        )
    if not owner_is_current(session, owner):
        return _ownership_lost(session, owner)
    if envelope.lifecycle is MaterializationLifecycle.CANCELLED:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_CANCELLED,
            reason_codes=("MATERIALIZATION_CANCELLED",),
            job_status=JobStatus.CANCELLED,
        )
    live_inputs = _revalidate_live_inputs(session, envelope)
    if live_inputs is None:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_SUPERSEDED,
            reason_codes=(REASON_UPSTREAM_REVALIDATION_REQUIRED,),
            envelope_values={
                "lifecycle": MaterializationLifecycle.BLOCKED,
                "semantic_outcome": MaterializationOutcome.UPSTREAM_REVALIDATION_REQUIRED,
                "cache_eligible": False,
                "human_review_required": False,
                "publication_ready": False,
            },
        )
    verified = _validate_render_binding(envelope, render)
    if verified is not None:
        return _disposition(
            session,
            owner,
            outcome=verified.outcome,
            reason_codes=verified.reason_codes,
        )
    identity_error = _complete_render_identity_error(render, integrated, true_peak)
    if identity_error is not None:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(identity_error,),
            envelope_values=_deferred_evidence_values(),
            semantic_outcome=MaterializationOutcome.DEFERRED_EVIDENCE.value,
        )
    if str(artifact.relative_path) != str(
        (render.artifact_reference or {}).get("relative_path") or ""
    ) or not _artifact_still_matches(storage, artifact):
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_ARTIFACT_HASH_MISMATCH,),
            envelope_values=_deferred_evidence_values(),
            semantic_outcome=MaterializationOutcome.DEFERRED_EVIDENCE.value,
        )

    # Recompute the assessment from the freshly-locked evidence and refuse to
    # publish an assessment computed against different render/QC/semantic
    # evidence.
    measured = render.qc_result.get("measured") if isinstance(render.qc_result, dict) else {}
    measured = measured if isinstance(measured, dict) else {}
    fresh_integrated = measured.get("final_integrated_lufs")
    fresh_peak = measured.get("final_true_peak_dbtp")
    if not _finite(fresh_integrated) or not _finite(fresh_peak):
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_MISSING_LOUDNESS_EVIDENCE,),
        )
    fresh_context = _build_assessment_context(
        session,
        envelope,
        render,
        inputs=live_inputs,
        integrated=float(cast(float, fresh_integrated)),
        true_peak=float(cast(float, fresh_peak)),
        config=config,
    )
    fresh_assessment = assess_materialization(
        fresh_context, output_duration=_output_duration(envelope), config=config
    )
    fresh_fingerprint = _assessment_fingerprint(envelope, render, measured, fresh_assessment)
    if fresh_fingerprint != fingerprint or fresh_assessment.outcome != assessment.outcome:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_EVIDENCE_CHANGED_DURING_PUBLICATION,),
        )

    existing = (
        envelope.metrics.get("assessment_fingerprint")
        if isinstance(envelope.metrics, dict)
        else None
    )
    if envelope.lifecycle is MaterializationLifecycle.COMPLETE and existing == fingerprint:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_COMPLETE,
            reason_codes=assessment.reason_codes,
            mutate_envelope=False,
            assessment_fingerprint=fingerprint,
            render_execution_id=str(render.id),
            reused=True,
            semantic_outcome=_existing_semantic_outcome(envelope, assessment.outcome),
        )
    return _disposition(
        session,
        owner,
        outcome=_disposition_outcome(assessment.outcome),
        reason_codes=reasons,
        job_status=JobStatus.SUCCEEDED,
        envelope_values=dict(envelope_values),
        assessment_fingerprint=fingerprint,
        render_execution_id=str(render.id),
        semantic_outcome=assessment.outcome,
        findings=findings,
    )



def _disposition_outcome(assessment_outcome: str) -> str:
    if assessment_outcome in _REVIEWABLE_OUTCOMES:
        return FINALIZATION_COMPLETE
    if assessment_outcome in _DEFERRED_OUTCOMES:
        return FINALIZATION_DEFERRED
    return FINALIZATION_COMPLETE


# ---------------------------------------------------------------------------
# Terminal-failure finalization and the one bounded repair
# ---------------------------------------------------------------------------

REPAIR_KIND_BOUNDED_PEAK = "BOUNDED_PEAK_REPAIR"
_REPAIR_GUARD_FALLBACK_DBTP = -1.8
_PEAK_REPAIR_MIN_DB = 0.3
_PEAK_REPAIR_MAX_DB = 3.0


def _render_reason_codes(render: RenderExecution) -> tuple[str, ...]:
    codes = [str(code) for code in (render.reason_codes or [])]
    if render.error_code:
        codes.append(str(render.error_code))
    return tuple(dict.fromkeys(codes))


def _render_partial_manifest(render: RenderExecution) -> Mapping[str, object]:
    manifest = render.execution_manifest
    return manifest if isinstance(manifest, Mapping) else {}


def _render_guard_dbtp(render: RenderExecution) -> float:
    request = render.request_payload if isinstance(render.request_payload, Mapping) else {}
    spec = request.get("spec")
    spec_map = spec if isinstance(spec, Mapping) else {}
    authored = spec_map.get("authored")
    authored_map = authored if isinstance(authored, Mapping) else {}
    mastering = authored_map.get("mastering")
    mastering_map = mastering if isinstance(mastering, Mapping) else {}
    value = mastering_map.get("render_guard_dbtp")
    if _finite(value):
        return float(cast(float, value))
    return _REPAIR_GUARD_FALLBACK_DBTP


@dataclass(frozen=True)
class _PeakRepair:
    reduction_db: float
    render_guard_dbtp: float
    evidence: Mapping[str, object]


def _select_peak_repair(
    render: RenderExecution, envelope: ClipMaterialization
) -> _PeakRepair | None:
    """Return the one justified bounded peak-limiter repair, or ``None``.

    A repair is only justified by a terminal ``AUDIO_PEAK_REPAIR_NEEDED`` render
    that persisted a real unmastered measurement showing the ideal linear target
    is infeasible under the render true-peak guard. A missing/invalid measurement,
    a reduction below the minimum, or a reduction beyond the bounded maximum is
    not repairable here.
    """

    if AUDIO_PEAK_REPAIR_NEEDED not in _render_reason_codes(render):
        return None
    measurement = _render_partial_manifest(render).get("mix_measurement")
    if not isinstance(measurement, Mapping):
        return None
    input_i = measurement.get("input_i")
    input_tp = measurement.get("input_tp")
    if not _finite(input_i) or not _finite(input_tp):
        return None
    recipe = envelope.audio_recipe if isinstance(envelope.audio_recipe, Mapping) else {}
    target = recipe.get("target_integrated_lufs")
    target_lufs = float(cast(float, target)) if _finite(target) else -16.0
    guard = _render_guard_dbtp(render)
    ideal_gain = target_lufs - float(cast(float, input_i))
    guard_gain = guard - float(cast(float, input_tp))
    needed = ideal_gain - guard_gain
    if needed <= 0 or needed > _PEAK_REPAIR_MAX_DB + 1e-9:
        return None
    reduction = max(_PEAK_REPAIR_MIN_DB, min(_PEAK_REPAIR_MAX_DB, needed))
    return _PeakRepair(
        reduction_db=round(reduction, 6),
        render_guard_dbtp=guard,
        evidence={
            "mix_measurement": dict(measurement),
            "ideal_gain_db": round(ideal_gain, 6),
            "guard_gain_db": round(guard_gain, 6),
        },
    )


def _repair_output_fingerprint(base_fp: str, action: Mapping[str, object]) -> str:
    """Revise the output identity for a repair revision (old render binds the old)."""

    return str(stable_digest({"base_output_fingerprint": base_fp, "repair_action": dict(action)}))


def _finalize_terminal_failure(
    session: Session,
    owner: MaterializationOwner,
    envelope: ClipMaterialization,
    render: RenderExecution,
    inputs: MaterializationInputs,
    *,
    settings: Settings,
    storage: StorageService,
) -> FinalizationResult:
    """Finalize a terminal FAILED/BLOCKED render truthfully.

    Allocates and dispatches the single reason-specific bounded repair when a
    persisted ``AUDIO_PEAK_REPAIR_NEEDED`` measurement justifies it (CAS 0 -> 1),
    preserving budgets/history and revising the output/render identity so the old
    callback can never finalize the repaired revision. Otherwise the render is
    rejected (unrepairable peak) or deferred (missing artifact evidence).
    """

    reasons = _render_reason_codes(render)
    repair = _select_peak_repair(render, envelope)
    peak_related = AUDIO_PEAK_REPAIR_NEEDED in reasons
    if repair is None or int(envelope.repair_count or 0) >= 1:
        if peak_related:
            return _reject_unrepairable(session, owner, envelope, reasons)
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=tuple(dict.fromkeys((REASON_RENDER_NOT_COMPLETE, *reasons))),
            envelope_values={
                "lifecycle": MaterializationLifecycle.COMPLETE,
                "semantic_outcome": MaterializationOutcome.DEFERRED_EVIDENCE,
                "human_review_required": False,
                "publication_ready": False,
                "cache_eligible": False,
            },
            semantic_outcome=MaterializationOutcome.DEFERRED_EVIDENCE.value,
        )

    reduction = repair.reduction_db
    action: dict[str, object] = {
        "kind": REPAIR_KIND_BOUNDED_PEAK,
        "reason_codes": [AUDIO_PEAK_REPAIR_NEEDED],
        "params": {
            "repair_peak_limit_db": reduction,
            "render_guard_dbtp": repair.render_guard_dbtp,
        },
        "evidence": dict(repair.evidence),
        "source_render_execution_id": str(render.id),
    }
    recipe = dict(envelope.audio_recipe or {})
    recipe["repair_peak_limit_db"] = reduction
    metrics = dict(envelope.metrics or {})
    metrics["repair_peak_limit_db"] = reduction
    new_fp = _repair_output_fingerprint(str(envelope.output_fingerprint or ""), action)

    if not allocate_repair(
        session,
        owner,
        {
            "lifecycle": MaterializationLifecycle.AWAITING_RENDER,
            "semantic_outcome": None,
            "human_review_required": False,
            "publication_ready": False,
            "cache_eligible": False,
            "output_fingerprint": new_fp,
            "render_execution_id": None,
            "audio_recipe": recipe,
            "repair_actions": [*(envelope.repair_actions or []), action],
            "metrics": metrics,
            "reason_codes": [],
            "active_job_id": None,
        },
    ):
        session.rollback()
        # Another worker spent the repair or the owner was superseded.
        return _reject_unrepairable(session, owner, envelope, reasons)
    if not finish_owner_job(session, owner, JobStatus.SUCCEEDED):
        session.rollback()
        return FinalizationResult(FINALIZATION_SUPERSEDED, ("OWNERSHIP_LOST",))
    session.commit()

    render_id = _dispatch_repaired_render(
        session, envelope.id, render.clip_candidate_id, settings=settings, storage=storage
    )
    return FinalizationResult(
        FINALIZATION_REPAIR_QUEUED,
        reason_codes=tuple(dict.fromkeys((REPAIR_KIND_BOUNDED_PEAK, *reasons))),
        render_execution_id=str(render_id) if render_id is not None else None,
    )


def _reject_unrepairable(
    session: Session,
    owner: MaterializationOwner,
    envelope: ClipMaterialization,
    reasons: tuple[str, ...],
) -> FinalizationResult:
    return _disposition(
        session,
        owner,
        outcome=FINALIZATION_COMPLETE,
        reason_codes=tuple(dict.fromkeys((REASON_PEAK_REPAIR_UNREPAIRABLE, *reasons))),
        envelope_values={
            "lifecycle": MaterializationLifecycle.COMPLETE,
            "semantic_outcome": MaterializationOutcome.REJECTED_QUALITY,
            "human_review_required": False,
            "publication_ready": False,
            "cache_eligible": False,
        },
        semantic_outcome=MaterializationOutcome.REJECTED_QUALITY.value,
    )


def _dispatch_repaired_render(
    session: Session,
    materialization_id: uuid.UUID,
    candidate_id: object,
    *,
    settings: Settings,
    storage: StorageService,
) -> uuid.UUID | None:
    envelope = session.get(ClipMaterialization, materialization_id, populate_existing=True)
    if envelope is None:
        return None
    return dispatch_transformed_render(
        session,
        envelope,
        candidate_id,  # type: ignore[arg-type]
        settings=settings,
        storage=storage,
    )


# ---------------------------------------------------------------------------
# Live currentness
# ---------------------------------------------------------------------------


def _revalidate_live_inputs(
    session: Session, envelope: ClipMaterialization
) -> MaterializationInputs | None:
    """Re-assemble live inputs from the persisted request options.

    Returns the freshly assembled inputs when the current generation still
    matches the envelope's immutable input fingerprint; otherwise ``None``.
    Never raises for a stale prerequisite; it reports staleness truthfully.
    """

    from app.materialization.service import MaterializationError, assemble_inputs

    request = envelope.request_payload if isinstance(envelope.request_payload, Mapping) else {}
    priority = str(request.get("priority") or MaterializationPriority.CANDIDATE.value)
    if priority not in {
        MaterializationPriority.CANDIDATE.value,
        MaterializationPriority.FINAL_CLIP.value,
    }:
        priority = MaterializationPriority.CANDIDATE.value
    raw_profile = request.get("requested_profile_key", request.get("profile_key"))
    profile_key = str(raw_profile) if raw_profile else None
    supporting = str(request.get("supporting_requested") or "OFF")
    try:
        session.expire_all()
        current = assemble_inputs(
            session,
            envelope.clip_candidate_id,
            priority=priority,
            profile_key=profile_key,
            supporting_requested=supporting,
        )
    except MaterializationError:
        return None
    except Exception:
        return None
    if input_fingerprint(current) != str(envelope.input_fingerprint or ""):
        return None
    return current


# ---------------------------------------------------------------------------
# Assessment context reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SlotEvidence:
    required_slots_fulfilled: bool
    explicit_failure: bool
    missing_resolution: bool


#: Findings the deterministic pre-render assessment always emits. A context
#: rebuilt from stored findings is complete only when every one is present; an
#: absent critical dimension is missing evidence, never a favorable default.
_REBUILD_REQUIRED_DIMENSIONS = frozenset(
    {
        "SLOT_FULFILLMENT",
        "GROUNDING_VERIFICATION",
        "HERO_PRESERVATION",
        "VISUAL_CAPTION_READABILITY",
        "PLATFORM_SUITABILITY",
    }
)

_SEVERITY = {"UNMET": 3, "UNKNOWN": 2, "CONCERN": 1, "SATISFIED": 0}

#: Levels that constitute a *known* (resolved) finding. ``UNKNOWN`` is explicitly
#: excluded: a mere dimension label never establishes a satisfied gate.
_KNOWN_LEVELS = frozenset({"SATISFIED", "CONCERN", "UNMET", "NOT_APPLICABLE"})

#: Semantically critical dimensions whose UNKNOWN/absent resolution is
#: insufficient evidence: the finalizer must defer rather than treat the missing
#: judgment as favorable.
_CRITICAL_SEMANTIC_DIMENSIONS = frozenset(
    {"GROUNDING_VERIFICATION", "FIDELITY_ATTRIBUTION", "SUBSTANTIVE_VALUE"}
)

#: Slot kinds whose FULFILLED label must correspond to actually accepted material.
_AUTHORED_SLOT_KINDS = frozenset({"ORIGINAL_VALUE", "TEXTUAL_ANNOTATION", "AUTHORED_TEXT"})


def _required_contract_slots(contract: RenderContract | None) -> list[Mapping[str, object]]:
    if contract is None:
        return []
    payload = dict(contract.contract_payload or {})
    materialization = payload.get("materialization")
    if not isinstance(materialization, Mapping):
        return []
    slots = materialization.get("slots")
    if not isinstance(slots, list):
        return []
    return [
        slot
        for slot in slots
        if isinstance(slot, Mapping) and bool(slot.get("required", False))
    ]


def _valid_authored_block(block: Mapping[str, object]) -> bool:
    index = block.get("block_index")
    if isinstance(index, bool) or not isinstance(index, int):
        return False
    if not isinstance(block.get("block_type"), str) or not str(block.get("block_type")):
        return False
    if not isinstance(block.get("slot_id"), str) or not str(block.get("slot_id")):
        return False
    return isinstance(block.get("copy"), str)


def _accepted_script_blocks(
    envelope: ClipMaterialization,
    inputs: MaterializationInputs,
    config: Any,
) -> tuple[bool, tuple[Mapping[str, object], ...]]:
    """Validate the SCRIPT checkpoint's *accepted content*, not its status label.

    An accepted checkpoint must carry a well-formed ``result`` with validated
    authored blocks, a matching ``result_fingerprint``, top-level
    ``authored_blocks`` that equal that content, and a dependency identity that
    still resolves against the *live* generation dependencies (selected semantic
    requirements, resolved source evidence, verification dependencies, target
    context, narration config, writer/critic identities). A bare
    ``{"status":"ACCEPTED"}``, a self-consistent content digest with no dependency
    binding, or a checkpoint copied from another generation is incomplete
    evidence and is refused.
    """

    checkpoint = _script_checkpoint(envelope)
    if not checkpoint or checkpoint.get("status") != "ACCEPTED":
        return False, ()
    if not _script_dependency_matches(checkpoint, inputs, config):
        return False, ()
    result = checkpoint.get("result")
    if not isinstance(result, Mapping):
        return False, ()
    raw = result.get("authored_blocks")
    if not isinstance(raw, list):
        return False, ()
    blocks = tuple(block for block in raw if isinstance(block, Mapping))
    if len(blocks) != len(raw) or any(not _valid_authored_block(block) for block in blocks):
        return False, ()
    fingerprint = checkpoint.get("result_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != stable_digest(list(raw)):
        return False, ()
    top = envelope.authored_blocks if isinstance(envelope.authored_blocks, list) else []
    top_blocks = [dict(block) for block in top if isinstance(block, Mapping)]
    if top_blocks != [dict(block) for block in blocks]:
        return False, ()
    return True, blocks


def _script_dependency_matches(
    checkpoint: Mapping[str, object],
    inputs: MaterializationInputs,
    config: Any,
) -> bool:
    """Prove the stored SCRIPT dependency identity resolves to *this* generation.

    The persisted dependency input set must be complete, must digest to its stored
    fingerprint, and must equal the set recomputed from the live inputs (with the
    persisted writer/critic identities). A missing field is never accepted through
    a default; a checkpoint from another generation is refused.
    """

    dependency = checkpoint.get("dependency_fingerprint")
    if not isinstance(dependency, str) or not dependency:
        return False
    raw = checkpoint.get("dependency_inputs")
    if not isinstance(raw, Mapping):
        return False
    raw_map = dict(raw)
    if stable_digest(raw_map) != dependency:
        return False
    writing_identity = raw_map.get("writing_identity")
    critique_identity = raw_map.get("critique_identity")
    if not isinstance(writing_identity, Mapping):
        return False
    if critique_identity is not None and not isinstance(critique_identity, Mapping):
        return False
    expected = script_dependency_inputs(
        inputs,
        writing_identity,
        critique_identity,
        config,
    )
    # Canonical comparison: the live-recomputed dependency set must digest to the
    # same stored fingerprint (tuple/list JSON round-trips are normalized).
    return bool(stable_digest(expected) == dependency)


def _narration_references(envelope: ClipMaterialization) -> tuple[Mapping[str, object], ...]:
    references: list[Mapping[str, object]] = []
    for item in envelope.narration_references or []:
        if not isinstance(item, Mapping):
            continue
        block_index = item.get("block_index")
        relative = item.get("relative_path")
        digest = item.get("sha256")
        if (
            isinstance(block_index, int)
            and not isinstance(block_index, bool)
            and isinstance(relative, str)
            and relative
            and isinstance(digest, str)
            and digest
        ):
            references.append(item)
    return tuple(references)


def _timeline_content(envelope: ClipMaterialization) -> tuple[bool, bool]:
    """Validate the persisted authored timeline content and hero protection.

    Returns ``(ok, hero_bound)``. A hero-marked timeline whose segments are
    malformed, whose hero is not an intact SOURCE interval, or whose frame/sample
    totals are missing is not acceptable evidence.
    """

    timeline = envelope.timeline_payload if isinstance(envelope.timeline_payload, Mapping) else {}
    segments = timeline.get("segments")
    if not isinstance(segments, list) or not segments:
        return False, False
    if not isinstance(timeline.get("schema_version"), str) or not timeline.get("schema_version"):
        return False, False
    for count_key in ("output_frame_count", "output_sample_count"):
        value = timeline.get(count_key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return False, False
    hero_segments: list[Mapping[str, object]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            return False, False
        if not isinstance(segment.get("segment_id"), str) or not segment.get("segment_id"):
            return False, False
        if not isinstance(segment.get("kind"), str) or not segment.get("kind"):
            return False, False
        index = segment.get("plan_block_index")
        if isinstance(index, bool) or not isinstance(index, int):
            return False, False
        start = segment.get("output_start")
        end = segment.get("output_end")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or float(end) < float(start)
        ):
            return False, False
        if segment.get("is_hero") is True:
            hero_segments.append(segment)
    if not hero_segments:
        return True, False
    hero_bound = True
    for hero in hero_segments:
        if hero.get("kind") != "SOURCE":
            hero_bound = False
        for bound_key in ("source_start", "source_end"):
            value = hero.get(bound_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                hero_bound = False
    return True, hero_bound


def _slot_evidence(
    session: Session,
    envelope: ClipMaterialization,
    *,
    authored_blocks: tuple[Mapping[str, object], ...],
    narration: tuple[Mapping[str, object], ...],
) -> _SlotEvidence:
    """Derive required-slot fulfilment from the contract + persisted resolutions.

    An empty resolution list never means "no requirements": a required contract
    slot without a persisted FULFILLED resolution is missing evidence, and a
    FULFILLED label that does not correspond to actually accepted material
    (an authored block or narration chunk) is likewise missing evidence, never a
    satisfied gate. An explicit DEFERRED/BLOCKED required resolution is a failure.
    """

    contract = (
        session.get(RenderContract, envelope.render_contract_id)
        if envelope.render_contract_id
        else None
    )
    rows = [dict(row) for row in (envelope.slot_resolutions or []) if isinstance(row, Mapping)]
    by_pair = {
        (str(row.get("slot_id")), str(row.get("slot_kind", ""))): row for row in rows
    }
    authored_indexes: set[int] = set()
    for block in authored_blocks:
        index = block.get("block_index")
        if isinstance(index, int) and not isinstance(index, bool):
            authored_indexes.add(index)
    narrated_indexes: set[int] = set()
    for chunk in narration:
        index = chunk.get("block_index")
        if isinstance(index, int) and not isinstance(index, bool):
            narrated_indexes.add(index)

    explicit_failure = False
    missing = contract is None
    for slot in _required_contract_slots(contract):
        pair = (str(slot.get("slot_id")), str(slot.get("slot_kind", "")))
        row = by_pair.get(pair)
        if row is None:
            missing = True
            continue
        resolution = str(row.get("resolution"))
        if resolution in {"DEFERRED", "BLOCKED", "DECLINED"}:
            explicit_failure = True
        elif resolution == "FULFILLED":
            if not _slot_material_present(pair[1], row, authored_indexes, narrated_indexes):
                missing = True
        else:
            missing = True
    for row in rows:
        if bool(row.get("required", True)) and str(row.get("resolution")) in {
            "DEFERRED",
            "BLOCKED",
        }:
            explicit_failure = True
    return _SlotEvidence(
        required_slots_fulfilled=not explicit_failure,
        explicit_failure=explicit_failure,
        missing_resolution=missing,
    )


def _slot_material_present(
    slot_kind: str,
    row: Mapping[str, object],
    authored_indexes: set[int],
    narrated_indexes: set[int],
) -> bool:
    index = row.get("block_index")
    if isinstance(index, bool) or not isinstance(index, int):
        return slot_kind not in (_AUTHORED_SLOT_KINDS | {"AUTHORED_NARRATION"})
    if slot_kind in _AUTHORED_SLOT_KINDS:
        return index in authored_indexes
    if slot_kind == "AUTHORED_NARRATION":
        return index in narrated_indexes
    return True


def _finding_levels(findings: list[QualityFinding]) -> dict[str, str]:
    levels: dict[str, str] = {}
    for finding in findings:
        current = levels.get(finding.dimension)
        if current is None or _SEVERITY.get(finding.level, -1) > _SEVERITY.get(current, -1):
            levels[finding.dimension] = finding.level
    return levels


def _merge_context_findings(
    prior: tuple[QualityFinding, ...],
    row: list[QualityFinding],
) -> list[QualityFinding]:
    """Union the snapshot's and the row's findings without losing provenance.

    Both the dependency-bound pre-render snapshot and the persisted row are
    authoritative adverse evidence: neither may silently replace the other. A
    finding is identified by ``(dimension, level, reason_codes)`` and the first
    occurrence wins, so the snapshot's record (including its evidence) is
    preserved while a contradictory row finding is still retained. Severity is
    never weakened: the worst level per dimension still dominates in
    :func:`_finding_levels` and every raw finding still reaches the assessment.
    """

    merged: list[QualityFinding] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for finding in (*prior, *row):
        key = (finding.dimension, finding.level, tuple(finding.reason_codes))
        if key in seen:
            continue
        seen.add(key)
        merged.append(finding)
    return merged


def _load_snapshot(envelope: ClipMaterialization) -> AssessmentContext | None:
    """Load the versioned, dependency-bound pre-render snapshot, or ``None``.

    A snapshot whose version is unknown, whose output-fingerprint binding does
    not match this exact output revision, or whose fields are incomplete/malformed
    is refused: the finalizer never deserializes favorable defaults from a
    partial or stale checkpoint.
    """

    metrics = envelope.metrics if isinstance(envelope.metrics, Mapping) else {}
    raw = metrics.get(PRE_RENDER_CONTEXT_METRIC)
    if not isinstance(raw, Mapping):
        return None
    if raw.get("version") != PRE_RENDER_CONTEXT_VERSION:
        return None
    if str(raw.get("materialization_output_fingerprint") or "") != str(
        envelope.output_fingerprint or ""
    ):
        return None
    context = raw.get("context")
    if not isinstance(context, Mapping) or not _valid_snapshot_context(context):
        return None
    return AssessmentContext.from_dict(context)


def _valid_snapshot_context(context: Mapping[str, object]) -> bool:
    if not _REQUIRED_CONTEXT_KEYS.issubset(set(context.keys())):
        return False
    for key in _BOOL_CONTEXT_KEYS:
        if not isinstance(context.get(key), bool):
            return False
    for key in ("measured_loudness_lufs", "measured_true_peak_dbtp"):
        value = context.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return False
    repair = context.get("repair_count")
    if isinstance(repair, bool) or not isinstance(repair, int):
        return False
    for key in (
        "extra_findings",
        "profile_mismatch_fields",
        "timeline_conflicts",
        "audio_conflicts",
    ):
        if not isinstance(context.get(key), list):
            return False
    qc_status = context.get("render_qc_status")
    if qc_status is not None and not isinstance(qc_status, str):
        return False
    return True


def _stored_findings(envelope: ClipMaterialization) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for item in envelope.quality_findings or []:
        if not isinstance(item, Mapping):
            continue
        dimension = str(item.get("dimension") or "")
        level = str(item.get("level") or "")
        if not dimension or not level:
            continue
        findings.append(
            QualityFinding(
                dimension=dimension,
                level=level,
                reason_codes=_reason_tuple(item),
                evidence=dict(item.get("evidence") or {}),
            )
        )
    return findings


def _script_checkpoint(envelope: ClipMaterialization) -> Mapping[str, object]:
    checkpoints = envelope.component_checkpoints
    if not isinstance(checkpoints, Mapping):
        return {}
    entry = checkpoints.get("SCRIPT")
    return entry if isinstance(entry, Mapping) else {}


def _derive_evidence(
    session: Session,
    envelope: ClipMaterialization,
    inputs: MaterializationInputs,
    config: Any,
) -> tuple[dict[str, object], bool, AssessmentContext]:
    """Derive semantic gates from validated, dependency-bound accepted content.

    Returns ``(updates, missing, base)``. ``missing`` is true when required
    evidence is absent or incomplete (which must defer), while explicit failures
    are encoded in the updated flags (which reject). Gates are only ever
    *tightened*: a favorable absence of findings can never override adverse
    snapshot evidence, an UNKNOWN critical dimension is insufficient evidence
    rather than a favorable default, and a status label ("ACCEPTED"/"FULFILLED")
    is never trusted without the corresponding accepted content.
    """

    # Evidence from every authoritative source is merged: the dependency-bound
    # pre-render snapshot and the persisted row. Neither may silently replace the
    # other, so adverse evidence in either survives (provenance preserved, severity
    # never weakened).
    snapshot = _load_snapshot(envelope)
    prior = snapshot.extra_findings if snapshot is not None else ()
    findings = _merge_context_findings(prior, _stored_findings(envelope))
    levels = _finding_levels(findings)
    unmet = {
        code for finding in findings if finding.level == "UNMET" for code in finding.reason_codes
    }

    blocks_ok, blocks = _accepted_script_blocks(envelope, inputs, config)
    timeline_ok, hero_bound = _timeline_content(envelope)
    narration = _narration_references(envelope)
    slots = _slot_evidence(
        session, envelope, authored_blocks=blocks, narration=narration
    )
    narration_required = _narration_required(inputs) and inputs.profile.narration_allowed

    # An UNKNOWN required dimension, or UNKNOWN critical semantic evidence, is a
    # missing judgment: the mere presence of a dimension label never establishes a
    # satisfied gate. Missing required content also defers.
    unknown_required = any(levels.get(dim) == "UNKNOWN" for dim in _REBUILD_REQUIRED_DIMENSIONS)
    unknown_critical = any(
        levels.get(dim) == "UNKNOWN" for dim in _CRITICAL_SEMANTIC_DIMENSIONS
    )
    missing = (
        not blocks_ok
        or not timeline_ok
        or not hero_bound
        or slots.missing_resolution
        or (narration_required and not narration)
        or unknown_required
        or unknown_critical
    )

    if snapshot is not None:
        base = snapshot
    else:
        # A missing/invalid snapshot may be rebuilt only from complete
        # authoritative evidence; any gap defers rather than defaulting favorably.
        rebuilt = _rebuild_context(
            envelope,
            inputs,
            findings,
            complete=blocks_ok and timeline_ok and hero_bound and not slots.missing_resolution,
            required_slots_fulfilled=slots.required_slots_fulfilled,
            hero_bound=hero_bound,
            narration_present=bool(narration),
            narration_required=narration_required,
        )
        if rebuilt is None:
            missing = True
            base = AssessmentContext()
        else:
            base = rebuilt

    updates: dict[str, object] = {}
    updates["required_slots_fulfilled"] = (
        (not slots.explicit_failure)
        and slots.required_slots_fulfilled
        and base.required_slots_fulfilled
    )
    updates["required_narration_required"] = base.required_narration_required
    updates["narration_accepted"] = base.narration_accepted and (
        bool(narration) or not narration_required
    )
    updates["profile_mismatch_fields"] = _profile_mismatch_fields(inputs)
    updates["unsupported_essential"] = (
        base.unsupported_essential
        or REASON_UNSUPPORTED_ESSENTIAL_CLAIM in unmet
        or REASON_INVALID_GROUNDING_REFERENCE in unmet
    )
    updates["hero_protected"] = (
        base.hero_protected and hero_bound and REASON_HERO_DAMAGED not in unmet
    )
    updates["readable_copy"] = (
        base.readable_copy and REASON_UNREADABLE_REQUIRED_COPY not in unmet
    )
    updates["asset_rights_ok"] = (
        base.asset_rights_ok and REASON_UNSUITABLE_ASSET_RIGHTS not in unmet
    )
    updates["script_audio_match"] = (
        base.script_audio_match and REASON_SCRIPT_AUDIO_MISMATCH not in unmet
    )
    updates["final_clip_usable"] = (
        base.final_clip_usable and REASON_MANUAL_TRANSCRIPT_REVIEW not in unmet
    )
    timeline = envelope.timeline_payload if isinstance(envelope.timeline_payload, Mapping) else {}
    updates["timeline_conflicts"] = tuple(
        dict(item) for item in (timeline.get("conflicts") or []) if isinstance(item, Mapping)
    )
    updates["extra_findings"] = tuple(
        finding for finding in findings if finding.dimension != "SLOT_FULFILLMENT"
    )
    return updates, missing, base


def _rebuild_context(
    envelope: ClipMaterialization,
    inputs: MaterializationInputs,
    findings: list[QualityFinding],
    *,
    complete: bool,
    required_slots_fulfilled: bool,
    hero_bound: bool,
    narration_present: bool,
    narration_required: bool,
) -> AssessmentContext | None:
    """Rebuild the pre-render context from complete stored findings + content.

    Returns ``None`` when the evidence is incomplete (content gap, or a critical
    deterministic dimension is absent). Positive gates come from an actual
    SATISFIED/CONCERN finding, never from the mere absence of an UNMET finding.
    """

    if not complete:
        return None
    levels = _finding_levels(findings)
    if not _REBUILD_REQUIRED_DIMENSIONS.issubset(levels.keys()):
        return None
    # A required dimension at UNKNOWN is a missing judgment, not a satisfied gate;
    # so is UNKNOWN critical semantic evidence.
    if any(levels.get(dim) == "UNKNOWN" for dim in _REBUILD_REQUIRED_DIMENSIONS):
        return None
    if any(levels.get(dim) == "UNKNOWN" for dim in _CRITICAL_SEMANTIC_DIMENSIONS):
        return None
    grounding = levels.get("GROUNDING_VERIFICATION")
    return AssessmentContext(
        required_slots_fulfilled=required_slots_fulfilled,
        required_narration_required=narration_required,
        narration_accepted=narration_present or not narration_required,
        unsupported_essential=grounding == "UNMET",
        hero_protected=levels.get("HERO_PRESERVATION") != "UNMET" and hero_bound,
        readable_copy=levels.get("VISUAL_CAPTION_READABILITY") != "UNMET",
        asset_rights_ok=levels.get("PLATFORM_SUITABILITY") != "UNMET",
        script_audio_match=levels.get("FIDELITY_ATTRIBUTION", "SATISFIED") != "UNMET",
        final_clip_usable=True,
        critical_evidence_current=True,
        render_complete=False,
        repair_count=int(envelope.repair_count or 0),
    )


def _build_assessment_context(
    session: Session,
    envelope: ClipMaterialization,
    render: RenderExecution,
    *,
    inputs: MaterializationInputs,
    integrated: float,
    true_peak: float,
    config: Any,
) -> AssessmentContext:
    """Rebuild the assessment context from a validated snapshot + real evidence."""

    derived, missing, base = _derive_evidence(session, envelope, inputs, config)
    context = replace(base, **derived)
    if missing:
        context = replace(context, critical_fidelity_unknown=True)

    qc_status = render.qc_status.value if render.qc_status is not None else None
    return replace(
        context,
        render_complete=True,
        render_qc_status=qc_status,
        measured_loudness_lufs=integrated,
        measured_true_peak_dbtp=true_peak,
        # Live inputs were just re-derived and matched; this is a real value, not
        # a constant standing in for unknown evidence.
        critical_evidence_current=True,
    )


def _narration_required(inputs: MaterializationInputs) -> bool:
    need = str(inputs.narration.get("need", "NONE"))
    return need in {"RECOMMENDED", "REQUIRED"}


def _profile_mismatch_fields(inputs: MaterializationInputs) -> tuple[str, ...]:
    ok, mismatches = semantic_context_matches(inputs.profile, inputs.plan_target_context)
    return () if ok else mismatches


def _as_uuid(value: object) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str) and value:
        try:
            return uuid.UUID(value)
        except ValueError:
            return None
    return None


def _existing_semantic_outcome(envelope: ClipMaterialization, assessed: str) -> str:
    current = envelope.semantic_outcome.value if envelope.semantic_outcome else None
    return current or assessed


def _validate_render_binding(
    envelope: ClipMaterialization, render: RenderExecution
) -> FinalizationResult | None:
    """Validate the complete authoritative materialization -> render chain.

    An ``is_current`` flag alone is insufficient: the exact bound render id,
    purpose, request identity, materialization output fingerprint, and repair
    revision must all agree with the current envelope, or the render is refused.
    A late callback for a superseded (e.g. pre-repair) recipe must never finalize
    the repaired revision.
    """

    if render.materialization_id != envelope.id:
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_INVALID,))
    if render.artifact_purpose.value != TRANSFORMED_SHORT:
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_INVALID,))
    request: dict[str, object] = (
        render.request_payload if isinstance(render.request_payload, dict) else {}
    )
    expected = str(request.get("expected_output_fingerprint") or "")
    if not expected or expected != (envelope.output_fingerprint or ""):
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_STALE,))
    # The complete exact binding is required: a missing/malformed bound
    # materialization id, spec, authored payload, or repair revision is invalid,
    # never silently accepted because the output fingerprint happens to match.
    bound_id = str(request.get("materialization_id") or "")
    if not bound_id or bound_id != str(envelope.id):
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_INVALID,))
    raw_spec = request.get("spec")
    if not isinstance(raw_spec, dict):
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_INVALID,))
    raw_authored = raw_spec.get("authored")
    if not isinstance(raw_authored, dict):
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_INVALID,))
    revision = raw_authored.get("repair_revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_INVALID,))
    if revision != int(envelope.repair_count or 0):
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_STALE,))
    # NOTE: a missing ``qc_fingerprint`` is *successful-QC evidence*, not callback
    # identity. A pre-QC failed/blocked render has no QC fingerprint and must
    # still be finalizable (repair or truthful defer/reject); the QC requirement
    # is enforced on the COMPLETE branch below.
    return None


def _reason_tuple(entry: Mapping[str, object]) -> tuple[str, ...]:
    raw = entry.get("reason_codes")
    if not isinstance(raw, list):
        return ()
    return tuple(str(code) for code in raw)


def _output_duration(envelope: ClipMaterialization) -> float:
    metrics = envelope.metrics if isinstance(envelope.metrics, dict) else {}
    value = metrics.get("output_duration")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    timeline = envelope.timeline_payload if isinstance(envelope.timeline_payload, dict) else {}
    value = timeline.get("output_duration")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assessment_fingerprint(
    envelope: ClipMaterialization,
    render: RenderExecution,
    measured: Mapping[str, object],
    assessment: Any,
) -> str:
    payload = {
        "materialization_output_fingerprint": envelope.output_fingerprint,
        "repair_revision": int(envelope.repair_count or 0),
        "render_id": str(render.id),
        "render_output_fingerprint": render.output_fingerprint,
        "render_qc_fingerprint": render.qc_fingerprint,
        "final_integrated_lufs": measured.get("final_integrated_lufs"),
        "final_true_peak_dbtp": measured.get("final_true_peak_dbtp"),
        "outcome": assessment.outcome,
        "reason_codes": list(assessment.reason_codes),
        "policy_version": ASSESSMENT_POLICY_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _merge_findings(
    envelope: ClipMaterialization,
    render: RenderExecution,
    assessment: Any,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    def add(entry: Mapping[str, object]) -> None:
        dimension = str(entry.get("dimension") or "")
        level = str(entry.get("level") or "")
        reason_tuple = _reason_tuple(entry)
        key = (dimension, level, reason_tuple)
        if key in seen:
            return
        seen.add(key)
        findings.append(dict(entry))

    for item in envelope.quality_findings or []:
        if isinstance(item, Mapping):
            add(item)
    for finding in assessment.findings:
        add(finding.as_dict())
    add(
        {
            "dimension": "AUDIO_MASTERING_INTELLIGIBILITY",
            "level": "PASS" if render.qc_status is not None else "UNKNOWN",
            "reason_codes": list(render.reason_codes or []),
            "evidence": {
                "render_execution_id": str(render.id),
                "qc_status": render.qc_status.value if render.qc_status else None,
            },
        }
    )
    return findings


__all__ = [
    "ASSESSMENT_POLICY_VERSION",
    "FINALIZATION_CANCELLED",
    "FINALIZATION_COMPLETE",
    "FINALIZATION_DEFERRED",
    "FINALIZATION_REPAIR_QUEUED",
    "FINALIZATION_SUPERSEDED",
    "FinalizationResult",
    "dispatch_transformed_render",
    "finalize_materialization",
    "request_finalization_job",
]
