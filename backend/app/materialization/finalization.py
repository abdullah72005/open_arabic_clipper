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
from app.materialization.fingerprints import input_fingerprint
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
    finish_owner_job,
    guarded_update,
    owner_is_current,
)
from app.materialization.types import QualityFinding
from app.models import ClipMaterialization, RenderContract
from app.models.render_execution import RenderExecution
from app.render.execution.policy import (
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
            session.rollback()
            return FinalizationResult(FINALIZATION_SUPERSEDED, ("OWNERSHIP_LOST",))
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
        return FinalizationResult(FINALIZATION_SUPERSEDED, ("OWNERSHIP_LOST",))

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
    if render.lifecycle.value != "COMPLETE":
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
        )
    if not _finite(integrated) or not _finite(true_peak):
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_MISSING_LOUDNESS_EVIDENCE,),
        )

    reference = render.artifact_reference if isinstance(render.artifact_reference, dict) else {}
    relative = reference.get("relative_path")
    expected_hash = str(reference.get("sha256") or "")
    if not isinstance(relative, str) or not relative or not expected_hash:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_ARTIFACT_MISSING,),
        )
    path = (storage.storage_root / relative).resolve()
    try:
        path.relative_to(storage.storage_root)
    except ValueError:
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_ARTIFACT_MISSING,),
        )
    if not path.is_file():
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_ARTIFACT_MISSING,),
        )
    if _sha256(path) != expected_hash.lower():
        return _disposition(
            session,
            owner,
            outcome=FINALIZATION_DEFERRED,
            reason_codes=(REASON_ARTIFACT_HASH_MISMATCH,),
        )

    context = _build_assessment_context(
        session,
        envelope,
        render,
        inputs=inputs,
        integrated=float(cast(float, integrated)),
        true_peak=float(cast(float, true_peak)),
    )
    config = settings.stage6_config()
    assessment = assess_materialization(
        context, output_duration=_output_duration(envelope), config=config
    )

    fingerprint = _assessment_fingerprint(envelope, render, measured, assessment)
    existing = (
        envelope.metrics.get("assessment_fingerprint")
        if isinstance(envelope.metrics, dict)
        else None
    )
    if envelope.lifecycle is MaterializationLifecycle.COMPLETE and existing == fingerprint:
        # Replay: identical evidence, zero provider/render calls, no envelope
        # mutation. Still complete the *current* job under its exact owner fence.
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
    return _disposition(
        session,
        owner,
        outcome=_disposition_outcome(assessment.outcome),
        reason_codes=reasons,
        job_status=JobStatus.SUCCEEDED,
        envelope_values={
            "lifecycle": MaterializationLifecycle.COMPLETE,
            "semantic_outcome": MaterializationOutcome(assessment.outcome),
            "human_review_required": bool(assessment.human_review_required)
            if reviewable
            else False,
            "publication_ready": False,
            "cache_eligible": reviewable,
            "quality_findings": findings,
            "metrics": metrics,
        },
        assessment_fingerprint=fingerprint,
        render_execution_id=str(render.id),
        semantic_outcome=assessment.outcome,
        findings=tuple(findings),
    )


def _disposition_outcome(assessment_outcome: str) -> str:
    if assessment_outcome in _REVIEWABLE_OUTCOMES:
        return FINALIZATION_COMPLETE
    if assessment_outcome in _DEFERRED_OUTCOMES:
        return FINALIZATION_DEFERRED
    return FINALIZATION_COMPLETE


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


def _slot_evidence(session: Session, envelope: ClipMaterialization) -> _SlotEvidence:
    """Derive required-slot fulfilment from the contract + persisted resolutions.

    An empty resolution list never means "no requirements": a required contract
    slot without a persisted FULFILLED resolution is missing evidence, while an
    explicit DEFERRED/BLOCKED required resolution is a failure.
    """

    contract = (
        session.get(RenderContract, envelope.render_contract_id)
        if envelope.render_contract_id
        else None
    )
    rows = [
        dict(row)
        for row in (envelope.slot_resolutions or [])
        if isinstance(row, Mapping)
    ]
    fulfilled_pairs = {
        (str(row.get("slot_id")), str(row.get("slot_kind", "")))
        for row in rows
        if str(row.get("resolution")) == "FULFILLED"
    }
    failed_pairs = {
        (str(row.get("slot_id")), str(row.get("slot_kind", "")))
        for row in rows
        if str(row.get("resolution")) in {"DEFERRED", "BLOCKED"}
    }
    explicit_failure = False
    missing = contract is None
    for slot in _required_contract_slots(contract):
        slot_id = str(slot.get("slot_id"))
        slot_kind = str(slot.get("slot_kind", ""))
        if (slot_id, slot_kind) in failed_pairs:
            explicit_failure = True
        elif (slot_id, slot_kind) not in fulfilled_pairs:
            missing = True
    for row in rows:
        if bool(row.get("required", False)) and str(row.get("resolution")) in {
            "DEFERRED",
            "BLOCKED",
        }:
            explicit_failure = True
    return _SlotEvidence(
        required_slots_fulfilled=not explicit_failure,
        explicit_failure=explicit_failure,
        missing_resolution=missing,
    )


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


def _narration_blocks(envelope: ClipMaterialization) -> set[int]:
    blocks: set[int] = set()
    for item in envelope.narration_references or []:
        if not isinstance(item, Mapping):
            continue
        index = item.get("block_index")
        if isinstance(index, int) and not isinstance(index, bool):
            blocks.add(index)
    return blocks


def _script_checkpoint(envelope: ClipMaterialization) -> Mapping[str, object]:
    checkpoints = envelope.component_checkpoints
    if not isinstance(checkpoints, Mapping):
        return {}
    entry = checkpoints.get("SCRIPT")
    return entry if isinstance(entry, Mapping) else {}


def _derive_evidence(
    session: Session, envelope: ClipMaterialization, inputs: MaterializationInputs
) -> tuple[dict[str, object], bool]:
    """Derive semantic gates from the persisted authoritative evidence.

    Returns ``(updates, missing)``. ``missing`` is true when required evidence is
    absent (which must defer), while explicit failures are encoded in the updated
    flags (which reject). Findings are preserved as explicit evidence.
    """

    updates: dict[str, object] = {}
    missing = False
    findings = _stored_findings(envelope)
    unmet = {
        code for finding in findings if finding.level == "UNMET" for code in finding.reason_codes
    }

    slots = _slot_evidence(session, envelope)
    updates["required_slots_fulfilled"] = slots.required_slots_fulfilled
    if slots.missing_resolution:
        missing = True

    script = _script_checkpoint(envelope)
    if not script or script.get("status") != "ACCEPTED":
        missing = True
    timeline = envelope.timeline_payload if isinstance(envelope.timeline_payload, Mapping) else {}
    segments = timeline.get("segments")
    if not isinstance(segments, list) or not segments:
        missing = True

    narration_required = _narration_required(inputs)
    narration_gate = narration_required and inputs.profile.narration_allowed
    updates["required_narration_required"] = narration_gate
    updates["narration_accepted"] = bool(_narration_blocks(envelope)) or not narration_gate
    if narration_gate and not _narration_blocks(envelope):
        missing = True

    updates["profile_mismatch_fields"] = _profile_mismatch_fields(inputs)
    updates["unsupported_essential"] = REASON_UNSUPPORTED_ESSENTIAL_CLAIM in unmet
    updates["hero_protected"] = _hero_protected(timeline) and REASON_HERO_DAMAGED not in unmet
    updates["readable_copy"] = REASON_UNREADABLE_REQUIRED_COPY not in unmet
    updates["asset_rights_ok"] = REASON_UNSUITABLE_ASSET_RIGHTS not in unmet
    updates["script_audio_match"] = REASON_SCRIPT_AUDIO_MISMATCH not in unmet
    updates["final_clip_usable"] = REASON_MANUAL_TRANSCRIPT_REVIEW not in unmet
    if REASON_INVALID_GROUNDING_REFERENCE in unmet:
        updates["unsupported_essential"] = True
    updates["timeline_conflicts"] = tuple(
        dict(item)
        for item in (timeline.get("conflicts") or [])
        if isinstance(item, Mapping)
    )
    updates["extra_findings"] = tuple(
        finding for finding in findings if finding.dimension != "SLOT_FULFILLMENT"
    )
    return updates, missing


def _build_assessment_context(
    session: Session,
    envelope: ClipMaterialization,
    render: RenderExecution,
    *,
    inputs: MaterializationInputs,
    integrated: float,
    true_peak: float,
) -> AssessmentContext:
    """Rebuild the assessment context from a validated snapshot + real evidence."""

    snapshot = _load_snapshot(envelope)
    derived, missing = _derive_evidence(session, envelope, inputs)
    base = snapshot if snapshot is not None else AssessmentContext()
    # Authoritative persisted evidence always overrides the snapshot: a stored
    # snapshot may never stand in for a missing/failed required slot or finding.
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


def _hero_protected(timeline: Mapping[str, object]) -> bool:
    segments = timeline.get("segments")
    if not isinstance(segments, list):
        return False
    return any(
        isinstance(segment, Mapping) and bool(segment.get("is_hero")) for segment in segments
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
    bound_id = str(request.get("materialization_id") or "")
    if bound_id and bound_id != str(envelope.id):
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_INVALID,))
    raw_spec = request.get("spec")
    spec: dict[str, object] = raw_spec if isinstance(raw_spec, dict) else {}
    raw_authored = spec.get("authored")
    authored: dict[str, object] = raw_authored if isinstance(raw_authored, dict) else {}
    revision = authored.get("repair_revision")
    if isinstance(revision, int) and not isinstance(revision, bool):
        if revision != int(envelope.repair_count or 0):
            return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_STALE,))
    if not render.qc_fingerprint:
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_MISSING_LOUDNESS_EVIDENCE,))
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
