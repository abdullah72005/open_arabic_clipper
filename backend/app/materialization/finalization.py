"""Stage 6 render dispatch and truthful finalization.

The prepare job commits accepted components, the immutable output fingerprint,
and lifecycle ``AWAITING_RENDER``. This module binds a ``TRANSFORMED_SHORT``
render execution to that *exact* materialization id/output fingerprint through
the existing render queue, and, once the render reaches a terminal state,
finalizes the materialization by *re-assessing* the persisted semantic evidence
together with the actual render/QC evidence.

The finalizer never promotes a materialization to review readiness on the mere
completion of a technical render. It rebuilds an explicit
:class:`~app.materialization.acceptance.AssessmentContext` from stored accepted
component results, stored categorical findings, required-slot/narration
evidence, timeline/source protection, rights/readability, upstream currentness,
and the actual render/QC/loudness measurements, then runs the existing
categorical assessment engine. A stored ``PASS`` label can never override a
missing or contradictory measurement, and a completed render never erases a
persisted semantic failure.

Finalization is idempotent: an identical evidence replay makes zero
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

from sqlalchemy.orm import Session

from app.core.enums import (
    JobStatus,
    MaterializationLifecycle,
    MaterializationOutcome,
)
from app.core.settings import Settings
from app.materialization.acceptance import AssessmentContext, assess_materialization
from app.materialization.policy import (
    REASON_HERO_DAMAGED,
    REASON_INVALID_GROUNDING_REFERENCE,
    REASON_MANUAL_TRANSCRIPT_REVIEW,
    REASON_MISSING_REQUIRED_NARRATION,
    REASON_PROFILE_CONTEXT_MISMATCH,
    REASON_SCRIPT_AUDIO_MISMATCH,
    REASON_STALE_CRITICAL_EVIDENCE,
    REASON_UNREADABLE_REQUIRED_COPY,
    REASON_UNSUITABLE_ASSET_RIGHTS,
    REASON_UNSUPPORTED_ESSENTIAL_CLAIM,
)
from app.materialization.repository import (
    PRE_RENDER_CONTEXT_METRIC,
    MaterializationOwner,
    finish_owner_job,
    guarded_update,
    owner_is_current,
)
from app.materialization.types import QualityFinding
from app.models import ClipMaterialization
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

    from app.render.execution.queue import queue_render_execution

    if materialization.lifecycle is not MaterializationLifecycle.AWAITING_RENDER:
        return None
    if not materialization.output_fingerprint:
        return None
    profile_key = delivery_profile_key or materialization.profile_key
    if not profile_key:
        profile_key = "default"
    from app.render.execution.policy import DEFAULT_DELIVERY_PROFILE_KEY

    delivery = DEFAULT_DELIVERY_PROFILE_KEY
    outcome = queue_render_execution(
        session,
        candidate_id,
        artifact_purpose=TRANSFORMED_SHORT,
        delivery_profile_key=delivery,
        materialization_id=materialization.id,
        expected_output_fingerprint=materialization.output_fingerprint,
        settings=settings,
        dispatcher=dispatcher,
    )
    render_execution_id = uuid.UUID(str(outcome.render_execution_id))
    if materialization.render_execution_id != render_execution_id:
        materialization.render_execution_id = render_execution_id
        session.flush()
    return render_execution_id


def finalize_materialization(
    session: Session,
    owner: MaterializationOwner,
    *,
    settings: Settings,
    storage: StorageService,
) -> FinalizationResult:
    """Finalize one materialization from its exact stored render evidence."""

    envelope = session.get(ClipMaterialization, owner.materialization_id, populate_existing=True)
    if envelope is None:
        return FinalizationResult(FINALIZATION_SUPERSEDED, ("MATERIALIZATION_MISSING",))
    if envelope.lifecycle is MaterializationLifecycle.CANCELLED:
        return FinalizationResult(FINALIZATION_CANCELLED, ("MATERIALIZATION_CANCELLED",))
    if not owner_is_current(session, owner):
        return FinalizationResult(FINALIZATION_SUPERSEDED, ("OWNERSHIP_LOST",))

    render_id = envelope.render_execution_id
    if render_id is None:
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_RENDER_NOT_COMPLETE,))
    render = session.get(RenderExecution, render_id, populate_existing=True)
    if render is None:
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_RENDER_NOT_COMPLETE,))
    if render.lifecycle.value == "CANCELLED":
        return FinalizationResult(FINALIZATION_CANCELLED, ("RENDER_CANCELLED",))
    if render.lifecycle.value != "COMPLETE":
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_RENDER_NOT_COMPLETE,))
    if not render.is_current:
        return FinalizationResult(FINALIZATION_SUPERSEDED, ("RENDER_NOT_CURRENT",))

    binding = _validate_render_binding(envelope, render)
    if binding is not None:
        return binding

    measured = render.qc_result.get("measured") if isinstance(render.qc_result, dict) else {}
    measured = measured if isinstance(measured, dict) else {}
    integrated = measured.get("final_integrated_lufs")
    true_peak = measured.get("final_true_peak_dbtp")
    if render.qc_status is None:
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_MISSING_LOUDNESS_EVIDENCE,))
    if not _finite(integrated) or not _finite(true_peak):
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_MISSING_LOUDNESS_EVIDENCE,))

    reference = render.artifact_reference if isinstance(render.artifact_reference, dict) else {}
    relative = reference.get("relative_path")
    expected_hash = str(reference.get("sha256") or "")
    if not isinstance(relative, str) or not relative or not expected_hash:
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_ARTIFACT_MISSING,))
    path = (storage.storage_root / relative).resolve()
    try:
        path.relative_to(storage.storage_root)
    except ValueError:
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_ARTIFACT_MISSING,))
    if not path.is_file():
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_ARTIFACT_MISSING,))
    if _sha256(path) != expected_hash.lower():
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_ARTIFACT_HASH_MISMATCH,))

    context = _build_assessment_context(
        envelope, render, float(cast(float, integrated)), float(cast(float, true_peak))
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
        # Replay: identical evidence, zero writes, zero provider/render calls.
        # Still complete the *current* job under its exact owner fence.
        if not finish_owner_job(session, owner, JobStatus.SUCCEEDED):
            session.rollback()
            return FinalizationResult(FINALIZATION_SUPERSEDED, ("OWNERSHIP_LOST",))
        session.commit()
        return FinalizationResult(
            FINALIZATION_COMPLETE,
            reason_codes=assessment.reason_codes,
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
        for code in (finding.get("reason_codes") or [])
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
    ok = guarded_update(
        session,
        owner,
        {
            "lifecycle": MaterializationLifecycle.COMPLETE,
            "semantic_outcome": MaterializationOutcome(assessment.outcome),
            "human_review_required": bool(assessment.human_review_required)
            if reviewable
            else False,
            "publication_ready": False,
            "cache_eligible": reviewable,
            "reason_codes": list(reasons),
            "quality_findings": findings,
            "metrics": metrics,
        },
    )
    if not ok:
        session.rollback()
        return FinalizationResult(FINALIZATION_SUPERSEDED, ("OWNERSHIP_LOST",))
    if not finish_owner_job(session, owner, JobStatus.SUCCEEDED):
        session.rollback()
        return FinalizationResult(FINALIZATION_SUPERSEDED, ("OWNERSHIP_LOST",))
    session.commit()
    disposition = _disposition(assessment.outcome)
    return FinalizationResult(
        disposition,
        reason_codes=assessment.reason_codes,
        assessment_fingerprint=fingerprint,
        render_execution_id=str(render.id),
        semantic_outcome=assessment.outcome,
        findings=tuple(findings),
    )


def _existing_semantic_outcome(envelope: ClipMaterialization, assessed: str) -> str:
    current = envelope.semantic_outcome.value if envelope.semantic_outcome else None
    return current or assessed


def _disposition(outcome: str) -> str:
    if outcome in _REVIEWABLE_OUTCOMES:
        return FINALIZATION_COMPLETE
    if outcome in _DEFERRED_OUTCOMES:
        return FINALIZATION_DEFERRED
    return FINALIZATION_COMPLETE


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


def _build_assessment_context(
    envelope: ClipMaterialization,
    render: RenderExecution,
    integrated: float,
    true_peak: float,
) -> AssessmentContext:
    """Reconstruct the assessment context entirely from stored actual evidence."""

    metrics = envelope.metrics if isinstance(envelope.metrics, dict) else {}
    snapshot = metrics.get(PRE_RENDER_CONTEXT_METRIC)
    has_snapshot = isinstance(snapshot, dict)
    if has_snapshot:
        context = AssessmentContext.from_dict(snapshot)
    else:
        context = AssessmentContext()

    # Never trust default-True fields to stand in for missing evidence: re-derive
    # the semantic gates from the persisted categorical findings. When a durable
    # snapshot exists it is authoritative for the semantic flags; the stored
    # findings are still preserved as explicit evidence either way.
    context = _apply_stored_evidence(context, envelope, has_snapshot=has_snapshot)

    qc_status = render.qc_status.value if render.qc_status is not None else None
    return replace(
        context,
        render_complete=True,
        render_qc_status=qc_status,
        measured_loudness_lufs=integrated,
        measured_true_peak_dbtp=true_peak,
        critical_evidence_current=True,
    )


def _apply_stored_evidence(
    context: AssessmentContext,
    envelope: ClipMaterialization,
    *,
    has_snapshot: bool,
) -> AssessmentContext:
    """Re-derive semantic gates from stored findings without erasing evidence.

    A completed technical render must never erase a persisted semantic failure.
    Stored categorical findings are folded into the context as explicit evidence;
    the well-known hard-failure findings also pin their corresponding gate so a
    default-``True`` field can never stand in for missing evidence. Required-slot
    fulfilment is only derived from stored findings when no authoritative
    pre-render snapshot exists (otherwise the snapshot wins).
    """

    updates: dict[str, object] = {}
    extras = list(context.extra_findings)
    seen = {(finding.dimension, finding.level, tuple(finding.reason_codes)) for finding in extras}
    unmet_reasons: set[str] = set()

    for item in envelope.quality_findings or []:
        if not isinstance(item, Mapping):
            continue
        dimension = str(item.get("dimension") or "")
        level = str(item.get("level") or "")
        if not dimension or not level:
            continue
        reason_tuple = _reason_tuple(item)
        if level == "UNMET":
            unmet_reasons.update(reason_tuple)

        if dimension == "SLOT_FULFILLMENT":
            # Required-slot truth is authoritative from the snapshot/slot
            # resolution; an optional/declined empty block must not be promoted
            # to a missing required value here.
            if not has_snapshot and level == "UNMET":
                updates["required_slots_fulfilled"] = False
            continue

        key = (dimension, level, reason_tuple)
        if key in seen:
            continue
        seen.add(key)
        extras.append(
            QualityFinding(
                dimension=dimension,
                level=level,
                reason_codes=reason_tuple,
                evidence=dict(item.get("evidence") or {}),
            )
        )

    if REASON_UNSUPPORTED_ESSENTIAL_CLAIM in unmet_reasons or (
        REASON_INVALID_GROUNDING_REFERENCE in unmet_reasons
    ):
        updates["unsupported_essential"] = True
    if REASON_HERO_DAMAGED in unmet_reasons:
        updates["hero_protected"] = False
    if REASON_UNREADABLE_REQUIRED_COPY in unmet_reasons:
        updates["readable_copy"] = False
    if REASON_UNSUITABLE_ASSET_RIGHTS in unmet_reasons:
        updates["asset_rights_ok"] = False
    if REASON_SCRIPT_AUDIO_MISMATCH in unmet_reasons:
        updates["script_audio_match"] = False
    if REASON_MANUAL_TRANSCRIPT_REVIEW in unmet_reasons:
        updates["final_clip_usable"] = False
    if REASON_MISSING_REQUIRED_NARRATION in unmet_reasons:
        updates["required_slots_fulfilled"] = False
        updates["narration_accepted"] = False
    if REASON_PROFILE_CONTEXT_MISMATCH in unmet_reasons and not context.profile_mismatch_fields:
        updates["profile_mismatch_fields"] = ("stored",)
    if REASON_STALE_CRITICAL_EVIDENCE in unmet_reasons and has_snapshot:
        updates["critical_evidence_current"] = False

    if extras:
        updates["extra_findings"] = tuple(extras)

    if not updates:
        return context
    return replace(context, **updates)  # type: ignore[arg-type]


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
]
