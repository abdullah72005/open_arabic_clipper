"""Stage 6 render dispatch and idempotent finalization.

The prepare job commits accepted components, the immutable output fingerprint,
and lifecycle ``AWAITING_RENDER``. This module binds a ``TRANSFORMED_SHORT``
render execution to that *exact* materialization id/output fingerprint through
the existing render queue, and finalizes the materialization from the stored
render evidence once the render reaches a terminal state.

Finalization is a no-op on replay: the assessment fingerprint is derived from the
materialization output fingerprint plus the exact render output/QC identity, so a
re-delivered finalizer job makes zero provider/render calls and writes nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.core.enums import (
    JobStatus,
    MaterializationLifecycle,
    MaterializationOutcome,
)
from app.core.settings import Settings
from app.materialization.repository import (
    MaterializationOwner,
    finish_owner_job,
    guarded_update,
    owner_is_current,
)
from app.models import ClipMaterialization
from app.models.render_execution import RenderExecution
from app.render.execution.policy import TRANSFORMED_SHORT
from app.services.storage import StorageService

FINALIZATION_COMPLETE = "COMPLETE"
FINALIZATION_REPAIR_QUEUED = "REPAIR_QUEUED"
FINALIZATION_DEFERRED = "DEFERRED"
FINALIZATION_SUPERSEDED = "SUPERSEDED"
FINALIZATION_CANCELLED = "CANCELLED"

REASON_RENDER_NOT_COMPLETE = "RENDER_NOT_COMPLETE"
REASON_RENDER_BINDING_INVALID = "RENDER_BINDING_INVALID"
REASON_MISSING_LOUDNESS_EVIDENCE = "MISSING_LOUDNESS_EVIDENCE"
REASON_ARTIFACT_MISSING = "ARTIFACT_MISSING"
REASON_ARTIFACT_HASH_MISMATCH = "ARTIFACT_HASH_MISMATCH"
ASSESSMENT_POLICY_VERSION = "stage6-assessment-v1"


@dataclass(frozen=True)
class FinalizationResult:
    """Truthful Stage 6 finalization outcome (never a publication permission)."""

    outcome: str
    reason_codes: tuple[str, ...] = ()
    assessment_fingerprint: str = ""
    render_execution_id: str | None = None
    reused: bool = False
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
    if render.materialization_id != envelope.id:
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_INVALID,))
    if render.artifact_purpose.value != TRANSFORMED_SHORT:
        return FinalizationResult(FINALIZATION_SUPERSEDED, (REASON_RENDER_BINDING_INVALID,))
    if render.lifecycle.value == "CANCELLED":
        return FinalizationResult(FINALIZATION_CANCELLED, ("RENDER_CANCELLED",))
    if render.lifecycle.value != "COMPLETE":
        return FinalizationResult(FINALIZATION_DEFERRED, (REASON_RENDER_NOT_COMPLETE,))
    if not render.is_current:
        return FinalizationResult(FINALIZATION_SUPERSEDED, ("RENDER_NOT_CURRENT",))

    measured = render.qc_result.get("measured") if isinstance(render.qc_result, dict) else {}
    measured = measured if isinstance(measured, dict) else {}
    integrated = measured.get("final_integrated_lufs")
    true_peak = measured.get("final_true_peak_dbtp")
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

    fingerprint = _assessment_fingerprint(envelope, render, measured)
    existing = (
        envelope.metrics.get("assessment_fingerprint")
        if isinstance(envelope.metrics, dict)
        else None
    )
    if envelope.lifecycle is MaterializationLifecycle.COMPLETE and existing == fingerprint:
        # Replay: identical evidence, zero writes, zero calls.
        return FinalizationResult(
            FINALIZATION_COMPLETE,
            assessment_fingerprint=fingerprint,
            render_execution_id=str(render.id),
            reused=True,
        )

    metrics = dict(envelope.metrics or {})
    metrics["assessment_fingerprint"] = fingerprint
    metrics["final_integrated_lufs"] = integrated
    metrics["final_true_peak_dbtp"] = true_peak
    metrics["render_output_fingerprint"] = render.output_fingerprint
    findings = _merge_findings(envelope, render)
    ok = guarded_update(
        session,
        owner,
        {
            "lifecycle": MaterializationLifecycle.COMPLETE,
            "semantic_outcome": MaterializationOutcome.READY_FOR_HUMAN_REVIEW,
            "human_review_required": True,
            "publication_ready": False,
            "cache_eligible": True,
            "reason_codes": [],
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
    return FinalizationResult(
        FINALIZATION_COMPLETE,
        assessment_fingerprint=fingerprint,
        render_execution_id=str(render.id),
        findings=tuple(findings),
    )


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
    measured: dict[str, object],
) -> str:
    payload = {
        "materialization_output_fingerprint": envelope.output_fingerprint,
        "render_output_fingerprint": render.output_fingerprint,
        "render_qc_fingerprint": render.qc_fingerprint,
        "final_integrated_lufs": measured.get("final_integrated_lufs"),
        "final_true_peak_dbtp": measured.get("final_true_peak_dbtp"),
        "policy_version": ASSESSMENT_POLICY_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _merge_findings(
    envelope: ClipMaterialization, render: RenderExecution
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for item in envelope.quality_findings or []:
        if isinstance(item, dict):
            findings.append(dict(item))
    findings.append(
        {
            "dimension": "technical_render",
            "level": "PASS" if render.qc_status is not None else "CONCERN",
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
