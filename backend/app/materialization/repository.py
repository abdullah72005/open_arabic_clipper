"""Durable Stage 6 envelope ownership, attempt ledger, and component checkpoints.

This module owns the short, fenced database transactions that the Stage 6
executor and queue use. It never performs provider, render, network, or FFmpeg
work: every function is a bounded database operation guarded by the paired
job/envelope ownership predicates. Two local ownership tokens exist and are
never compared across jobs or envelopes:

* ``processing_jobs.claim_version`` advances whenever *that job* is claimed.
* ``clip_materializations.claim_version`` advances whenever *that envelope* is
  claimed.

An owner is valid only when all four fields of :class:`MaterializationOwner`
match, the job is RUNNING, the envelope's ``active_job_id`` equals the job, the
job's ``materialization_id`` equals the envelope, and the envelope is current.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.enums import (
    JobKind,
    JobStatus,
    MaterializationLifecycle,
    MaterializationOutcome,
    MaterializationPriority,
)
from app.materialization.fingerprints import input_fingerprint
from app.materialization.inputs import MaterializationInputs
from app.models import ClipMaterialization, ProcessingJob

ATTEMPT_LEDGER_VERSION = "stage6-attempts-v2"

#: Durable metric key holding the serialized pre-render :class:`AssessmentContext`
#: so the finalizer can reconstruct the semantic gates from stored evidence.
PRE_RENDER_CONTEXT_METRIC = "pre_render_assessment_context"

#: Version of the durable pre-render snapshot. The snapshot is a versioned,
#: dependency-bound checkpoint: it records the exact materialization output
#: fingerprint it was computed for, so a stale/mismatched snapshot is never
#: deserialized into favorable default gates.
PRE_RENDER_CONTEXT_VERSION = "stage6-prerender-context-v2"

CHECKPOINT_PENDING = "PENDING"
CHECKPOINT_ACCEPTED = "ACCEPTED"
CHECKPOINT_DEFERRED = "DEFERRED"
CHECKPOINT_REJECTED = "REJECTED"
CHECKPOINT_STATUSES = frozenset(
    {CHECKPOINT_PENDING, CHECKPOINT_ACCEPTED, CHECKPOINT_DEFERRED, CHECKPOINT_REJECTED}
)

ATTEMPT_RESERVED = "RESERVED"
ATTEMPT_RESPONDED = "RESPONDED"
ATTEMPT_FAILED_KNOWN = "FAILED_KNOWN"
ATTEMPT_UNCERTAIN = "UNCERTAIN"
ATTEMPT_STATUSES = frozenset(
    {ATTEMPT_RESERVED, ATTEMPT_RESPONDED, ATTEMPT_FAILED_KNOWN, ATTEMPT_UNCERTAIN}
)


class MaterializationOwnershipError(RuntimeError):
    """An authoritative envelope update was attempted without current ownership."""


@dataclass(frozen=True)
class MaterializationOwner:
    """The four-field paired job/envelope ownership token.

    ``job_claim_version`` and ``materialization_claim_version`` are *local* to
    their own row and are never compared to another row's counter.
    """

    materialization_id: uuid.UUID
    job_id: uuid.UUID
    job_claim_version: int
    materialization_claim_version: int


@dataclass
class PreparedMaterialization:
    """Result of executing/replaying all pre-render Stage 6 components.

    Populated by ``components.prepare_materialization``. The fields mirror the
    accepted persisted evidence; ``output_fingerprint`` is empty until a complete
    executable recipe exists.
    """

    lifecycle: MaterializationLifecycle = MaterializationLifecycle.COMPLETE
    semantic_outcome: str | None = None
    human_review_required: bool = False
    repair_count: int = 0
    reason_codes: tuple[str, ...] = ()
    component_checkpoints: dict[str, object] = field(default_factory=dict)
    authored_blocks: list[dict[str, object]] = field(default_factory=list)
    claim_ledger: list[dict[str, object]] = field(default_factory=list)
    slot_resolutions: list[dict[str, object]] = field(default_factory=list)
    narration_references: list[dict[str, object]] = field(default_factory=list)
    asset_references: list[dict[str, object]] = field(default_factory=list)
    timeline_payload: dict[str, object] = field(default_factory=dict)
    audio_recipe: dict[str, object] = field(default_factory=dict)
    quality_findings: list[dict[str, object]] = field(default_factory=list)
    provider_identity: dict[str, object] = field(default_factory=dict)
    upstream_snapshot: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)
    output_fingerprint: str = ""


# ---------------------------------------------------------------------------
# Ownership predicates and guarded updates
# ---------------------------------------------------------------------------


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _job_fence(owner: MaterializationOwner) -> Any:
    """A correlated EXISTS predicate matching the owner's exact RUNNING job."""

    return (
        select(ProcessingJob.id)
        .where(
            ProcessingJob.id == owner.job_id,
            ProcessingJob.kind == JobKind.MATERIALIZATION,
            ProcessingJob.status == JobStatus.RUNNING,
            ProcessingJob.claim_version == owner.job_claim_version,
            ProcessingJob.materialization_id == owner.materialization_id,
        )
        .exists()
    )


def _owner_where(owner: MaterializationOwner) -> tuple[Any, ...]:
    """The complete ownership predicate: envelope token + current + job fence."""

    return (
        ClipMaterialization.id == owner.materialization_id,
        ClipMaterialization.is_current.is_(True),
        ClipMaterialization.active_job_id == owner.job_id,
        ClipMaterialization.claim_version == owner.materialization_claim_version,
        _job_fence(owner),
    )


def guarded_update(
    session: Session, owner: MaterializationOwner, values: Mapping[str, Any]
) -> bool:
    """Apply an authoritative envelope update under the full ownership predicate.

    Returns ``True`` only when exactly one current owned row was updated. The
    caller must not commit authoritative state when this returns ``False``.
    """

    result = session.execute(
        update(ClipMaterialization)
        .where(*_owner_where(owner))
        .values(**dict(values))
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def owner_is_current(session: Session, owner: MaterializationOwner) -> bool:
    """Fresh scalar read proving the owner is still valid (no identity map)."""

    row = session.execute(select(ClipMaterialization.id).where(*_owner_where(owner))).first()
    return row is not None


# ---------------------------------------------------------------------------
# Envelope lifecycle
# ---------------------------------------------------------------------------


def get_or_create_envelope(
    session: Session,
    inputs: MaterializationInputs,
    request_identity: Mapping[str, object],
) -> ClipMaterialization:
    """Return the one envelope for this exact input generation, creating it once.

    The row is created *before* any provider work. A matching row is reused; a
    genuinely changed input demotes the previous current row only when this is a
    valid new generation. The caller must already hold the candidate scope lock.
    """

    candidate_uuid = _as_uuid(inputs.candidate_id)
    profile_key = inputs.profile.profile_key
    input_fp = input_fingerprint(inputs)

    existing = session.scalars(
        select(ClipMaterialization)
        .where(
            ClipMaterialization.clip_candidate_id == candidate_uuid,
            ClipMaterialization.profile_key == profile_key,
            ClipMaterialization.input_fingerprint == input_fp,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    ).first()

    # Demote every *other* current row in this scope before reactivating or
    # creating the target generation, so the scoped partial unique index is never
    # transiently violated (historical CANDIDATE -> FINAL_CLIP -> CANDIDATE
    # reactivation) and history/checkpoints/budgets are preserved in place.
    demote = update(ClipMaterialization).where(
        ClipMaterialization.clip_candidate_id == candidate_uuid,
        ClipMaterialization.profile_key == profile_key,
        ClipMaterialization.is_current.is_(True),
    )
    if existing is not None:
        demote = demote.where(ClipMaterialization.id != existing.id)
    session.execute(demote.values(is_current=False).execution_options(synchronize_session=False))

    if existing is not None:
        if not existing.is_current:
            existing.is_current = True
            session.flush()
        return existing

    row = ClipMaterialization(
        source_video_id=_as_uuid(inputs.source_id),
        clip_candidate_id=candidate_uuid,
        profile_key=profile_key,
        priority=MaterializationPriority(inputs.priority),
        lifecycle=MaterializationLifecycle.QUEUED,
        is_current=True,
    )
    _apply_identity(row, inputs, input_fp, request_identity)
    session.add(row)
    session.flush()
    return row


def _apply_identity(
    row: ClipMaterialization,
    inputs: MaterializationInputs,
    input_fp: str,
    request_identity: Mapping[str, object],
) -> None:
    row.channel_production_profile_id = _optional_uuid(inputs.profile.profile_id)
    row.selection_id = _optional_uuid(inputs.selection_id)
    row.selected_plan_id = _optional_uuid(inputs.plan_id)
    row.refinement_id = _optional_uuid(inputs.final_clip.refinement_id)
    row.render_contract_id = _optional_uuid(inputs.contract_id)
    row.visual_composition_plan_id = _optional_uuid(inputs.visual_plan.visual_plan_id)
    row.input_fingerprint = input_fp
    row.request_payload = {**dict(request_identity), "input_fingerprint": input_fp}
    row.upstream_fingerprints = {
        "selection_output_fingerprint": inputs.selection_output_fingerprint,
        "selected_plan_fingerprint": inputs.selected_plan_fingerprint,
        "contract_output_fingerprint": inputs.contract_output_fingerprint,
        "final_clip_output_fingerprint": inputs.final_clip.output_fingerprint,
        "visual_plan_output_fingerprint": inputs.visual_plan.output_fingerprint,
        "source_media_fingerprint": inputs.visual_plan.source_media_fingerprint,
    }


def claim_materialization(
    session: Session,
    materialization_id: uuid.UUID | str,
    job_id: uuid.UUID | str,
    *,
    expected_job_claim_version: int,
) -> MaterializationOwner | None:
    """Atomically claim one envelope for a RUNNING MATERIALIZATION job.

    Advances only the envelope's local ``claim_version`` and binds the job's
    ``materialization_id`` to the envelope. The caller's *expected* job claim
    token is verified against a fresh row read (never the stale identity map,
    never whichever token happens to be stored). Returns ``None`` when the job is
    not RUNNING, is the wrong kind, the expected token is stale, or the envelope
    is not the one the job already owns via the queue CAS.
    """

    envelope_id = _as_uuid(materialization_id)
    job_uuid = _as_uuid(job_id)
    job = _fresh_get(session, ProcessingJob, job_uuid)
    if job is None or job.kind is not JobKind.MATERIALIZATION:
        return None
    if job.status is not JobStatus.RUNNING:
        return None
    if int(job.claim_version or 0) != int(expected_job_claim_version):
        return None

    result = session.execute(
        update(ClipMaterialization)
        .where(
            ClipMaterialization.id == envelope_id,
            ClipMaterialization.is_current.is_(True),
            ClipMaterialization.active_job_id == job_uuid,
        )
        .values(
            claim_version=ClipMaterialization.claim_version + 1,
            active_job_id=job_uuid,
        )
        .execution_options(synchronize_session=False)
    )
    if int(result.rowcount) != 1:
        session.rollback()
        return None

    session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == job_uuid)
        .values(materialization_id=envelope_id)
        .execution_options(synchronize_session=False)
    )
    # Fresh scalar reads: the identity map still holds the pre-increment values
    # because the updates ran with synchronize_session=False.
    envelope_claim = session.scalar(
        select(ClipMaterialization.claim_version).where(ClipMaterialization.id == envelope_id)
    )
    if envelope_claim is None:
        session.rollback()
        return None
    owner = MaterializationOwner(
        materialization_id=envelope_id,
        job_id=job_uuid,
        job_claim_version=int(expected_job_claim_version),
        materialization_claim_version=int(envelope_claim),
    )
    session.commit()
    return owner


def set_lifecycle(
    session: Session, owner: MaterializationOwner, lifecycle: MaterializationLifecycle
) -> bool:
    return guarded_update(session, owner, {"lifecycle": lifecycle})


def bind_active_job(
    session: Session, materialization_id: uuid.UUID | str, job_id: uuid.UUID | None
) -> bool:
    """Queue-time CAS binding the envelope's active job (NULL -> job)."""

    result = session.execute(
        update(ClipMaterialization)
        .where(
            ClipMaterialization.id == _as_uuid(materialization_id),
            ClipMaterialization.is_current.is_(True),
            ClipMaterialization.active_job_id.is_(None),
        )
        .values(active_job_id=job_id)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


# ---------------------------------------------------------------------------
# Durable attempt ledger (stage6-attempts-v2)
# ---------------------------------------------------------------------------


def new_ledger() -> dict[str, object]:
    return {"version": ATTEMPT_LEDGER_VERSION, "entries": []}


def ledger_entries(ledger: Mapping[str, object] | None) -> list[dict[str, object]]:
    if not isinstance(ledger, Mapping):
        return []
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        return []
    return [dict(entry) for entry in entries if isinstance(entry, Mapping)]


def ledger_counts(ledger: Mapping[str, object] | None) -> dict[str, int]:
    entries = ledger_entries(ledger)
    return {
        "reserved": sum(1 for e in entries if e.get("status") == ATTEMPT_RESERVED),
        "responded": sum(1 for e in entries if e.get("status") == ATTEMPT_RESPONDED),
        "uncertain": sum(1 for e in entries if e.get("status") == ATTEMPT_UNCERTAIN),
        "failed_known": sum(1 for e in entries if e.get("status") == ATTEMPT_FAILED_KNOWN),
        "semantic": sum(1 for e in entries if e.get("kind") in {"semantic", "critique"}),
        "strong": sum(1 for e in entries if e.get("tier") == "STRONG"),
        "synthesis": sum(1 for e in entries if e.get("kind") == "synthesis"),
        "total": len(entries),
    }


def _lock_envelope(session: Session, envelope_id: uuid.UUID) -> ClipMaterialization | None:
    """Load the envelope's *current* row values under a row lock.

    The application session uses ``autoflush=False``/``expire_on_commit=False``,
    so a plain ``session.get`` could return stale identity-map attributes.
    ``populate_existing`` forces a fresh row read and ``with_for_update`` locks it
    for the read-modify-write that follows.
    """

    return session.get(
        ClipMaterialization,
        envelope_id,
        populate_existing=True,
        with_for_update=True,
    )


def _fresh_get(session: Session, model: Any, pk: uuid.UUID) -> Any:
    """Read one row freshly, bypassing stale identity-map attributes."""

    return session.get(model, pk, populate_existing=True)


def reserve_attempt(
    session: Session,
    owner: MaterializationOwner,
    component: str,
    dependency_fingerprint: str,
    kind: str,
    tier: str | None,
) -> dict[str, object]:
    """Allocate a unique attempt identity and persist it as RESERVED.

    A database observer must see RESERVED before the provider method starts. The
    reservation is conservatively budget-consumed even if the process crashes
    before an actual external call.
    """

    row = _lock_envelope(session, owner.materialization_id)
    if row is None or not owner_is_current(session, owner):
        session.rollback()
        raise MaterializationOwnershipError("cannot reserve an attempt without ownership")

    ledger = row.call_ledger if isinstance(row.call_ledger, Mapping) else {}
    entries = ledger_entries(ledger)
    ordinal = 1 + sum(1 for e in entries if e.get("component") == component)
    entry: dict[str, object] = {
        "attempt_id": str(uuid.uuid4()),
        "component": component,
        "dependency_fingerprint": dependency_fingerprint,
        "phase": "INITIAL",
        "ordinal": ordinal,
        "kind": kind,
        "tier": tier,
        "status": ATTEMPT_RESERVED,
        "provider_identity": {},
        "response_fingerprint": None,
    }
    entries.append(entry)
    new_ledger = {"version": ATTEMPT_LEDGER_VERSION, "entries": entries}
    if not guarded_update(session, owner, {"call_ledger": new_ledger}):
        session.rollback()
        raise MaterializationOwnershipError("attempt reservation lost ownership")
    session.commit()
    return entry


def save_attempt_result(
    session: Session,
    owner: MaterializationOwner,
    attempt_id: str,
    *,
    status: str,
    response_fingerprint: str | None = None,
    provider_identity: Mapping[str, object] | None = None,
) -> bool:
    """Update one attempt's terminal status/response under the owner fence."""

    if status not in ATTEMPT_STATUSES:
        raise ValueError("unknown attempt status")
    row = _lock_envelope(session, owner.materialization_id)
    if row is None:
        session.rollback()
        return False
    entries = ledger_entries(row.call_ledger)
    found = False
    for entry in entries:
        if entry.get("attempt_id") == attempt_id:
            entry["status"] = status
            if response_fingerprint is not None:
                entry["response_fingerprint"] = response_fingerprint
            if provider_identity is not None:
                entry["provider_identity"] = dict(provider_identity)
            found = True
            break
    if not found:
        session.rollback()
        return False
    new_ledger = {"version": ATTEMPT_LEDGER_VERSION, "entries": entries}
    ok = guarded_update(session, owner, {"call_ledger": new_ledger})
    if not ok:
        session.rollback()
        return False
    session.commit()
    return True


def record_attempt_response(
    session: Session,
    owner: MaterializationOwner,
    attempt_id: str,
    *,
    status: str,
    component: str,
    response_fingerprint: str | None = None,
    provider_identity: Mapping[str, object] | None = None,
    checkpoint_patch: Mapping[str, object] | None = None,
) -> bool:
    """Atomically persist an attempt's terminal status and its reusable checkpoint.

    The locked, freshly-read envelope is read once so the attempt entry and the
    component checkpoint (e.g. an accepted writing response) commit together, and
    existing subcomponents/chunks are preserved rather than overwritten by stale
    identity-map state.
    """

    if status not in ATTEMPT_STATUSES:
        raise ValueError("unknown attempt status")
    row = _lock_envelope(session, owner.materialization_id)
    if row is None:
        session.rollback()
        return False
    entries = ledger_entries(row.call_ledger)
    found = False
    for entry in entries:
        if entry.get("attempt_id") == attempt_id:
            entry["status"] = status
            if response_fingerprint is not None:
                entry["response_fingerprint"] = response_fingerprint
            if provider_identity is not None:
                entry["provider_identity"] = dict(provider_identity)
            found = True
            break
    if not found:
        session.rollback()
        return False
    values: dict[str, object] = {
        "call_ledger": {"version": ATTEMPT_LEDGER_VERSION, "entries": entries}
    }
    if checkpoint_patch is not None:
        checkpoints = dict(row.component_checkpoints or {})
        existing = checkpoints.get(component)
        merged = dict(existing) if isinstance(existing, Mapping) else {}
        merged.update(dict(checkpoint_patch))
        checkpoints[component] = merged
        values["component_checkpoints"] = checkpoints
    ok = guarded_update(session, owner, values)
    if not ok:
        session.rollback()
        return False
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Component checkpoints
# ---------------------------------------------------------------------------


def load_component(
    session: Session,
    materialization_id: uuid.UUID | str,
    component: str,
    dependency_fingerprint: str,
) -> dict[str, object] | None:
    """Return an exact dependency-matching ACCEPTED checkpoint, else ``None``.

    This is a read: it performs no network calls and mutates no state.
    """

    row = _fresh_get(session, ClipMaterialization, _as_uuid(materialization_id))
    if row is None:
        return None
    checkpoints = row.component_checkpoints
    if not isinstance(checkpoints, Mapping):
        return None
    entry = checkpoints.get(component)
    if not isinstance(entry, Mapping):
        return None
    if entry.get("status") != CHECKPOINT_ACCEPTED:
        return None
    if entry.get("dependency_fingerprint") != dependency_fingerprint:
        return None
    return dict(entry)


def save_component_checkpoint(
    session: Session,
    owner: MaterializationOwner,
    component: str,
    checkpoint: Mapping[str, object],
) -> bool:
    """Persist one accepted (or explicit) component checkpoint atomically.

    Uses a fresh dict assignment so SQLAlchemy JSON mutation tracking fires. The
    result fingerprint is *not* inferred here; the caller supplies the complete
    validated result.
    """

    row = _lock_envelope(session, owner.materialization_id)
    if row is None:
        return False
    checkpoints = dict(row.component_checkpoints or {})
    checkpoints[component] = dict(checkpoint)
    return guarded_update(session, owner, {"component_checkpoints": checkpoints})


def save_local_component(
    session: Session,
    owner: MaterializationOwner,
    component: str,
    dependency_fingerprint: str,
    result: Mapping[str, object],
) -> bool:
    """Persist a locally-executed component result as an accepted checkpoint."""

    checkpoint = {
        "dependency_fingerprint": dependency_fingerprint,
        "status": CHECKPOINT_ACCEPTED,
        "result": dict(result),
    }
    return save_component_checkpoint(session, owner, component, checkpoint)


def merge_component_checkpoint(
    session: Session,
    owner: MaterializationOwner,
    component: str,
    patch: Mapping[str, object],
) -> bool:
    """Read-modify-write one component checkpoint under the owner fence.

    Preserves existing keys so an accepted intermediate subcomponent (for
    example an accepted writing response recorded before critique) survives a
    later partial update. Uses a fresh dict assignment for JSON mutation tracking.
    """

    row = _lock_envelope(session, owner.materialization_id)
    if row is None:
        return False
    checkpoints = dict(row.component_checkpoints or {})
    existing = checkpoints.get(component)
    merged: dict[str, object] = dict(existing) if isinstance(existing, Mapping) else {}
    merged.update(dict(patch))
    checkpoints[component] = merged
    ok = guarded_update(session, owner, {"component_checkpoints": checkpoints})
    if ok:
        session.commit()
    return ok


def commit_prepared(
    session: Session, owner: MaterializationOwner, prepared: PreparedMaterialization
) -> bool:
    """Persist a fully prepared pre-render materialization atomically.

    Commits accepted components, the immutable output fingerprint, lifecycle
    AWAITING_RENDER (or terminal), human-review flag, and job completion is left
    to the executor. Releases the envelope's active job binding.
    """

    checkpoints = dict(prepared.component_checkpoints)
    outcome = (
        MaterializationOutcome(prepared.semantic_outcome)
        if prepared.semantic_outcome is not None
        else None
    )
    values: dict[str, object] = {
        "lifecycle": prepared.lifecycle,
        "semantic_outcome": outcome,
        "human_review_required": bool(prepared.human_review_required),
        "publication_ready": False,
        "cache_eligible": False,
        "repair_count": int(prepared.repair_count),
        "reason_codes": list(prepared.reason_codes),
        "component_checkpoints": checkpoints,
        "authored_blocks": list(prepared.authored_blocks),
        "claim_ledger": list(prepared.claim_ledger),
        "slot_resolutions": list(prepared.slot_resolutions),
        "narration_references": list(prepared.narration_references),
        "asset_references": list(prepared.asset_references),
        "timeline_payload": dict(prepared.timeline_payload),
        "audio_recipe": dict(prepared.audio_recipe),
        "quality_findings": list(prepared.quality_findings),
        "provider_identity": dict(prepared.provider_identity),
        "upstream_snapshot": dict(prepared.upstream_snapshot),
        "metrics": dict(prepared.metrics),
        "output_fingerprint": prepared.output_fingerprint,
        "active_job_id": None,
    }
    return guarded_update(session, owner, values)


def finish_owner_job(
    session: Session,
    owner: MaterializationOwner,
    status: JobStatus,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
) -> bool:
    """Transition the owned RUNNING job (no commit; caller commits atomically).

    Fences on the exact claim token and a RUNNING status so a superseded worker
    cannot finalize another worker's job.
    """

    values: dict[str, object] = {
        "status": status,
        "completed_at": _utcnow(),
    }
    if status is JobStatus.FAILED:
        values["error_code"] = error_code
        values["error_message"] = error_message
    result = session.execute(
        update(ProcessingJob)
        .where(
            ProcessingJob.id == owner.job_id,
            ProcessingJob.claim_version == owner.job_claim_version,
            ProcessingJob.status == JobStatus.RUNNING,
        )
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def _utcnow() -> Any:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _optional_uuid(value: object) -> uuid.UUID | None:
    if value is None:
        return None
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


__all__ = [
    "ATTEMPT_LEDGER_VERSION",
    "ATTEMPT_RESERVED",
    "ATTEMPT_RESPONDED",
    "ATTEMPT_FAILED_KNOWN",
    "ATTEMPT_UNCERTAIN",
    "CHECKPOINT_ACCEPTED",
    "CHECKPOINT_DEFERRED",
    "CHECKPOINT_PENDING",
    "CHECKPOINT_REJECTED",
    "PRE_RENDER_CONTEXT_METRIC",
    "PRE_RENDER_CONTEXT_VERSION",
    "MaterializationOwner",
    "MaterializationOwnershipError",
    "PreparedMaterialization",
    "bind_active_job",
    "claim_materialization",
    "commit_prepared",
    "finish_owner_job",
    "get_or_create_envelope",
    "guarded_update",
    "ledger_counts",
    "ledger_entries",
    "load_component",
    "merge_component_checkpoint",
    "new_ledger",
    "owner_is_current",
    "record_attempt_response",
    "reserve_attempt",
    "save_attempt_result",
    "save_component_checkpoint",
    "save_local_component",
    "set_lifecycle",
]
