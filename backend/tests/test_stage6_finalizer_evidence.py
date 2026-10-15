"""Stage 6 finalizer: live currentness and versioned, dependency-bound evidence.

These regressions reproduce the remaining SOL finalizer defects:

* live upstream effectiveness is re-derived at finalization (a stale FINAL_CLIP /
  selection / plan / contract / composition never authorizes review readiness);
* the pre-render snapshot is a versioned, dependency-bound checkpoint — a
  missing/empty/partial/malformed/dependency-mismatched snapshot is refused and
  rebuilt from authoritative evidence, and required slots are derived from the
  contract rather than inferred from an empty list;
* every handled finalization disposition closes its owned job.

Synthetic fixtures only; product acceptance uses genuine authorized media.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from test_stage6_finalization_gates import _attach_render
from test_stage6_vertical_slice import _bound_owner, _persisted_materialization

from app.core.enums import (
    JobStatus,
    RefinementPriority,
    RefinementStatus,
)
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.finalization import (
    FINALIZATION_COMPLETE,
    FINALIZATION_DEFERRED,
    FINALIZATION_SUPERSEDED,
    finalize_materialization,
)
from app.materialization.repository import PRE_RENDER_CONTEXT_METRIC
from app.materialization.service import MaterializationError, assemble_inputs
from app.models import CandidateRefinement, ClipMaterialization, ProcessingJob

_REVIEWABLE = {"READY_FOR_HUMAN_REVIEW", "MATERIALIZED_WITH_CAUTION"}


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _final_clip(session: Session, candidate_id: uuid.UUID) -> CandidateRefinement:
    return session.scalars(
        select(CandidateRefinement).where(
            CandidateRefinement.clip_candidate_id == candidate_id,
            CandidateRefinement.priority == RefinementPriority.FINAL_CLIP,
        )
    ).one()


def _set_snapshot(envelope: ClipMaterialization, raw: object) -> None:
    metrics = dict(envelope.metrics or {})
    metrics[PRE_RENDER_CONTEXT_METRIC] = raw
    envelope.metrics = metrics


# --- 1. live currentness ------------------------------------------------------


def test_finalize_refuses_stale_final_clip(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _attach_render(session, fixture, envelope)
    candidate = fixture.stage50.selection.candidate

    refine = _final_clip(session, candidate.id)
    refine.status = RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
    session.flush()
    session.commit()

    with pytest.raises(MaterializationError):
        assemble_inputs(session, candidate.id, priority="CANDIDATE")

    job, owner = _bound_owner(session, envelope, candidate)
    result = finalize_materialization(
        session, owner, settings=get_settings(), storage=fixture.stage50.storage
    )
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_SUPERSEDED
    assert envelope.human_review_required is False
    assert envelope.semantic_outcome.value != "READY_FOR_HUMAN_REVIEW"
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False
    # The stale run's owned job is terminal, never left RUNNING.
    refreshed = session.get(ProcessingJob, job.id)
    assert refreshed is not None and refreshed.status is JobStatus.SUCCEEDED


def test_finalize_refuses_stale_selected_plan(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _attach_render(session, fixture, envelope)
    candidate = fixture.stage50.selection.candidate

    # A changed selection plan identity invalidates the generation.
    from app.models import TransformationPlanSelection

    selection = session.get(TransformationPlanSelection, envelope.selection_id)
    selection.selected_plan_id = uuid.uuid4()
    session.flush()
    session.commit()

    job, owner = _bound_owner(session, envelope, candidate)
    result = finalize_materialization(
        session, owner, settings=get_settings(), storage=fixture.stage50.storage
    )
    session.refresh(envelope)
    assert result.outcome in {FINALIZATION_SUPERSEDED, FINALIZATION_DEFERRED}
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    refreshed = session.get(ProcessingJob, job.id)
    assert refreshed is not None and refreshed.status is not JobStatus.RUNNING


# --- 2. versioned, dependency-bound evidence ---------------------------------


def test_missing_snapshot_rebuilds_from_authoritative_evidence(
    session: Session, monkeypatch: Any
) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    metrics = dict(envelope.metrics or {})
    metrics.pop(PRE_RENDER_CONTEXT_METRIC, None)
    envelope.metrics = metrics
    session.flush()
    _attach_render(session, fixture, envelope)

    _finalize(session, fixture, envelope)
    session.refresh(envelope)
    # The snapshot is missing but the accepted evidence is complete: the context
    # is rebuilt deterministically, so the result is truthfully reviewable.
    assert envelope.semantic_outcome.value in _REVIEWABLE


@pytest.mark.parametrize(
    "raw",
    [
        {},  # empty dictionary
        {"version": "stage6-prerender-context-v2"},  # partial (no binding/context)
        {
            "version": "stage6-prerender-context-v2",
            "materialization_output_fingerprint": "mismatch",
            "context": {},
        },  # dependency-mismatched (favorable empty context)
        {
            "version": "stage6-prerender-context-v2",
            "materialization_output_fingerprint": None,
            "context": {"required_slots_fulfilled": True},
        },  # malformed types
    ],
)
def test_refused_snapshot_never_implies_satisfied_gates(
    session: Session, monkeypatch: Any, raw: object
) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    # A favorable-but-invalid snapshot must not satisfy a genuinely failed slot:
    # persist a real required AUTHORED_TEXT DEFERRED resolution.
    resolutions = [dict(item) for item in (envelope.slot_resolutions or [])]
    for item in resolutions:
        if item.get("slot_kind") == "AUTHORED_TEXT":
            item["resolution"] = "DEFERRED"
            item["reason_code"] = "MISSING_REQUIRED_VALUE"
    envelope.slot_resolutions = resolutions
    _set_snapshot(envelope, raw)
    session.flush()
    _attach_render(session, fixture, envelope)

    _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert envelope.semantic_outcome.value == "NO_OUTPUT"
    assert envelope.human_review_required is False


def test_missing_evidence_defers_distinct_from_explicit_failure(
    session: Session, monkeypatch: Any
) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    # Remove the accepted SCRIPT checkpoint and the timeline: required evidence is
    # absent (not explicitly failed), which must defer rather than reject.
    envelope.component_checkpoints = {}
    envelope.timeline_payload = {}
    envelope.slot_resolutions = []
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"
    assert envelope.semantic_outcome.value != "NO_OUTPUT"


# --- 3. handled dispositions close the owned job ------------------------------


@pytest.mark.parametrize("mode", ["stale_binding", "missing_render", "cancelled"])
def test_early_return_paths_close_the_job(session: Session, monkeypatch: Any, mode: str) -> None:
    from app.core.enums import MaterializationLifecycle

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    candidate = fixture.stage50.selection.candidate

    if mode == "stale_binding":
        _attach_render(session, fixture, envelope, expected_fingerprint="0" * 64)
    elif mode == "missing_render":
        envelope.render_execution_id = None
        session.flush()
        session.commit()
    else:
        envelope.lifecycle = MaterializationLifecycle.CANCELLED
        session.flush()
        session.commit()

    job, owner = _bound_owner(session, envelope, candidate)
    finalize_materialization(
        session, owner, settings=get_settings(), storage=fixture.stage50.storage
    )
    refreshed = session.get(ProcessingJob, job.id)
    assert refreshed is not None
    # No handled branch ever leaves its job RUNNING.
    assert refreshed.status is not JobStatus.RUNNING
    assert refreshed.status in {JobStatus.SUCCEEDED, JobStatus.CANCELLED}


def _finalize(session: Session, fixture: Any, envelope: ClipMaterialization) -> Any:
    _job, owner = _bound_owner(session, envelope, fixture.stage50.selection.candidate)
    return finalize_materialization(
        session, owner, settings=get_settings(), storage=fixture.stage50.storage
    )


# --- 4. accepted CONTENT, not status labels -----------------------------------


def _drop_snapshot(envelope: ClipMaterialization) -> None:
    metrics = dict(envelope.metrics or {})
    metrics.pop(PRE_RENDER_CONTEXT_METRIC, None)
    envelope.metrics = metrics


def test_accepted_status_label_without_content_defers(
    session: Session, monkeypatch: Any
) -> None:
    """A bare SCRIPT ``{status: ACCEPTED}`` is not accepted content."""

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _drop_snapshot(envelope)
    envelope.component_checkpoints = {"SCRIPT": {"status": "ACCEPTED"}}
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False


def test_fulfilled_slot_without_actual_material_defers(
    session: Session, monkeypatch: Any
) -> None:
    """A FULFILLED slot resolution must correspond to accepted authored material."""

    from app.materialization.fingerprints import stable_digest

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _drop_snapshot(envelope)
    # Self-consistent but empty accepted content: the fingerprint matches the
    # empty block list, so only the slot/material correspondence can catch it.
    envelope.authored_blocks = []
    checkpoints = dict(envelope.component_checkpoints or {})
    script = dict(checkpoints.get("SCRIPT") or {})
    script["result"] = {"authored_blocks": []}
    script["result_fingerprint"] = stable_digest([])
    checkpoints["SCRIPT"] = script
    envelope.component_checkpoints = checkpoints
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"


def test_contradictory_script_content_defers(session: Session, monkeypatch: Any) -> None:
    """Top-level authored blocks that disagree with the checkpoint are refused."""

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _drop_snapshot(envelope)
    envelope.authored_blocks = []
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"


def test_favorable_snapshot_never_overrides_missing_content(
    session: Session, monkeypatch: Any
) -> None:
    """A favorable persisted snapshot cannot stand in for absent accepted content."""

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    # Keep the (favorable) snapshot, but strip the accepted content it describes.
    envelope.component_checkpoints = {}
    envelope.timeline_payload = {}
    envelope.slot_resolutions = []
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"
    assert envelope.human_review_required is False
    assert envelope.publication_ready is False


# --- 5. authoritative SCRIPT dependency + UNKNOWN critical evidence ------------


def _snapshot_context(envelope: ClipMaterialization) -> dict[str, Any]:
    metrics = dict(envelope.metrics or {})
    raw = dict(metrics[PRE_RENDER_CONTEXT_METRIC])
    context = dict(raw.get("context") or {})
    raw["context"] = context
    metrics[PRE_RENDER_CONTEXT_METRIC] = raw
    envelope.metrics = metrics
    return context


def _add_snapshot_finding(
    envelope: ClipMaterialization,
    *,
    dimension: str,
    level: str,
    reason_codes: list[str],
) -> None:
    context = _snapshot_context(envelope)
    extra = [dict(item) for item in (context.get("extra_findings") or [])]
    extra.append(
        {
            "dimension": dimension,
            "level": level,
            "reason_codes": list(reason_codes),
            "evidence": {},
        }
    )
    context["extra_findings"] = extra


def test_snapshot_adverse_finding_is_not_discarded(
    session: Session, monkeypatch: Any
) -> None:
    """Adverse evidence in the dependency-bound snapshot is authoritative.

    Reproduces SOL's confirmed bug: ``_derive_evidence`` replaced
    ``base.extra_findings`` with the (favorable) top-level row findings, so an
    UNMET ``SUBSTANTIVE_VALUE`` / ``PARAPHRASE_ONLY`` finding that lived only in
    the validated snapshot silently disappeared and the materialization became
    reviewable. The adverse finding must survive and reject the generation.
    """

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _add_snapshot_finding(
        envelope,
        dimension="SUBSTANTIVE_VALUE",
        level="UNMET",
        reason_codes=["PARAPHRASE_ONLY"],
    )
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_COMPLETE
    assert envelope.semantic_outcome.value == "REJECTED_QUALITY"
    assert envelope.semantic_outcome.value not in _REVIEWABLE
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False
    persisted = {
        (str(item.get("dimension")), str(item.get("level"))): set(item.get("reason_codes") or [])
        for item in (envelope.quality_findings or [])
    }
    assert ("SUBSTANTIVE_VALUE", "UNMET") in persisted
    assert "PARAPHRASE_ONLY" in persisted[("SUBSTANTIVE_VALUE", "UNMET")]
    assert "PARAPHRASE_ONLY" in (envelope.reason_codes or [])


def test_snapshot_critical_unknown_defers(session: Session, monkeypatch: Any) -> None:
    """UNKNOWN critical evidence in the snapshot is missing evidence, not favorable.

    A contradiction between a favorable row and an UNKNOWN critical semantic
    dimension in the validated snapshot must defer truthfully rather than
    promote to review readiness.
    """

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _add_snapshot_finding(
        envelope,
        dimension="GROUNDING_VERIFICATION",
        level="UNKNOWN",
        reason_codes=[],
    )
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"
    assert envelope.semantic_outcome.value not in _REVIEWABLE
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False


def test_snapshot_adverse_finding_survives_idempotent_replay(
    session: Session, monkeypatch: Any
) -> None:
    """An adverse snapshot finding is preserved across an identical replay."""

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _add_snapshot_finding(
        envelope,
        dimension="SUBSTANTIVE_VALUE",
        level="UNMET",
        reason_codes=["PARAPHRASE_ONLY"],
    )
    session.flush()
    _attach_render(session, fixture, envelope)

    first = _finalize(session, fixture, envelope)
    assert first.outcome == FINALIZATION_COMPLETE
    session.refresh(envelope)
    assert envelope.semantic_outcome.value == "REJECTED_QUALITY"

    replay = _finalize(session, fixture, envelope)
    assert replay.reused is True
    assert replay.assessment_fingerprint == first.assessment_fingerprint
    session.refresh(envelope)
    assert envelope.semantic_outcome.value == "REJECTED_QUALITY"
    assert "PARAPHRASE_ONLY" in (envelope.reason_codes or [])


def test_missing_script_dependency_binding_defers(session: Session, monkeypatch: Any) -> None:
    """A self-consistent SCRIPT result hash with no dependency binding defers.

    Reproduces SOL blocker A: removing ``SCRIPT.dependency_fingerprint`` (while
    preserving the result/content hash) must never become reviewable.
    """

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    checkpoints = dict(envelope.component_checkpoints or {})
    script = dict(checkpoints.get("SCRIPT") or {})
    script.pop("dependency_fingerprint", None)
    checkpoints["SCRIPT"] = script
    envelope.component_checkpoints = checkpoints
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False


def test_script_dependency_inputs_must_match_live_generation(
    session: Session, monkeypatch: Any
) -> None:
    """A dependency binding from another generation is refused, not trusted."""

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    checkpoints = dict(envelope.component_checkpoints or {})
    script = dict(checkpoints.get("SCRIPT") or {})
    raw = dict(script.get("dependency_inputs") or {})
    raw["validation_version"] = "other-generation"
    script["dependency_inputs"] = raw
    checkpoints["SCRIPT"] = script
    envelope.component_checkpoints = checkpoints
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"


def test_unknown_grounding_dimension_defers(session: Session, monkeypatch: Any) -> None:
    """An UNKNOWN critical grounding dimension is missing evidence, not a gate.

    Reproduces SOL blocker B: with the pre-render snapshot removed, an empty claim
    ledger, the five required deterministic dimensions retained, and
    ``GROUNDING_VERIFICATION`` set to UNKNOWN, finalization must defer rather than
    return MATERIALIZED_WITH_CAUTION with human review.
    """

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _drop_snapshot(envelope)
    envelope.claim_ledger = []
    findings = [dict(item) for item in (envelope.quality_findings or [])]
    present = {str(item.get("dimension")) for item in findings}
    for dimension in (
        "SLOT_FULFILLMENT",
        "GROUNDING_VERIFICATION",
        "HERO_PRESERVATION",
        "VISUAL_CAPTION_READABILITY",
        "PLATFORM_SUITABILITY",
    ):
        assert dimension in present, dimension
    for item in findings:
        if item.get("dimension") == "GROUNDING_VERIFICATION":
            item["level"] = "UNKNOWN"
            item["reason_codes"] = []
    envelope.quality_findings = findings
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False


# --- 6. successful-render identity requirements (separate from callback) -------


def test_complete_render_requires_exact_identity(session: Session, monkeypatch: Any) -> None:
    """A COMPLETE render missing output/runtime/QC fingerprints must defer.

    Reproduces SOL blocker C: the callback/QC separation must not let a successful
    assessment pass without its independent complete-evidence identity.
    """

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    render = _attach_render(session, fixture, envelope)
    render.output_fingerprint = ""
    render.runtime_fingerprint = ""
    render.qc_fingerprint = ""
    session.flush()
    session.commit()

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False


# --- 7. late demotion before publication ---------------------------------------


def _other_session(session: Session) -> Session:
    return Session(session.get_bind())


def test_render_demoted_before_publication_is_not_published(
    session: Session, monkeypatch: Any
) -> None:
    """Reproduces SOL blocker D: a bound render demoted immediately before
    ``_publish_assessment`` is never published as current review readiness."""

    import app.materialization.finalization as fin
    from app.models.render_execution import RenderExecution

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    render = _attach_render(session, fixture, envelope)
    original = fin._publish_assessment
    fired = {"done": False}

    def hook(*args: Any, **kwargs: Any) -> Any:
        if not fired["done"]:
            fired["done"] = True
            other = _other_session(session)
            try:
                row = other.get(RenderExecution, render.id)
                row.is_current = False
                other.commit()
            finally:
                other.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(fin, "_publish_assessment", hook)
    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_SUPERSEDED
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False


def test_materialization_demoted_before_publication_closes_owned_job(
    session: Session, monkeypatch: Any
) -> None:
    """Reproduces SOL blocker E: demoting the materialization immediately before
    ``_publish_assessment`` supersedes it and never strands the owned job RUNNING.
    """

    import app.materialization.finalization as fin

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _attach_render(session, fixture, envelope)
    original = fin._publish_assessment
    fired = {"done": False}

    def hook(*args: Any, **kwargs: Any) -> Any:
        if not fired["done"]:
            fired["done"] = True
            other = _other_session(session)
            try:
                row = other.get(ClipMaterialization, envelope.id)
                row.is_current = False
                other.commit()
            finally:
                other.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(fin, "_publish_assessment", hook)
    job, owner = _bound_owner(
        session, envelope, fixture.stage50.selection.candidate
    )
    result = finalize_materialization(
        session, owner, settings=get_settings(), storage=fixture.stage50.storage
    )
    assert result.outcome == FINALIZATION_SUPERSEDED
    refreshed = session.get(ProcessingJob, job.id)
    assert refreshed is not None
    assert refreshed.status is not JobStatus.RUNNING
    assert refreshed.status is JobStatus.SUCCEEDED
