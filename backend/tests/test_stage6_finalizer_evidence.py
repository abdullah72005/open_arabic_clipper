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
