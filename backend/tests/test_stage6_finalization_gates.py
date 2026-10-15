"""Stage 6 finalizer gates: content/technical enforcement, exact binding, replay.

These regressions reproduce the SOL finalizer defects. They drive the *persisted*
production path (the real materialization executor) and the real
``finalize_materialization`` assessment, then assert the truthful non-promotion
behavior. Synthetic fixtures only; product acceptance uses genuine authorized
media through the same production route.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from test_stage6_vertical_slice import _bound_owner, _persisted_materialization

from app.core.enums import (
    JobStatus,
    RenderArtifactPurpose,
    RenderExecutionLifecycle,
    RenderQCStatus,
)
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.finalization import (
    FINALIZATION_CANCELLED,
    FINALIZATION_COMPLETE,
    FINALIZATION_DEFERRED,
    FINALIZATION_SUPERSEDED,
    finalize_materialization,
)
from app.models import ClipMaterialization, ProcessingJob
from app.models.render_execution import RenderExecution

_REVIEWABLE = {"READY_FOR_HUMAN_REVIEW", "MATERIALIZED_WITH_CAUTION"}


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _attach_render(
    session: Session,
    fixture: Any,
    envelope: ClipMaterialization,
    *,
    lufs: float = -16.1,
    peak: float = -1.8,
    qc_status: RenderQCStatus = RenderQCStatus.PASS,
    expected_fingerprint: str | None = None,
    repair_revision: int | None = None,
) -> RenderExecution:
    candidate = fixture.stage50.selection.candidate
    storage = fixture.stage50.storage
    directory = storage.source_directory(candidate.source_video_id) / "render-executions" / "g"
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "output.mp4"
    artifact.write_bytes(b"\x00" * 4096)
    relative = storage.storage_relative(artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    render = RenderExecution(
        id=uuid.uuid4(),
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        materialization_id=envelope.id,
        artifact_purpose=RenderArtifactPurpose.TRANSFORMED_SHORT,
        lifecycle=RenderExecutionLifecycle.COMPLETE,
        qc_status=qc_status,
        is_current=True,
        output_fingerprint="f" * 64,
        runtime_fingerprint="r" * 64,
        qc_fingerprint="q" * 64,
        request_payload={
            "materialization_id": str(envelope.id),
            "expected_output_fingerprint": (
                envelope.output_fingerprint
                if expected_fingerprint is None
                else expected_fingerprint
            ),
            "spec": {
                "authored": {
                    "repair_revision": (
                        int(envelope.repair_count or 0)
                        if repair_revision is None
                        else repair_revision
                    )
                },
            },
        },
        artifact_reference={"relative_path": relative, "sha256": digest, "size_bytes": 4096},
        qc_result={
            "status": qc_status.value,
            "measured": {"final_integrated_lufs": lufs, "final_true_peak_dbtp": peak},
        },
        reason_codes=[],
    )
    session.add(render)
    envelope.render_execution_id = render.id
    session.flush()
    session.commit()
    return render


def _finalize(session: Session, fixture: Any, envelope: ClipMaterialization) -> Any:
    _job, owner = _bound_owner(session, envelope, fixture.stage50.selection.candidate)
    return finalize_materialization(
        session, owner, settings=get_settings(), storage=fixture.stage50.storage
    )


# --- A. content and technical gates ------------------------------------------


def test_unmet_substantive_value_is_not_promoted(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    # Persist a substantive-original-value UNMET finding and a REJECTED_QUALITY
    # semantic outcome on an AWAITING_RENDER row (the reproduced defect).
    snapshot = dict(envelope.metrics.get("pre_render_assessment_context") or {})
    envelope.metrics = {**envelope.metrics, "pre_render_assessment_context": snapshot}
    envelope.quality_findings = [
        {
            "dimension": "SUBSTANTIVE_VALUE",
            "level": "UNMET",
            "reason_codes": ["PARAPHRASE_ONLY"],
            "evidence": {},
        }
    ]
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_COMPLETE
    assert envelope.semantic_outcome.value not in _REVIEWABLE
    assert envelope.semantic_outcome.value == "REJECTED_QUALITY"
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False
    assert "PARAPHRASE_ONLY" in envelope.reason_codes


def test_missing_required_slot_is_no_output(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    snapshot = dict(envelope.metrics.get("pre_render_assessment_context") or {})
    snapshot["required_slots_fulfilled"] = False
    envelope.metrics = {**envelope.metrics, "pre_render_assessment_context": snapshot}
    session.flush()
    _attach_render(session, fixture, envelope)

    _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert envelope.semantic_outcome.value == "NO_OUTPUT"
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False


def test_unsupported_verification_blocks(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    snapshot = dict(envelope.metrics.get("pre_render_assessment_context") or {})
    snapshot["unsupported_essential"] = True
    envelope.metrics = {**envelope.metrics, "pre_render_assessment_context": snapshot}
    session.flush()
    _attach_render(session, fixture, envelope)

    _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert envelope.semantic_outcome.value == "BLOCKED_VERIFICATION"
    assert envelope.cache_eligible is False


def test_unknown_critical_fidelity_defers(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    snapshot = dict(envelope.metrics.get("pre_render_assessment_context") or {})
    snapshot["critical_fidelity_unknown"] = True
    envelope.metrics = {**envelope.metrics, "pre_render_assessment_context": snapshot}
    session.flush()
    _attach_render(session, fixture, envelope)

    result = _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False


def test_out_of_policy_loudness_rejects_despite_pass_label(
    session: Session, monkeypatch: Any
) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _attach_render(session, fixture, envelope, lufs=-11.0, peak=-1.0, qc_status=RenderQCStatus.PASS)

    _finalize(session, fixture, envelope)
    session.refresh(envelope)
    assert envelope.semantic_outcome.value == "REJECTED_QUALITY"
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False


def test_missing_loudness_measurement_defers(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    render = _attach_render(session, fixture, envelope)
    render.qc_result = {"status": "PASS", "measured": {}}
    session.flush()
    session.commit()

    result = _finalize(session, fixture, envelope)
    assert result.outcome == FINALIZATION_DEFERRED
    session.refresh(envelope)
    assert envelope.semantic_outcome.value != "READY_FOR_HUMAN_REVIEW"
    assert envelope.cache_eligible is False


# --- B. exact generation binding ---------------------------------------------


def test_stale_output_fingerprint_is_superseded(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _attach_render(session, fixture, envelope, expected_fingerprint="0" * 64)

    result = _finalize(session, fixture, envelope)
    assert result.outcome == FINALIZATION_SUPERSEDED
    session.refresh(envelope)
    # The stale render never finalizes: the envelope stays pre-render
    # (AWAITING_RENDER) and is never marked review-ready or cache-eligible.
    assert envelope.lifecycle.value == "AWAITING_RENDER"
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False


def test_stale_repair_revision_is_superseded(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    envelope.repair_count = 1
    session.flush()
    _attach_render(session, fixture, envelope, repair_revision=0)

    result = _finalize(session, fixture, envelope)
    assert result.outcome == FINALIZATION_SUPERSEDED
    session.refresh(envelope)
    assert envelope.cache_eligible is False


# --- E. idempotence finishes the current job ---------------------------------


def test_replay_finishes_current_job_and_reuses(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _attach_render(session, fixture, envelope)

    request1 = _finalize(session, fixture, envelope)
    assert request1.outcome == FINALIZATION_COMPLETE
    session.refresh(envelope)
    assert envelope.semantic_outcome.value in _REVIEWABLE

    job2, owner2 = _bound_owner(session, envelope, fixture.stage50.selection.candidate)
    replay = finalize_materialization(
        session, owner2, settings=get_settings(), storage=fixture.stage50.storage
    )
    assert replay.reused is True
    assert replay.assessment_fingerprint == request1.assessment_fingerprint
    # The newly claimed finalization job must be terminal, never left RUNNING.
    refreshed = session.get(ProcessingJob, job2.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.SUCCEEDED


def test_cancelled_materialization_is_not_finalized(session: Session, monkeypatch: Any) -> None:
    from app.core.enums import MaterializationLifecycle

    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _attach_render(session, fixture, envelope)
    envelope.lifecycle = MaterializationLifecycle.CANCELLED
    session.flush()
    session.commit()

    result = _finalize(session, fixture, envelope)
    assert result.outcome == FINALIZATION_CANCELLED
