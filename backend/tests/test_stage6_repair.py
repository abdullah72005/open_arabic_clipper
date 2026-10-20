"""Stage 6 terminal-failure finalization and the single bounded repair.

A terminal ``FAILED`` ``AUDIO_PEAK_REPAIR_NEEDED`` render (which persists a real
unmastered measurement manifest, not a QC fingerprint) is finalizable: the
finalizer allocates the one reason-specific bounded peak repair by CAS 0 -> 1,
revises the output/render identity, and requests exactly one repaired render.
No second repair is ever allocated, and a re-delivered historical callback can
never revive the pre-repair revision.

This drives the real persisted production path. Broker/render delivery is
captured (the renderer execution itself is covered by the real-FFmpeg suites);
synthetic fixtures only, not product acceptance.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from test_stage6_vertical_slice import _bound_owner, _persisted_materialization

from app.core.enums import (
    JobStatus,
    RenderArtifactPurpose,
    RenderExecutionLifecycle,
)
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.finalization import (
    FINALIZATION_REPAIR_QUEUED,
    REPAIR_KIND_BOUNDED_PEAK,
    finalize_materialization,
    request_finalization_job,
)
from app.models import ClipMaterialization, ProcessingJob
from app.models.render_execution import RenderExecution

#: A genuinely infeasible measurement: at target loudness but a very hot peak, so
#: the ideal linear -16 LUFS gain cannot honour the -1.8 dBTP render guard
#: without a bounded peak limiter (needed reduction = 1.3 dB).
_HOT_MIX = {
    "input_i": -16.0,
    "input_tp": -0.5,
    "input_lra": 4.0,
    "input_thresh": -26.0,
    "target_offset": 0.0,
    "normalization_type": "linear",
}


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _current_transformed_render(
    session: Session, fixture: Any, envelope: ClipMaterialization
) -> RenderExecution:
    render = session.scalars(
        select(RenderExecution).where(
            RenderExecution.materialization_id == envelope.id,
            RenderExecution.artifact_purpose == RenderArtifactPurpose.TRANSFORMED_SHORT,
            RenderExecution.is_current.is_(True),
        )
    ).first()
    if render is None:
        candidate = fixture.stage50.selection.candidate
        render = RenderExecution(
            id=uuid.uuid4(),
            source_video_id=candidate.source_video_id,
            clip_candidate_id=candidate.id,
            materialization_id=envelope.id,
            artifact_purpose=RenderArtifactPurpose.TRANSFORMED_SHORT,
            lifecycle=RenderExecutionLifecycle.QUEUED,
            delivery_profile_key="MP4_H264_AAC_1080X1920_V1",
            is_current=True,
            input_fingerprint="i" * 64,
        )
        session.add(render)
        envelope.render_execution_id = render.id
        session.flush()
        session.commit()
    return render


def _make_peak_failure(
    session: Session,
    fixture: Any,
    envelope: ClipMaterialization,
    *,
    repair_revision: int = 0,
) -> RenderExecution:
    render = _current_transformed_render(session, fixture, envelope)
    render.lifecycle = RenderExecutionLifecycle.FAILED
    render.qc_status = None
    render.cache_eligible = False
    render.output_fingerprint = ""
    render.qc_fingerprint = ""
    render.error_code = "AUDIO_PEAK_REPAIR_NEEDED"
    render.reason_codes = ["AUDIO_PEAK_REPAIR_NEEDED"]
    render.execution_manifest = {"phase": "MASTER", "mix_measurement": dict(_HOT_MIX)}
    render.request_payload = {
        "materialization_id": str(envelope.id),
        "expected_output_fingerprint": envelope.output_fingerprint,
        "spec": {
            "authored": {
                "repair_revision": repair_revision,
                "mastering": {"render_guard_dbtp": -1.8},
            }
        },
    }
    render.artifact_reference = {}
    render.qc_result = {}
    envelope.render_execution_id = render.id
    session.flush()
    session.commit()
    return render


def _capture_dispatch(monkeypatch: Any) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def _fake(session: Any, _envelope: Any, _candidate_id: Any, **kwargs: Any) -> uuid.UUID:
        calls.append(dict(kwargs))
        new_id = uuid.uuid4()
        row = session.get(ClipMaterialization, _envelope.id, populate_existing=True)
        row.render_execution_id = new_id
        session.commit()
        return new_id

    monkeypatch.setattr(
        "app.materialization.finalization.dispatch_transformed_render", _fake
    )
    return calls


def test_terminal_peak_failure_allocates_one_bounded_repair(
    session: Session, monkeypatch: Any
) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    original_fp = envelope.output_fingerprint
    render = _make_peak_failure(session, fixture, envelope)
    calls = _capture_dispatch(monkeypatch)
    candidate = fixture.stage50.selection.candidate

    job, owner = _bound_owner(session, envelope, candidate)
    result = finalize_materialization(
        session, owner, settings=get_settings(), storage=fixture.stage50.storage
    )

    assert result.outcome == FINALIZATION_REPAIR_QUEUED
    assert len(calls) == 1
    session.refresh(envelope)
    # The repair is allocated exactly once and the identity is revised so the old
    # render can never satisfy the repaired revision.
    assert envelope.repair_count == 1
    assert envelope.output_fingerprint != original_fp
    assert envelope.lifecycle.value == "AWAITING_RENDER"
    assert int(envelope.audio_recipe["repair_peak_limit_db"] * 1000) == 1300
    actions = list(envelope.repair_actions or [])
    assert len(actions) == 1
    assert actions[0]["kind"] == REPAIR_KIND_BOUNDED_PEAK
    assert actions[0]["source_render_execution_id"] == str(render.id)
    refreshed_job = session.get(ProcessingJob, job.id)
    assert refreshed_job is not None and refreshed_job.status is JobStatus.SUCCEEDED

    # A re-delivered historical callback cannot request finalization of the
    # superseded (pre-repair) revision.
    assert request_finalization_job(session, render.id, settings=get_settings()) is None


def test_second_peak_failure_is_not_repaired_again(
    session: Session, monkeypatch: Any
) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _make_peak_failure(session, fixture, envelope)
    _capture_dispatch(monkeypatch)
    candidate = fixture.stage50.selection.candidate

    _job, owner = _bound_owner(session, envelope, candidate)
    first = finalize_materialization(
        session, owner, settings=get_settings(), storage=fixture.stage50.storage
    )
    assert first.outcome == FINALIZATION_REPAIR_QUEUED

    # The repaired render fails the same way, but the single repair is already
    # spent: the result is a truthful quality rejection, never a second repair.
    session.refresh(envelope)
    repaired_fp = envelope.output_fingerprint
    render = _current_transformed_render(session, fixture, envelope)
    render.lifecycle = RenderExecutionLifecycle.FAILED
    render.qc_status = None
    render.cache_eligible = False
    render.error_code = "AUDIO_PEAK_REPAIR_NEEDED"
    render.reason_codes = ["AUDIO_PEAK_REPAIR_NEEDED"]
    render.execution_manifest = {"phase": "MASTER", "mix_measurement": dict(_HOT_MIX)}
    render.request_payload = {
        "materialization_id": str(envelope.id),
        "expected_output_fingerprint": repaired_fp,
        "spec": {
            "authored": {
                "repair_revision": int(envelope.repair_count),
                "mastering": {"render_guard_dbtp": -1.8},
            }
        },
    }
    envelope.render_execution_id = render.id
    session.flush()
    session.commit()

    _job2, owner2 = _bound_owner(session, envelope, candidate)
    second = finalize_materialization(
        session, owner2, settings=get_settings(), storage=fixture.stage50.storage
    )
    session.refresh(envelope)
    assert second.outcome != FINALIZATION_REPAIR_QUEUED
    assert envelope.repair_count == 1
    assert envelope.semantic_outcome.value == "REJECTED_QUALITY"
    assert envelope.publication_ready is False


def test_terminal_non_peak_failure_defers(
    session: Session, monkeypatch: Any
) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    render = _current_transformed_render(session, fixture, envelope)
    render.lifecycle = RenderExecutionLifecycle.FAILED
    render.qc_status = None
    render.cache_eligible = False
    render.reason_codes = ["RENDER_PROCESS_FAILED"]
    render.error_code = "RENDER_PROCESS_FAILED"
    render.execution_manifest = {"phase": "MIX"}
    render.request_payload = {
        "materialization_id": str(envelope.id),
        "expected_output_fingerprint": envelope.output_fingerprint,
        "spec": {"authored": {"repair_revision": 0}},
    }
    envelope.render_execution_id = render.id
    session.flush()
    session.commit()
    _capture_dispatch(monkeypatch)

    candidate = fixture.stage50.selection.candidate
    _job, owner = _bound_owner(session, envelope, candidate)
    result = finalize_materialization(
        session, owner, settings=get_settings(), storage=fixture.stage50.storage
    )
    session.refresh(envelope)
    assert result.outcome == "DEFERRED"
    assert envelope.semantic_outcome.value == "DEFERRED_EVIDENCE"
    assert envelope.human_review_required is False
    assert envelope.repair_count == 0
