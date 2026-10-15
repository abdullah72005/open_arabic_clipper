"""Stage 6 single durable delivery/resume router regressions.

These reproduce the SOL router defects: an AWAITING_RENDER POST must not blindly
return ``AWAITING_FINALIZATION``; a lost render callback is recovered through
POST/resume; a dispatch-failed FINALIZE job is redispatched to finalization (never
to preparation); duplicate/nonterminal/historical callbacks are handled
truthfully. Synthetic fixtures only.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session
from test_stage6_finalization_gates import _attach_render
from test_stage6_vertical_slice import _persisted_materialization

from app.core.enums import JobKind, JobStatus, RenderExecutionLifecycle
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization import queue as queue_module
from app.materialization.finalization import request_finalization_job
from app.materialization.queue import (
    DISPATCH_FAILED_MARKER,
    PHASE_FINALIZE,
    queue_materialization,
)
from app.materialization.repository import bind_active_job
from app.models import ProcessingJob
from app.models.render_execution import RenderExecution


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


class _CaptureDispatcher:
    def __init__(self) -> None:
        self.prepared: list[tuple[str, str]] = []
        self.finalized: list[tuple[str, str]] = []
        self.rendered: list[tuple[str, str]] = []

    def dispatch(
        self, render_execution_id: object, job_id: object, force: bool = False
    ) -> None:
        self.rendered.append((str(render_execution_id), str(job_id)))

    def dispatch_finalization(self, candidate_id: str, job_id: str) -> None:
        self.finalized.append((candidate_id, job_id))


def _finalizer_count(session: Session, envelope_id: uuid.UUID) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(ProcessingJob)
            .where(
                ProcessingJob.materialization_id == envelope_id,
                ProcessingJob.kind == JobKind.MATERIALIZATION,
                ProcessingJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
            )
        )
        or 0
    )


def test_lost_callback_is_recovered_by_post(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _attach_render(session, fixture, envelope)
    candidate = fixture.stage50.selection.candidate
    captured = _CaptureDispatcher()
    monkeypatch.setattr(queue_module, "_dispatch", lambda *a, **k: None)

    # No finalizer job exists: the render callback was lost.
    assert _finalizer_count(session, envelope.id) == 0
    outcome = queue_materialization(
        session, candidate.id, priority="CANDIDATE", render_dispatcher=captured
    )
    assert outcome.phase == PHASE_FINALIZE
    assert outcome.job_id is not None
    assert captured.finalized
    assert _finalizer_count(session, envelope.id) == 1


def test_failed_finalizer_delivery_redispatches_finalization(
    session: Session, monkeypatch: Any
) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    _attach_render(session, fixture, envelope)
    candidate = fixture.stage50.selection.candidate

    job = ProcessingJob(
        source_video_id=candidate.source_video_id,
        kind=JobKind.MATERIALIZATION,
        materialization_id=envelope.id,
        status=JobStatus.QUEUED,
        error_code=DISPATCH_FAILED_MARKER,
    )
    session.add(job)
    envelope.active_job_id = None
    session.commit()
    bind_active_job(session, envelope.id, job.id)
    session.commit()

    captured = _CaptureDispatcher()
    monkeypatch.setattr(queue_module, "_dispatch", lambda *a, **k: None)
    outcome = queue_materialization(
        session, candidate.id, priority="CANDIDATE", render_dispatcher=captured
    )
    assert outcome.job_id == job.id
    assert captured.finalized and not captured.rendered
    refreshed = session.get(ProcessingJob, job.id)
    assert refreshed is not None and refreshed.error_code is None


def test_duplicate_callback_coalesces(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    render = _attach_render(session, fixture, envelope)
    first = _CaptureDispatcher()
    job_id = request_finalization_job(
        session, render.id, settings=get_settings(), dispatcher=first
    )
    second = _CaptureDispatcher()
    again = request_finalization_job(
        session, render.id, settings=get_settings(), dispatcher=second
    )
    assert job_id is not None
    assert again == job_id
    assert _finalizer_count(session, envelope.id) == 1


def test_nonterminal_callback_is_ignored(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    render = _attach_render(session, fixture, envelope)
    render.lifecycle = RenderExecutionLifecycle.RENDERING
    session.flush()
    session.commit()
    assert (
        request_finalization_job(session, render.id, settings=get_settings()) is None
    )
    assert _finalizer_count(session, envelope.id) == 0


def test_historical_callback_is_ignored(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    render = _attach_render(session, fixture, envelope)
    render.is_current = False
    session.flush()
    session.commit()
    assert (
        request_finalization_job(session, render.id, settings=get_settings()) is None
    )
    assert _finalizer_count(session, envelope.id) == 0


def test_active_render_returns_identity_not_finalizer(
    session: Session, monkeypatch: Any
) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    render = _attach_render(session, fixture, envelope)
    render.lifecycle = RenderExecutionLifecycle.RENDERING
    session.flush()
    session.commit()
    candidate = fixture.stage50.selection.candidate
    captured = _CaptureDispatcher()
    monkeypatch.setattr(queue_module, "_dispatch", lambda *a, **k: None)
    outcome = queue_materialization(
        session, candidate.id, priority="CANDIDATE", render_dispatcher=captured
    )
    # An active render is reported; no finalizer is queued while it is running.
    assert outcome.job_id is None
    assert _finalizer_count(session, envelope.id) == 0
    assert session.get(RenderExecution, render.id).lifecycle is RenderExecutionLifecycle.RENDERING
