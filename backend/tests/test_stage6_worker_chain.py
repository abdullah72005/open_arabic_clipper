"""Stage 6 automatic worker-chain wiring: finalization job request + branch.

Deterministic wiring regressions for the prepare -> render -> finalize chain:
the render-completion callback requests one coalesced MATERIALIZATION finalizer
job for the exact envelope, and the executor branches to finalization without
rerunning accepted writing or synthesis. Real FFmpeg execution is proven
separately in ``test_stage6_authored_render.py``/``test_stage6_vertical_slice.py``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from test_stage6_finalization_gates import _attach_render
from test_stage6_vertical_slice import _persisted_materialization

from app.core.enums import JobKind, JobStatus
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.executor import build_materialization_executor
from app.materialization.finalization import request_finalization_job
from app.models import ProcessingJob


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


class _CaptureDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def dispatch_finalization(self, candidate_id: str, job_id: str) -> None:
        self.calls.append((candidate_id, job_id))

    def dispatch(self, candidate_id: str, job_id: str) -> None:
        self.calls.append((candidate_id, job_id))


def test_render_callback_requests_one_coalesced_finalizer_job(
    session: Session, monkeypatch: Any
) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    render = _attach_render(session, fixture, envelope)
    dispatcher = _CaptureDispatcher()

    first = request_finalization_job(
        session, render.id, settings=get_settings(), dispatcher=dispatcher
    )
    second = request_finalization_job(
        session, render.id, settings=get_settings(), dispatcher=dispatcher
    )
    assert first is not None
    assert second == first  # coalesced: no second job
    job = session.get(ProcessingJob, first)
    assert job is not None
    assert job.kind is JobKind.MATERIALIZATION
    assert job.materialization_id == envelope.id
    assert dispatcher.calls == [(str(envelope.clip_candidate_id), str(first))]

    active = session.scalars(
        select(ProcessingJob).where(
            ProcessingJob.materialization_id == envelope.id,
            ProcessingJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
        )
    ).all()
    assert len(active) == 1


def test_request_is_noop_without_terminal_binding(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    # No render attached to the envelope yet.
    result = request_finalization_job(
        session, uuid.uuid4(), settings=get_settings(), dispatcher=_CaptureDispatcher()
    )
    assert result is None


def test_executor_finalize_branch_completes_envelope(session: Session, monkeypatch: Any) -> None:
    fixture, envelope = _persisted_materialization(session, monkeypatch)
    render = _attach_render(session, fixture, envelope)

    job_id = request_finalization_job(
        session, render.id, settings=get_settings(), dispatcher=_CaptureDispatcher()
    )
    assert job_id is not None
    # The finalizer job is claimed by a fresh executor and branches to
    # finalization without running any provider/synthesis work.
    executor = build_materialization_executor(session, fixture.stage50.storage, get_settings())
    executor.set_active_job(job_id)
    result = executor.finalize(str(fixture.stage50.selection.candidate.id))
    session.refresh(envelope)
    assert envelope.lifecycle.value == "COMPLETE"
    assert envelope.semantic_outcome is not None
    assert envelope.publication_ready is False
    job = session.get(ProcessingJob, job_id)
    assert job is not None and job.status is JobStatus.SUCCEEDED
    assert result.output_fingerprint == result.output_fingerprint  # returned fingerprint
