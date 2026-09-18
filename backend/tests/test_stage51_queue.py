"""Stage 5.1 queue: prerequisite validation, idempotency, and cache reuse."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    Stage51Fixture,
    seed_stage51,
)

from app.composition.policy import VisualCompositionStatus
from app.composition.queue import (
    CompositionQueueError,
    get_or_create_plan_row,
    queue_visual_composition,
    validate_candidate_for_composition,
)
from app.composition.service import execute_visual_composition, get_current_visual_composition
from app.core.enums import JobKind, JobStatus
from app.db.base import Base
from app.models import ProcessingJob
from app.models.visual_composition_plan import VisualCompositionPlan


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.composition.queue._dispatch", lambda *args: None)


def _run(session: Session, fixture: Stage51Fixture) -> None:
    row = get_current_visual_composition(session, fixture.stage50.selection.candidate.id)
    assert row is not None
    result = execute_visual_composition(
        session,
        row.id,
        storage=fixture.stage50.storage,
        settings=fixture.settings,  # type: ignore[arg-type]
        display_probe=fixture.display_probe,
        frame_sampler=FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=FakeDetector(),
    )
    assert result is not None and result.plan_ready is True
    # Simulate the worker finalizing the queued job it was dispatched for.
    for job in (
        session.query(ProcessingJob)
        .filter(
            ProcessingJob.kind == JobKind.VISUAL_COMPOSITION,
            ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
        )
        .all()
    ):
        job.status = JobStatus.SUCCEEDED
    session.commit()


def test_validate_rejects_when_contract_missing(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    session.delete(fixture.contract.row)
    session.flush()
    with pytest.raises(CompositionQueueError):
        validate_candidate_for_composition(session, fixture.stage50.selection.candidate.id)


def test_validate_rejects_non_executable_contract(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    fixture.contract.row.contract_ready = False
    fixture.contract.row.status = VisualCompositionStatus.BLOCKED
    session.flush()
    with pytest.raises(CompositionQueueError):
        validate_candidate_for_composition(session, fixture.stage50.selection.candidate.id)


def test_queue_creates_one_row_and_job(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    candidate = fixture.stage50.selection.candidate
    outcome = queue_visual_composition(session, candidate, settings=fixture.settings)  # type: ignore[arg-type]
    assert outcome.queued is True and outcome.active is False and outcome.cached is False
    assert outcome.job_id is not None
    row = get_current_visual_composition(session, candidate.id)
    assert row is not None and row.id == outcome.plan_id
    jobs = (
        session.query(ProcessingJob).filter(ProcessingJob.kind == JobKind.VISUAL_COMPOSITION).all()
    )
    assert len(jobs) == 1
    assert jobs[0].visual_composition_plan_id == row.id
    assert jobs[0].status is JobStatus.QUEUED
    assert row.active_job_id == jobs[0].id


def test_identical_concurrent_requests_converge(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    candidate = fixture.stage50.selection.candidate
    first = queue_visual_composition(session, candidate, settings=fixture.settings)  # type: ignore[arg-type]
    second = queue_visual_composition(session, candidate, settings=fixture.settings)  # type: ignore[arg-type]
    assert first.plan_id == second.plan_id
    assert second.active is True
    assert second.job_id == first.job_id
    assert session.query(VisualCompositionPlan).count() == 1
    assert session.query(ProcessingJob).count() == 1


def test_queue_reuses_cache_after_execution(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    candidate = fixture.stage50.selection.candidate
    queued = queue_visual_composition(session, candidate, settings=fixture.settings)  # type: ignore[arg-type]
    _run(session, fixture)
    cached = queue_visual_composition(session, candidate, settings=fixture.settings)  # type: ignore[arg-type]
    assert cached.cached is True
    assert cached.job_id is None
    assert cached.queued is False
    assert cached.plan_id == queued.plan_id


def test_force_rerun_bypasses_cache(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    candidate = fixture.stage50.selection.candidate
    queue_visual_composition(session, candidate, settings=fixture.settings)  # type: ignore[arg-type]
    _run(session, fixture)
    forced = queue_visual_composition(
        session,
        candidate,
        force=True,
        settings=fixture.settings,  # type: ignore[arg-type]
    )
    assert forced.queued is True and forced.job_id is not None and forced.cached is False


def test_get_or_create_row_is_idempotent(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    candidate = fixture.stage50.selection.candidate
    first = get_or_create_plan_row(session, candidate)
    second = get_or_create_plan_row(session, candidate)
    assert first.id == second.id
    assert session.query(VisualCompositionPlan).count() == 1
