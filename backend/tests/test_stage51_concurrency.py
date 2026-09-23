"""Stage 5.1 concurrency: idempotent rows, single current plan, claim fencing."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    Stage51Fixture,
    seed_stage51,
)

from app.composition.executor import build_visual_composition_executor
from app.composition.policy import Stage51Config, VisualCompositionExecutionStatus
from app.composition.queue import get_or_create_plan_row, queue_visual_composition
from app.composition.service import CompositionInputError, execute_visual_composition
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


def _run(
    session: Session, fixture: Stage51Fixture, config: Stage51Config | None = None
) -> VisualCompositionPlan:
    if config is not None:
        fixture.settings.config = config
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
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
    assert result is not None
    return result


def test_two_queue_requests_converge_to_one_job(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    candidate = fixture.stage50.selection.candidate
    first = queue_visual_composition(session, candidate, settings=fixture.settings)  # type: ignore[arg-type]
    second = queue_visual_composition(session, candidate, settings=fixture.settings)  # type: ignore[arg-type]
    assert first.job_id == second.job_id
    assert session.query(ProcessingJob).count() == 1
    assert session.query(VisualCompositionPlan).count() == 1


def test_repeated_execution_keeps_one_row(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    first = _run(session, fixture)
    second = _run(session, fixture)
    assert first.id == second.id
    assert session.query(VisualCompositionPlan).count() == 1


def test_only_one_current_plan_after_config_change(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    first = _run(session, fixture)
    second = _run(session, fixture, config=Stage51Config(analysis_fps=4.0))
    assert first.id != second.id
    current = list(
        session.scalars(
            select(VisualCompositionPlan).where(VisualCompositionPlan.is_current.is_(True))
        ).all()
    )
    assert len(current) == 1
    assert current[0].id == second.id


def test_superseded_worker_cannot_overwrite_current_plan(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    current = _run(session, fixture)
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
    job = ProcessingJob(
        source_video_id=fixture.stage50.selection.candidate.source_video_id,
        kind=JobKind.VISUAL_COMPOSITION,
        visual_composition_plan_id=row.id,
    )
    session.add(job)
    session.commit()
    executor = build_visual_composition_executor(
        session,
        fixture.stage50.storage,
        fixture.settings,  # type: ignore[arg-type]
        display_probe=fixture.display_probe,
        frame_sampler=FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=FakeDetector(),
    )
    executor.set_active_job(job.id)
    assert executor._claim_job() is True  # noqa: SLF001 - test seam
    executor._plan_id = row.id  # noqa: SLF001 - test seam
    session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == job.id)
        .values(claim_version=ProcessingJob.claim_version + 1, status=JobStatus.RUNNING)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    executor.record_failure(CompositionInputError("stale worker"))
    session.refresh(current)
    assert current.is_current is True
    assert current.plan_ready is True
    assert current.execution_status is VisualCompositionExecutionStatus.COMPLETE
