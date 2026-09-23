"""Stage 5.1 executor: claiming, fencing, cancellation, retries, and cleanup."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, update
from sqlalchemy.orm import Session
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    Stage51Fixture,
    seed_stage51,
)

from app.composition.executor import (
    VisualCompositionExecutor,
    build_visual_composition_executor,
)
from app.composition.policy import VisualCompositionExecutionStatus
from app.composition.queue import get_or_create_plan_row
from app.composition.service import CompositionInputError, get_current_visual_composition
from app.core.enums import JobKind, JobStatus
from app.db.base import Base
from app.models import ProcessingJob
from app.pipeline.executor import StageCancelled


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _job(
    session: Session, fixture: Stage51Fixture, row_id: Any, *, status: JobStatus = JobStatus.QUEUED
) -> ProcessingJob:
    job = ProcessingJob(
        source_video_id=fixture.stage50.selection.candidate.source_video_id,
        kind=JobKind.VISUAL_COMPOSITION,
        visual_composition_plan_id=row_id,
        status=status,
    )
    session.add(job)
    session.commit()
    return job


def _executor(
    session: Session,
    fixture: Stage51Fixture,
    *,
    sampler: FakeFrameSampler | None = None,
    detector: FakeDetector | None = None,
) -> VisualCompositionExecutor:
    return build_visual_composition_executor(
        session,
        fixture.stage50.storage,
        fixture.settings,  # type: ignore[arg-type]
        display_probe=fixture.display_probe,
        frame_sampler=sampler or FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=detector or FakeDetector(),
    )


def test_execute_claims_job_and_persists_ready_plan(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
    job = _job(session, fixture, row.id)
    executor = _executor(session, fixture)
    executor.set_active_job(job.id)
    result = executor.execute(row.id)
    assert result is not None
    session.refresh(job)
    assert job.status is JobStatus.SUCCEEDED
    assert job.claim_version >= 1
    assert job.heartbeat_at is not None
    current = get_current_visual_composition(session, row.clip_candidate_id)
    assert current is not None and current.plan_ready is True
    assert current.execution_status is VisualCompositionExecutionStatus.COMPLETE
    assert current.active_job_id is None


def test_cache_hit_skips_planning_and_releases_resources(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
    first = _executor(session, fixture)
    first.set_active_job(_job(session, fixture, row.id).id)
    first.execute(row.id)
    current = get_current_visual_composition(session, row.clip_candidate_id)
    assert current is not None
    sampler = FakeFrameSampler()
    detector = FakeDetector()
    second = _executor(session, fixture, sampler=sampler, detector=detector)
    second.set_active_job(_job(session, fixture, current.id).id)
    second.execute(current.id)
    assert sampler.calls == 0
    assert sampler.released >= 1
    assert detector.released >= 1


def test_cancelled_job_prevents_stale_persistence(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
    job = _job(session, fixture, row.id, status=JobStatus.CANCELLED)
    executor = _executor(session, fixture)
    executor.set_active_job(job.id)
    with pytest.raises(StageCancelled):
        executor.execute(row.id)
    session.refresh(row)
    session.refresh(job)
    assert row.execution_status is VisualCompositionExecutionStatus.CANCELLED
    assert row.plan_payload == {}
    assert job.status is JobStatus.CANCELLED


def test_duplicate_delivery_is_fenced(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
    job = _job(session, fixture, row.id)
    first = _executor(session, fixture)
    first.set_active_job(job.id)
    first.execute(row.id)
    duplicate = _executor(session, fixture)
    duplicate.set_active_job(job.id)
    duplicate.execute(row.id)
    assert duplicate.skipped_duplicate is True
    session.refresh(job)
    assert job.status is JobStatus.SUCCEEDED


def test_stale_worker_cannot_overwrite_newer_claim(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
    job = _job(session, fixture, row.id)
    executor = _executor(session, fixture)
    executor.set_active_job(job.id)
    assert executor._claim_job() is True  # noqa: SLF001 - test seam
    executor._plan_id = row.id  # noqa: SLF001 - test seam
    session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == job.id)
        .values(claim_version=ProcessingJob.claim_version + 1)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    executor.record_failure(CompositionInputError("prerequisite missing"))
    session.refresh(job)
    session.refresh(row)
    assert job.status is JobStatus.RUNNING
    assert row.execution_status != VisualCompositionExecutionStatus.FAILED


def test_record_failure_writes_sanitized_diagnostics(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
    job = _job(session, fixture, row.id)
    executor = _executor(session, fixture)
    executor.set_active_job(job.id)
    assert executor._claim_job() is True  # noqa: SLF001 - test seam
    executor._plan_id = row.id  # noqa: SLF001 - test seam
    executor.record_failure(CompositionInputError("prerequisite missing"))
    session.refresh(job)
    assert job.status is JobStatus.FAILED
    assert job.error_code == "COMPOSITION_INPUT"
    assert job.error_message == "prerequisite missing"


def test_failed_planner_persists_truthful_failure(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
    # Mutate the contract payload so no valid source span can be planned.
    payload = dict(fixture.contract.row.contract_payload)
    blocks = [dict(block) for block in payload["blocks"]]
    for block in blocks:
        if block.get("block_type") == "SOURCE_EXCERPT" and "source_binding" in block:
            binding = dict(block["source_binding"])
            binding["rebind_valid"] = False
            block["source_binding"] = binding
    payload["blocks"] = blocks
    fixture.contract.row.contract_payload = payload
    session.flush()
    job = _job(session, fixture, row.id)
    executor = _executor(session, fixture)
    executor.set_active_job(job.id)
    result = executor.execute(row.id)
    session.refresh(job)
    assert result is not None
    assert job.status is JobStatus.SUCCEEDED


def test_disabled_flag_fails_closed_in_executor(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    fixture.settings.visual_composition_enabled = False
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
    job = _job(session, fixture, row.id)
    executor = _executor(session, fixture)
    executor.set_active_job(job.id)

    with pytest.raises(CompositionInputError):
        executor.execute(row.id)

    session.refresh(job)
    session.refresh(row)
    assert job.status is JobStatus.FAILED
    assert row.execution_status is VisualCompositionExecutionStatus.FAILED
