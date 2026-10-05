"""Stage 5.2 lifecycle: queue convergence, fencing, cancellation, cache."""

from __future__ import annotations

import hashlib
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

from app.composition.queue import queue_visual_composition
from app.composition.service import execute_visual_composition, get_current_visual_composition
from app.core.enums import JobStatus, RenderExecutionLifecycle
from app.core.settings import get_settings
from app.db.base import Base
from app.models import ProcessingJob
from app.models.render_execution import RenderExecution
from app.pipeline.executor import StageCancelled
from app.render.execution.concurrency import NullRenderAdmission
from app.render.execution.executor import build_render_execution_executor
from app.render.execution.queue import queue_render_execution
from app.render.execution.service import read_render_execution
from app.render.execution.types import (
    RenderArtifacts,
    RuntimeIdentity,
    TechnicalQCResult,
)
from app.services.storage import StorageService


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.render.execution.queue._dispatch", lambda *args: None)
    monkeypatch.setattr("app.composition.queue._dispatch", lambda *args: None)


def _plan_ready(session: Session, fixture: Stage51Fixture) -> None:
    queue_visual_composition(session, fixture.stage50.selection.candidate)
    row = get_current_visual_composition(session, fixture.stage50.selection.candidate.id)
    assert row is not None
    result = execute_visual_composition(
        session,
        row.id,
        storage=fixture.stage50.storage,
        settings=fixture.settings,
        display_probe=fixture.display_probe,
        frame_sampler=FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=FakeDetector(),
    )
    assert result is not None and result.plan_ready is True
    session.commit()


class _FakeRunner:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, compiled: Any, context: Any) -> RenderArtifacts:
        self.calls += 1
        output = context.attempt_directory / "output.mp4"
        payload = b"fake-render-bytes"
        output.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        probe = {
            "width": 1080,
            "height": 1920,
            "video_codec": "h264",
            "pix_fmt": "yuv420p",
            "audio_codec": "aac",
            "audio_sample_rate": 48000,
            "audio_channels": 2,
            "duration_seconds": 2.0,
            "frame_count": 60,
            "streams": {"video": 1, "audio": 1, "subtitle": 0, "data": 0},
            "rotation_degrees": 0,
            "sample_aspect_ratio": "1:1",
        }
        return RenderArtifacts(
            output_path=output,
            output_relative_path="output.mp4",
            sha256=digest,
            size_bytes=len(payload),
            probe=probe,
            manifest=compiled.manifest.as_dict(),
            duration_seconds=2.0,
            frame_count=60,
            sample_count=96000,
            sample_rate=48000,
            channels=2,
        )


def _fake_qc(
    artifacts: Any, manifest: Any, config: Any, *, source_path: Any = None
) -> TechnicalQCResult:
    return TechnicalQCResult(status="PASS", checks=(), reason_codes=(), policy_version="test")


def _make_executor(
    session: Session,
    runner: _FakeRunner,
    *,
    runtime_factory: Any,
) -> Any:
    settings = get_settings()
    return build_render_execution_executor(
        session,
        StorageService(settings.storage_root),
        settings,
        admission=NullRenderAdmission(),
        compiler=None,
        runner=runner,
        qc_checker=_fake_qc,
        runtime_factory=runtime_factory,
    )


def _runtime_factory_for(fixture: Stage51Fixture) -> Any:
    from app.render.execution.service import resolve_runtime_identity

    settings = get_settings()

    def factory(
        _settings: Any, *, source_absolute_path: str = "", attempt_directory: str = ""
    ) -> RuntimeIdentity:
        return resolve_runtime_identity(
            settings,
            source_absolute_path=source_absolute_path,
            attempt_directory=attempt_directory,
        )

    return factory


def test_queue_and_execute_completes_with_injected_seams(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert outcome.queued is True and outcome.job_id is not None
    runner = _FakeRunner()
    executor = _make_executor(session, runner, runtime_factory=_runtime_factory_for(fixture))
    executor.set_active_job(outcome.job_id)
    executor.execute(outcome.render_execution_id)
    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None
    assert row.lifecycle is RenderExecutionLifecycle.COMPLETE
    assert row.cache_eligible is True
    assert row.publication_ready is False
    assert row.stage6_implemented is False
    assert runner.calls == 1
    assert row.artifact_reference.get("relative_path")
    assert row.input_fingerprint


def test_duplicate_delivery_performs_no_render_work(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    runner = _FakeRunner()
    first = _make_executor(session, runner, runtime_factory=_runtime_factory_for(fixture))
    first.set_active_job(outcome.job_id)
    first.execute(outcome.render_execution_id)
    second = _make_executor(session, runner, runtime_factory=_runtime_factory_for(fixture))
    second.set_active_job(outcome.job_id)
    second.execute(outcome.render_execution_id)
    assert second.skipped_duplicate is True
    assert runner.calls == 1


def test_queued_cancellation_stops_work(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert outcome.job_id is not None
    job = session.get(ProcessingJob, outcome.job_id)
    assert job is not None
    job.status = JobStatus.CANCELLED
    session.commit()
    runner = _FakeRunner()
    executor = _make_executor(session, runner, runtime_factory=_runtime_factory_for(fixture))
    executor.set_active_job(outcome.job_id)
    with pytest.raises(StageCancelled):
        executor.execute(outcome.render_execution_id)
    assert runner.calls == 0
    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None and row.lifecycle is RenderExecutionLifecycle.CANCELLED


def test_old_claim_cannot_fail_newer_run(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert outcome.job_id is not None
    runner = _FakeRunner()
    executor = _make_executor(session, runner, runtime_factory=_runtime_factory_for(fixture))
    executor.set_active_job(outcome.job_id)
    assert executor._claim_job() is True
    executor._row_id = outcome.render_execution_id
    old_version = executor._claim_version
    session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == outcome.job_id)
        .values(claim_version=old_version + 5)
    )
    session.commit()
    executor.record_failure(RuntimeError("boom"))
    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None and row.lifecycle is not RenderExecutionLifecycle.FAILED
    job = session.get(ProcessingJob, outcome.job_id)
    assert job is not None and job.status is not JobStatus.FAILED


def test_source_change_prevents_success(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    source = fixture.stage50.source_path
    source.write_bytes(source.read_bytes() + b"changed")
    runner = _FakeRunner()
    executor = _make_executor(session, runner, runtime_factory=_runtime_factory_for(fixture))
    executor.set_active_job(outcome.job_id)
    executor.execute(outcome.render_execution_id)
    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None
    assert row.lifecycle is not RenderExecutionLifecycle.COMPLETE
    assert row.cache_eligible is False
    assert runner.calls == 0


def test_cache_eligible_completed_row_is_reused(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    runner = _FakeRunner()
    executor = _make_executor(session, runner, runtime_factory=_runtime_factory_for(fixture))
    executor.set_active_job(outcome.job_id)
    executor.execute(outcome.render_execution_id)
    cached = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert cached.cached is True
    assert cached.job_id is None
    view = read_render_execution(session, fixture.stage50.selection.candidate.id)
    assert view is not None and view.effective is True
