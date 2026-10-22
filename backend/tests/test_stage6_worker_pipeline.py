"""Stage 6 persisted execution milestone through actual worker-task entry points.

Unlike ``test_stage6_real_executor.py`` (which invokes the executors directly),
this drives the *Celery worker-task callables* through the full controlled chain:

    queue -> ``run_materialization_job`` (prepare) -> render dispatch ->
    ``run_render_execution`` (real FFmpeg compiler/runner/probe/QC) ->
    render-completion callback -> ``run_materialization_finalization`` (shared
    router) -> final assessment.

Broker delivery is controlled (the render job is enqueued to the running broker
but executed synchronously here). Real generated media; SYNTHETIC INTEGRATION
evidence, not product acceptance.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    seed_stage51,
)
from test_stage6_real_executor import _write_source

from app.composition.executor import build_visual_composition_executor
from app.composition.queue import get_or_create_plan_row
from app.core.enums import JobKind, JobStatus
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization import queue as queue_module
from app.materialization.providers import FakeCritiqueProvider, FakeWritingProvider
from app.materialization.queue import queue_materialization
from app.models import ClipMaterialization, ProcessingJob
from app.models.render_execution import RenderExecution

_ANNOTATION_COPY = "Promotion rates fell after teams changed."
_TERMINAL = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _ready_composition(session: Session, fixture: Any) -> Any:
    candidate = fixture.stage50.selection.candidate
    row = get_or_create_plan_row(session, candidate)
    job = ProcessingJob(
        source_video_id=candidate.source_video_id,
        kind=JobKind.VISUAL_COMPOSITION,
        visual_composition_plan_id=row.id,
    )
    session.add(job)
    session.commit()
    executor = build_visual_composition_executor(
        session,
        fixture.stage50.storage,
        fixture.settings,
        display_probe=fixture.display_probe,
        frame_sampler=FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=FakeDetector(),
    )
    executor.set_active_job(job.id)
    executor.execute(row.id)
    session.refresh(row)
    assert row.plan_ready is True
    return row


def _seed_with_core(engine: Engine, monkeypatch: Any) -> tuple[Any, uuid.UUID]:
    """Seed real media + composition and a preexisting current CORE render."""

    from types import SimpleNamespace

    from app.core.enums import RenderArtifactPurpose, RenderExecutionLifecycle

    session = Session(engine)
    try:
        fixture = seed_stage51(session, monkeypatch, narration="NONE", media_writer=_write_source)
        candidate = fixture.stage50.selection.candidate
        context = SimpleNamespace(
            candidate_id=candidate.id,
            source_video_id=candidate.source_video_id,
            storage=fixture.stage50.storage,
        )
        _ready_composition(session, fixture)
        core = RenderExecution(
            id=uuid.uuid4(),
            source_video_id=candidate.source_video_id,
            clip_candidate_id=candidate.id,
            artifact_purpose=RenderArtifactPurpose.CORE_SOURCE_VALIDATION,
            lifecycle=RenderExecutionLifecycle.COMPLETE,
            qc_status="PASS",
            is_current=True,
            delivery_profile_key="MP4_H264_AAC_1080X1920_V1",
            input_fingerprint="c" * 64,
            output_fingerprint="d" * 64,
            qc_fingerprint="e" * 64,
            artifact_reference={"relative_path": "unused.mp4", "sha256": "a" * 64},
            qc_result={"status": "PASS", "measured": {}},
        )
        session.add(core)
        session.commit()
        return context, core.id
    finally:
        session.close()


def test_worker_task_chain_end_to_end(engine: Engine, monkeypatch: Any) -> None:
    settings = get_settings()
    storage = settings.storage_root
    from app.services.storage import StorageService

    storage_service = StorageService(storage)

    monkeypatch.setattr(queue_module, "_dispatch", lambda *a, **k: None)
    # Controlled provider injection: the production task builds real providers.
    import app.workers.tasks as tasks_module
    from app.materialization.executor import build_materialization_executor as _real_builder

    def _builder(session: Session, storage: Any, settings: Any, **kwargs: Any) -> Any:
        return _real_builder(
            session,
            storage,
            settings,
            writing=FakeWritingProvider({1: _ANNOTATION_COPY}),
            critique=FakeCritiqueProvider(accepted={1: True}),
        )

    monkeypatch.setattr(tasks_module, "build_materialization_executor", _builder, raising=False)
    monkeypatch.setattr("app.materialization.executor.build_materialization_executor", _builder)

    context, core_render_id = _seed_with_core(engine, monkeypatch)
    candidate_id = context.candidate_id

    # 1. Queue -> prepare job.
    session = Session(engine)
    try:
        outcome = queue_materialization(session, candidate_id, priority="CANDIDATE")
        session.commit()
        assert outcome.job_id is not None
        prepare_job_id = outcome.job_id
        materialization_id = outcome.materialization_id
    finally:
        session.close()

    # 2. Prepare via the actual worker task.
    from app.workers.tasks import run_materialization_job

    run_materialization_job(str(candidate_id), str(prepare_job_id))

    session = Session(engine)
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        assert envelope.lifecycle.value == "AWAITING_RENDER", envelope.reason_codes
        render_id = envelope.render_execution_id
        assert render_id is not None
        render_job = session.scalars(
            select(ProcessingJob).where(ProcessingJob.render_execution_id == render_id)
        ).first()
        assert render_job is not None
        render_job_id = render_job.id
    finally:
        session.close()

    # 3. Real render via the actual worker task (FFmpeg/compiler/runner/probe/QC).
    from app.workers.tasks import run_render_execution

    run_render_execution(str(render_id), str(render_job_id))

    session = Session(engine)
    try:
        render = session.get(RenderExecution, render_id)
        assert render is not None
        assert render.lifecycle.value == "COMPLETE", render.reason_codes
        measured = (render.qc_result or {}).get("measured") or {}
        assert isinstance(measured.get("final_integrated_lufs"), (int, float))
        assert isinstance(measured.get("final_true_peak_dbtp"), (int, float))
        # Coexistence: the transformed row does not demote the core row.
        assert render.artifact_purpose.value == "TRANSFORMED_SHORT"
        core = session.get(RenderExecution, core_render_id)
        assert core is not None and core.is_current is True
        # Finalizer job was requested by the render-completion callback.
        finalizer_job = session.scalars(
            select(ProcessingJob)
            .where(
                ProcessingJob.materialization_id == materialization_id,
                ProcessingJob.kind == JobKind.MATERIALIZATION,
                ProcessingJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
            )
            .order_by(ProcessingJob.created_at.desc())
        ).first()
        assert finalizer_job is not None
        finalizer_job_id = finalizer_job.id
    finally:
        session.close()

    # 4. Finalize via the actual worker task (shared durable router).
    from app.workers.tasks import run_materialization_finalization

    run_materialization_finalization(str(candidate_id), str(finalizer_job_id))

    session = Session(engine)
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        assert envelope.lifecycle.value == "COMPLETE"
        assert envelope.publication_ready is False
        assert envelope.semantic_outcome is not None
        assert envelope.semantic_outcome.value in {
            "READY_FOR_HUMAN_REVIEW",
            "MATERIALIZED_WITH_CAUTION",
        }
        assert envelope.human_review_required is True
        first_fingerprint = (envelope.metrics or {}).get("assessment_fingerprint")
        assert first_fingerprint
        # Every handled job is terminal.
        for job in session.scalars(
            select(ProcessingJob).where(ProcessingJob.materialization_id == materialization_id)
        ).all():
            assert job.status in _TERMINAL
        for job in session.scalars(
            select(ProcessingJob).where(ProcessingJob.render_execution_id == render_id)
        ).all():
            assert job.status in _TERMINAL
    finally:
        session.close()

    # 5. Replay: the finalizer task is idempotent and makes zero provider/render calls.
    from datetime import datetime, timezone

    from app.materialization.repository import bind_active_job, claim_materialization

    session = Session(engine)
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        envelope.active_job_id = None
        replay_job = ProcessingJob(
            source_video_id=envelope.source_video_id,
            kind=JobKind.MATERIALIZATION,
            materialization_id=envelope.id,
            status=JobStatus.RUNNING,
            claim_version=1,
            started_at=datetime.now(timezone.utc),
            heartbeat_at=datetime.now(timezone.utc),
        )
        session.add(replay_job)
        session.commit()
        bind_active_job(session, envelope.id, replay_job.id)
        session.commit()
        owner = claim_materialization(
            session, envelope.id, replay_job.id, expected_job_claim_version=1
        )
        assert owner is not None
        from app.materialization.finalization import finalize_materialization

        result = finalize_materialization(
            session, owner, settings=settings, storage=storage_service
        )
        session.commit()
        assert result.reused is True
        assert result.assessment_fingerprint == first_fingerprint
    finally:
        session.close()


def test_worker_task_render_cancellation_is_truthful(engine: Engine, monkeypatch: Any) -> None:
    """An operator cancel before the render task runs keeps work cancelled/terminal."""

    monkeypatch.setattr(queue_module, "_dispatch", lambda *a, **k: None)
    import app.workers.tasks as tasks_module
    from app.materialization.executor import build_materialization_executor as _real_builder

    def _builder(session: Session, storage: Any, settings: Any, **kwargs: Any) -> Any:
        return _real_builder(
            session,
            storage,
            settings,
            writing=FakeWritingProvider({1: _ANNOTATION_COPY}),
            critique=FakeCritiqueProvider(accepted={1: True}),
        )

    monkeypatch.setattr(tasks_module, "build_materialization_executor", _builder, raising=False)
    monkeypatch.setattr("app.materialization.executor.build_materialization_executor", _builder)

    context, _core = _seed_with_core(engine, monkeypatch)
    candidate_id = context.candidate_id
    session = Session(engine)
    try:
        outcome = queue_materialization(session, candidate_id, priority="CANDIDATE")
        session.commit()
        prepare_job_id = outcome.job_id
        materialization_id = outcome.materialization_id
    finally:
        session.close()

    from app.workers.tasks import run_materialization_job

    run_materialization_job(str(candidate_id), str(prepare_job_id))

    session = Session(engine)
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        render_id = envelope.render_execution_id
        render_job = session.scalars(
            select(ProcessingJob).where(ProcessingJob.render_execution_id == render_id)
        ).first()
        render_job_id = render_job.id
        # Cancel the render job before the task runs.
        render_job.status = JobStatus.CANCELLED
        session.commit()
    finally:
        session.close()

    from app.workers.tasks import run_render_execution

    result = run_render_execution(str(render_id), str(render_job_id))
    assert result.get("cancelled") is True

    session = Session(engine)
    try:
        render = session.get(RenderExecution, render_id)
        assert render.lifecycle.value != "COMPLETE"
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope.human_review_required is False
        assert envelope.publication_ready is False
    finally:
        session.close()
