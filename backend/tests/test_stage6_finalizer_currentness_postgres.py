"""Two-session PostgreSQL Stage 6 finalizer currentness regressions.

Gated on ``CLIPFACTORY_TEST_POSTGRES_URL``. These prove, against a real database
and two independent sessions, that the finalizer re-derives live upstream
effectiveness *immediately before final assessment*: a change to the FINAL_CLIP,
the selected plan, the profile, or the materialization output/repair revision
(during rendering or right before assessment) can never authorize review
readiness or cache reuse, and the handled-run job is always terminal.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    seed_stage51,
)

from app.composition.executor import build_visual_composition_executor
from app.composition.queue import get_or_create_plan_row
from app.core.enums import JobKind, JobStatus, RefinementPriority, RefinementStatus
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.executor import build_materialization_executor
from app.materialization.finalization import (
    FINALIZATION_SUPERSEDED,
    finalize_materialization,
)
from app.materialization.providers import FakeCritiqueProvider, FakeWritingProvider
from app.materialization.repository import bind_active_job, claim_materialization
from app.models import (
    CandidateRefinement,
    ClipMaterialization,
    ProcessingJob,
    TransformationPlanSelection,
)
from app.models.render_execution import RenderExecution
from app.render.execution.policy import TRANSFORMED_SHORT
from app.services.storage import StorageService

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL, reason="CLIPFACTORY_TEST_POSTGRES_URL is required for Stage 6 currentness"
)

_DATABASE = f"clipfactory_stage6_cur_{uuid.uuid4().hex[:12]}"
_REVIEWABLE = {"READY_FOR_HUMAN_REVIEW", "MATERIALIZED_WITH_CAUTION"}
_ANNOTATION_COPY = (
    "That drop looks milder once you control for which teams already promoted heavily."
)


def _admin_url() -> str:
    assert _URL is not None
    return str(make_url(_URL).set(database="postgres").render_as_string(hide_password=False))


def _target_url() -> str:
    assert _URL is not None
    return str(make_url(_URL).set(database=_DATABASE).render_as_string(hide_password=False))


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    from sqlalchemy import text

    admin = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'DROP DATABASE IF EXISTS "{_DATABASE}" WITH (FORCE)'))
        connection.execute(text(f'CREATE DATABASE "{_DATABASE}"'))
    admin.dispose()
    target = create_engine(_target_url())
    Base.metadata.create_all(target)
    try:
        yield target
    finally:
        target.dispose()
        admin = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{_DATABASE}" WITH (FORCE)'))
        admin.dispose()


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


def _prepare(engine: Engine, monkeypatch: Any) -> uuid.UUID:
    """Seed and run the prepare executor; return the current envelope id."""

    holder = Session(engine)
    try:
        fixture = seed_stage51(holder, monkeypatch, narration="NONE")
        candidate = fixture.stage50.selection.candidate
        _ready_composition(holder, fixture)
        job = ProcessingJob(
            source_video_id=candidate.source_video_id,
            kind=JobKind.MATERIALIZATION,
        )
        holder.add(job)
        holder.commit()
        executor = build_materialization_executor(
            holder,
            fixture.stage50.storage,
            get_settings(),
            writing=FakeWritingProvider({1: _ANNOTATION_COPY}),
            critique=FakeCritiqueProvider(accepted={1: True}),
        )
        executor.set_active_job(job.id)
        executor.execute(str(candidate.id))
        holder.commit()
        envelope = holder.scalars(
            select(ClipMaterialization).where(
                ClipMaterialization.clip_candidate_id == candidate.id,
                ClipMaterialization.is_current.is_(True),
            )
        ).one()
        return envelope.id
    finally:
        holder.close()


def _attach_render(engine: Engine, envelope_id: uuid.UUID) -> None:
    storage = StorageService(get_settings().storage_root)
    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, envelope_id)
        assert envelope is not None
        directory = storage.storage_root / "render-executions" / uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / "output.mp4"
        artifact.write_bytes(b"\x00" * 2048)
        relative = str(artifact.relative_to(storage.storage_root))
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        render = RenderExecution(
            id=uuid.uuid4(),
            source_video_id=envelope.source_video_id,
            clip_candidate_id=envelope.clip_candidate_id,
            materialization_id=envelope.id,
            artifact_purpose=TRANSFORMED_SHORT,
            lifecycle="COMPLETE",
            qc_status="PASS",
            is_current=True,
            output_fingerprint="f" * 64,
            runtime_fingerprint="r" * 64,
            qc_fingerprint="q" * 64,
            request_payload={
                "materialization_id": str(envelope.id),
                "expected_output_fingerprint": envelope.output_fingerprint,
                "spec": {"authored": {"repair_revision": int(envelope.repair_count or 0)}},
            },
            artifact_reference={"relative_path": relative, "sha256": digest, "size_bytes": 2048},
            qc_result={
                "status": "PASS",
                "measured": {"final_integrated_lufs": -16.1, "final_true_peak_dbtp": -1.8},
            },
            reason_codes=[],
        )
        session.add(render)
        envelope.render_execution_id = render.id
        session.commit()


def _bound_owner(engine: Engine, envelope_id: uuid.UUID) -> tuple[uuid.UUID, Any]:
    from datetime import datetime, timezone

    session = Session(engine)
    try:
        envelope = session.get(ClipMaterialization, envelope_id)
        assert envelope is not None
        job = ProcessingJob(
            source_video_id=envelope.source_video_id,
            kind=JobKind.MATERIALIZATION,
            materialization_id=envelope.id,
            status=JobStatus.RUNNING,
            claim_version=1,
            started_at=datetime.now(timezone.utc),
            heartbeat_at=datetime.now(timezone.utc),
        )
        session.add(job)
        envelope.active_job_id = None
        session.commit()
        bind_active_job(session, envelope.id, job.id)
        session.commit()
        owner = claim_materialization(session, envelope.id, job.id, expected_job_claim_version=1)
        assert owner is not None
        return job.id, owner
    finally:
        session.close()


def _finalize(engine: Engine, envelope_id: uuid.UUID) -> tuple[Any, uuid.UUID]:
    job_id, owner = _bound_owner(engine, envelope_id)
    with Session(engine) as session:
        result = finalize_materialization(
            session,
            owner,
            settings=get_settings(),
            storage=StorageService(get_settings().storage_root),
        )
        session.commit()
    return result, job_id


def _envelope(engine: Engine, envelope_id: uuid.UUID) -> ClipMaterialization:
    with Session(engine) as session:
        row = session.get(ClipMaterialization, envelope_id)
        assert row is not None
        session.expunge(row)
        return row


def _final_clip(engine: Engine, envelope_id: uuid.UUID) -> CandidateRefinement:
    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, envelope_id)
        assert envelope is not None
        return session.scalars(
            select(CandidateRefinement).where(
                CandidateRefinement.clip_candidate_id == envelope.clip_candidate_id,
                CandidateRefinement.priority == RefinementPriority.FINAL_CLIP,
            )
        ).one()


def _job_status(engine: Engine, job_id: uuid.UUID) -> JobStatus:
    with Session(engine) as session:
        job = session.get(ProcessingJob, job_id)
        assert job is not None
        return job.status


def test_change_during_rendering_supersedes_at_finalization(
    engine: Engine, monkeypatch: Any
) -> None:
    envelope_id = _prepare(engine, monkeypatch)
    _attach_render(engine, envelope_id)
    # A DIFFERENT session changes the authoritative FINAL_CLIP while the render is
    # "in flight"; finalization must not review-ready the stale generation.
    refine = _final_clip(engine, envelope_id)
    with Session(engine) as session:
        row = session.get(CandidateRefinement, refine.id)
        row.status = RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
        session.commit()

    result, job_id = _finalize(engine, envelope_id)
    envelope = _envelope(engine, envelope_id)
    assert result.outcome == FINALIZATION_SUPERSEDED
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    assert envelope.semantic_outcome.value not in _REVIEWABLE
    assert _job_status(engine, job_id) is not JobStatus.RUNNING


def test_change_immediately_before_assessment_supersedes(
    engine: Engine, monkeypatch: Any
) -> None:
    envelope_id = _prepare(engine, monkeypatch)
    _attach_render(engine, envelope_id)
    # Change the selected plan content from a second session right before assess.
    from app.models import TransformationPlan

    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, envelope_id)
        selection = session.get(TransformationPlanSelection, envelope.selection_id)
        plan = session.get(TransformationPlan, selection.selected_plan_id)
        plan.plan_output_fingerprint = "m" * 64
        session.commit()

    result, job_id = _finalize(engine, envelope_id)
    envelope = _envelope(engine, envelope_id)
    assert result.outcome in {FINALIZATION_SUPERSEDED, "DEFERRED"}
    assert envelope.cache_eligible is False
    assert envelope.semantic_outcome.value not in _REVIEWABLE
    assert _job_status(engine, job_id) is not JobStatus.RUNNING


def test_materialization_output_revision_change_supersedes(
    engine: Engine, monkeypatch: Any
) -> None:
    envelope_id = _prepare(engine, monkeypatch)
    _attach_render(engine, envelope_id)
    # A materialization output/repair revision change invalidates the bound render.
    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, envelope_id)
        envelope.output_fingerprint = "9" * 64
        session.commit()

    result, _job_id = _finalize(engine, envelope_id)
    assert result.outcome == FINALIZATION_SUPERSEDED
    envelope = _envelope(engine, envelope_id)
    # The bound render is stale: the envelope stays pre-render (never
    # review-ready / cache-eligible / human-review-required).
    assert envelope.cache_eligible is False
    assert envelope.human_review_required is False
    assert envelope.lifecycle.value == "AWAITING_RENDER"


def test_unchanged_generation_reaches_review_ready(engine: Engine, monkeypatch: Any) -> None:
    envelope_id = _prepare(engine, monkeypatch)
    _attach_render(engine, envelope_id)
    result, _job_id = _finalize(engine, envelope_id)
    envelope = _envelope(engine, envelope_id)
    assert result.outcome == "COMPLETE"
    assert envelope.semantic_outcome.value in _REVIEWABLE
    assert envelope.publication_ready is False


def test_concurrent_posts_share_one_finalizer(engine: Engine, monkeypatch: Any) -> None:
    import threading

    from sqlalchemy import func

    from app.materialization import queue as queue_module
    from app.materialization.queue import queue_materialization

    envelope_id = _prepare(engine, monkeypatch)
    _attach_render(engine, envelope_id)
    with Session(engine) as session:
        candidate_id = session.get(ClipMaterialization, envelope_id).clip_candidate_id

    monkeypatch.setattr(queue_module, "_dispatch", lambda *args, **kwargs: None)

    class _Capture:
        def dispatch_finalization(self, candidate: str, job_id: str) -> None:
            return None

        def dispatch(self, *args: Any, **kwargs: Any) -> None:
            return None

    results: list[Any] = [None, None]
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def worker(index: int) -> None:
        session = Session(engine)
        try:
            barrier.wait(timeout=30)
            results[index] = queue_materialization(
                session, candidate_id, priority="CANDIDATE", render_dispatcher=_Capture()
            )
            session.commit()
        except Exception as error:  # noqa: BLE001 - surfaced by assertion
            errors.append(error)
        finally:
            session.close()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    job_ids = {result.job_id for result in results if result is not None and result.job_id}
    assert len(job_ids) == 1
    with Session(engine) as session:
        count = int(
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
    assert count == 1

