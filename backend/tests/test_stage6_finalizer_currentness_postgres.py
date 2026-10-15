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
    FINALIZATION_CANCELLED,
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


# --- 4. publication-race two-session synchronization --------------------------


def _production_session(engine: Engine) -> Session:
    """A session with the production application semantics (no autoflush/expire)."""

    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return factory()


def _finalize_in_production_session(
    engine: Engine, envelope_id: uuid.UUID
) -> tuple[Any, uuid.UUID]:
    job_id, owner = _bound_owner(engine, envelope_id)
    session = _production_session(engine)
    try:
        result = finalize_materialization(
            session,
            owner,
            settings=get_settings(),
            storage=StorageService(get_settings().storage_root),
        )
        session.commit()
    finally:
        session.close()
    return result, job_id


@pytest.mark.parametrize("mutation", ["final_clip", "plan", "output_revision", "cancel"])
def test_change_after_revalidation_before_publication_refuses(
    engine: Engine, monkeypatch: Any, mutation: str
) -> None:
    """A change committed after initial revalidation but before publication must
    never authorize review readiness: the locked publication revalidation refuses
    it (or serializes), and the owned job stays terminal.
    """

    import app.materialization.finalization as fin

    envelope_id = _prepare(engine, monkeypatch)
    _attach_render(engine, envelope_id)
    original = fin._build_assessment_context
    fired = {"done": False}

    def hook(*args: Any, **kwargs: Any) -> Any:
        # Runs after the lock-free upstream revalidation and before the locked
        # publication transaction, from an independent committed session.
        if not fired["done"]:
            fired["done"] = True
            mut = _production_session(engine)
            try:
                envelope = mut.get(ClipMaterialization, envelope_id)
                assert envelope is not None
                if mutation == "final_clip":
                    from app.core.enums import RefinementStatus as _RS

                    refine = mut.scalars(
                        select(CandidateRefinement).where(
                            CandidateRefinement.clip_candidate_id == envelope.clip_candidate_id,
                            CandidateRefinement.priority == RefinementPriority.FINAL_CLIP,
                        )
                    ).one()
                    refine.status = _RS.NEEDS_MANUAL_TRANSCRIPT_REVIEW
                elif mutation == "plan":
                    from app.models import TransformationPlan

                    selection = mut.get(TransformationPlanSelection, envelope.selection_id)
                    plan = mut.get(TransformationPlan, selection.selected_plan_id)
                    plan.plan_output_fingerprint = "m" * 64
                elif mutation == "output_revision":
                    envelope.output_fingerprint = "9" * 64
                else:
                    from app.core.enums import MaterializationLifecycle

                    envelope.lifecycle = MaterializationLifecycle.CANCELLED
                mut.commit()
            finally:
                mut.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(fin, "_build_assessment_context", hook)
    result, job_id = _finalize_in_production_session(engine, envelope_id)
    envelope = _envelope(engine, envelope_id)

    # A pre-render semantic label (MATERIALIZED_WITH_CAUTION) is legal and is NOT
    # review readiness: the label alone never authorizes artifact exposure or
    # human review. Readiness requires an effective completed assessment plus the
    # exact bindings and flags, so assert the exact disposition instead of
    # conflating the pre-render label with readiness.
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False
    # No effective completed assessment was ever published for the refused
    # generation, so the render artifact is never eligible for review/cache.
    assert not (envelope.metrics or {}).get("assessment_fingerprint")
    assert envelope.render_execution_id is None or not envelope.cache_eligible

    if mutation in {"final_clip", "plan"}:
        # A changed authoritative prerequisite is refused under the lock and the
        # generation is truthfully blocked for upstream revalidation.
        assert result.outcome == FINALIZATION_SUPERSEDED
        assert "UPSTREAM_REVALIDATION_REQUIRED" in result.reason_codes
        assert envelope.lifecycle.value == "BLOCKED"
        assert envelope.semantic_outcome.value == "UPSTREAM_REVALIDATION_REQUIRED"
        assert _job_status(engine, job_id) is JobStatus.SUCCEEDED
    elif mutation == "output_revision":
        # The bound render no longer matches the changed output revision.
        assert result.outcome == FINALIZATION_SUPERSEDED
        assert "RENDER_BINDING_STALE" in result.reason_codes
        assert envelope.lifecycle.value == "AWAITING_RENDER"
        assert envelope.semantic_outcome.value == "MATERIALIZED_WITH_CAUTION"
        assert _job_status(engine, job_id) is JobStatus.SUCCEEDED
    else:  # cancel
        assert result.outcome == FINALIZATION_CANCELLED
        assert "MATERIALIZATION_CANCELLED" in result.reason_codes
        assert envelope.lifecycle.value == "CANCELLED"
        assert envelope.semantic_outcome.value == "MATERIALIZED_WITH_CAUTION"
        assert _job_status(engine, job_id) is JobStatus.CANCELLED


def test_unchanged_publication_with_production_session_reaches_review_ready(
    engine: Engine, monkeypatch: Any
) -> None:
    envelope_id = _prepare(engine, monkeypatch)
    _attach_render(engine, envelope_id)
    result, _job_id = _finalize_in_production_session(engine, envelope_id)
    envelope = _envelope(engine, envelope_id)
    assert result.outcome == "COMPLETE"
    assert envelope.semantic_outcome.value in _REVIEWABLE
    assert envelope.publication_ready is False



def _autocommit_connection(engine: Engine) -> Any:
    return engine.connect().execution_options(isolation_level="AUTOCOMMIT")


def _wait_for_lock_waiter(engine: Engine, holder_xid: int, *, timeout: float = 15.0) -> bool:
    """Block until a backend is waiting on the exact holder transaction id.

    This is the deterministic boundary handshake: it proves the finalizer is
    genuinely waiting on the intended PostgreSQL row lock (a not-granted lock on
    the holder's transaction id) before the holder is released, rather than
    relying on an elapsed sleep.
    """

    import time

    from sqlalchemy import text

    connection = _autocommit_connection(engine)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            waiting = connection.execute(
                text(
                    "SELECT count(*) FROM pg_locks "
                    "WHERE NOT granted AND transactionid = :xid"
                ),
                {"xid": holder_xid},
            ).scalar()
            if waiting and int(waiting) > 0:
                return True
            time.sleep(0.05)
    finally:
        connection.close()
    return False


def test_publication_waits_on_concurrent_candidate_lock_then_refuses(
    engine: Engine, monkeypatch: Any
) -> None:
    """A real synchronized interleaving at the candidate lock boundary.

    A second session holds the candidate row lock, mutates the authoritative
    FINAL_CLIP, and commits while the finalizer is *verifiably* blocked inside its
    publication path (a not-granted PostgreSQL lock on the holder's transaction
    id). The finalizer then revalidates under the lock and refuses to publish the
    now-stale generation. Bounded ``lock_timeout``/``statement_timeout`` turn any
    deadlock into a failure, and all threads must terminate.
    """

    import threading

    from sqlalchemy import text

    envelope_id = _prepare(engine, monkeypatch)
    _attach_render(engine, envelope_id)

    invalidator_locked = threading.Event()
    release = threading.Event()
    errors: list[Exception] = []
    holder: dict[str, int] = {}

    def invalidator() -> None:
        session = _production_session(engine)
        try:
            session.execute(text("SET lock_timeout = '15000ms'"))
            session.execute(text("SET statement_timeout = '20000ms'"))
            envelope = session.get(ClipMaterialization, envelope_id)
            assert envelope is not None
            # Take the shared candidate lock first (same order the finalizer uses).
            session.execute(
                text("SELECT id FROM clip_candidates WHERE id = :cid FOR UPDATE"),
                {"cid": str(envelope.clip_candidate_id)},
            ).first()
            holder["xid"] = int(session.execute(text("SELECT txid_current()")).scalar())
            invalidator_locked.set()
            release.wait(timeout=20)
            refine = session.scalars(
                select(CandidateRefinement).where(
                    CandidateRefinement.clip_candidate_id == envelope.clip_candidate_id,
                    CandidateRefinement.priority == RefinementPriority.FINAL_CLIP,
                )
            ).one()
            refine.status = RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
            session.commit()
        except Exception as error:  # noqa: BLE001 - surfaced by assertion
            errors.append(error)
            try:
                session.rollback()
            except Exception:
                pass
        finally:
            session.close()

    thread = threading.Thread(target=invalidator)
    thread.start()
    assert invalidator_locked.wait(timeout=15)
    assert "xid" in holder

    outcome_holder: dict[str, Any] = {}

    def finalize() -> None:
        try:
            result, job_id = _finalize_in_production_session(engine, envelope_id)
            outcome_holder["result"] = result
            outcome_holder["job_id"] = job_id
        except Exception as error:  # noqa: BLE001 - surfaced by assertion
            errors.append(error)

    finalizer_thread = threading.Thread(target=finalize)
    finalizer_thread.start()

    # Deterministic handshake: the finalizer must be genuinely blocked on the
    # intended PostgreSQL lock before the holder is released.
    assert _wait_for_lock_waiter(engine, holder["xid"], timeout=15)
    assert "result" not in outcome_holder
    assert finalizer_thread.is_alive()
    release.set()

    finalizer_thread.join(timeout=30)
    thread.join(timeout=30)
    assert not finalizer_thread.is_alive()
    assert not thread.is_alive()
    assert not errors, errors

    result = outcome_holder["result"]
    job_id = outcome_holder["job_id"]
    assert result.outcome == FINALIZATION_SUPERSEDED
    assert "UPSTREAM_REVALIDATION_REQUIRED" in result.reason_codes
    envelope = _envelope(engine, envelope_id)
    assert envelope.lifecycle.value == "BLOCKED"
    assert envelope.semantic_outcome.value == "UPSTREAM_REVALIDATION_REQUIRED"
    assert envelope.human_review_required is False
    assert envelope.cache_eligible is False
    assert envelope.publication_ready is False
    assert not (envelope.metrics or {}).get("assessment_fingerprint")
    assert _job_status(engine, job_id) is JobStatus.SUCCEEDED


def test_publication_and_queue_share_lock_order_without_deadlock(
    engine: Engine, monkeypatch: Any
) -> None:
    """Publication and a live queue POST acquire the candidate lock in the same
    order and must serialize without deadlock.

    A real ``queue_materialization`` POST holds the candidate row lock (in the
    production order) while the finalizer enters its publication path; the
    finalizer must wait on the same lock and then proceed -- never invert the lock
    order. Both threads must terminate within bounded timeouts.
    """

    import threading

    from sqlalchemy import text as _text

    import app.materialization.queue as queue_module
    from app.materialization.queue import queue_materialization

    envelope_id = _prepare(engine, monkeypatch)
    _attach_render(engine, envelope_id)

    queue_locked = threading.Event()
    queue_release = threading.Event()
    errors: list[Exception] = []
    holder: dict[str, int] = {}

    original_lock_candidate = queue_module._lock_candidate

    def hook_lock_candidate(session: Any, candidate_id: Any) -> Any:
        candidate = original_lock_candidate(session, candidate_id)
        holder["xid"] = int(session.execute(_text("SELECT txid_current()")).scalar())
        queue_locked.set()
        queue_release.wait(timeout=20)
        return candidate

    monkeypatch.setattr(queue_module, "_lock_candidate", hook_lock_candidate)
    monkeypatch.setattr(queue_module, "_dispatch", lambda *args, **kwargs: None)

    with Session(engine) as session:
        candidate_id = session.get(ClipMaterialization, envelope_id).clip_candidate_id

    queue_outcome: dict[str, Any] = {}

    def queue_post() -> None:
        session = _production_session(engine)
        try:
            session.execute(_text("SET lock_timeout = '15000ms'"))
            session.execute(_text("SET statement_timeout = '20000ms'"))
            queue_outcome["result"] = queue_materialization(
                session, candidate_id, priority="CANDIDATE"
            )
            session.commit()
        except Exception as error:  # noqa: BLE001 - surfaced by assertion
            errors.append(error)
            try:
                session.rollback()
            except Exception:
                pass
        finally:
            session.close()

    queue_thread = threading.Thread(target=queue_post)
    queue_thread.start()
    assert queue_locked.wait(timeout=15), errors

    outcome_holder: dict[str, Any] = {}

    def finalize() -> None:
        try:
            result, _job = _finalize_in_production_session(engine, envelope_id)
            outcome_holder["result"] = result
        except Exception as error:  # noqa: BLE001 - surfaced by assertion
            errors.append(error)

    finalizer_thread = threading.Thread(target=finalize)
    finalizer_thread.start()

    # The finalizer must be verifiably waiting on the candidate lock the queue POST
    # holds before we release it, so the interleaving is deterministic.
    assert _wait_for_lock_waiter(engine, holder["xid"], timeout=15)
    assert "result" not in outcome_holder
    assert finalizer_thread.is_alive()
    queue_release.set()

    finalizer_thread.join(timeout=40)
    queue_thread.join(timeout=40)
    assert not finalizer_thread.is_alive()
    assert not queue_thread.is_alive()
    assert not errors, errors

    # Neither party deadlocked; publication produced a truthful outcome.
    assert "result" in outcome_holder
    assert outcome_holder["result"].outcome in {
        FINALIZATION_SUPERSEDED,
        "COMPLETE",
        "DEFERRED",
    }
    assert "result" in queue_outcome

