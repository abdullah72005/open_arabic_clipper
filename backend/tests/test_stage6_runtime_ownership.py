"""Stage 6 durable runtime ownership, envelope reuse, and dispatch recovery.

Gated on ``CLIPFACTORY_TEST_POSTGRES_URL``. These regressions exercise the real
persisted prerequisite fixture (source -> ... -> ready Stage 5.1 composition)
through two independent PostgreSQL sessions, so the candidate scope lock, the
envelope-first queue, the atomic ``active_job_id`` CAS, and the paired
job/envelope ownership predicates are all proven against a real database.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, func, select
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
from app.core.enums import JobKind, JobStatus, MaterializationLifecycle
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization import queue as queue_module
from app.materialization.executor import build_materialization_executor
from app.materialization.fingerprints import input_fingerprint
from app.materialization.providers import DeterministicWritingProvider
from app.materialization.queue import (
    DISPATCH_FAILED_MARKER,
    MaterializationQueueOutcome,
    queue_materialization,
)
from app.materialization.repository import MaterializationOwner, guarded_update
from app.materialization.service import assemble_inputs
from app.models import ClipMaterialization, ProcessingJob
from app.services.storage import StorageService

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL, reason="CLIPFACTORY_TEST_POSTGRES_URL is required for Stage 6 ownership"
)

_DATABASE = f"clipfactory_stage6_own_{uuid.uuid4().hex[:12]}"


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


def _seed_ready(engine: Engine, monkeypatch: Any) -> uuid.UUID:
    session = Session(engine)
    try:
        fixture = seed_stage51(session, monkeypatch, narration="NONE")
        candidate_id = fixture.stage50.selection.candidate.id
        _ready_composition(session, fixture)
        session.commit()
        return candidate_id
    finally:
        session.close()


def count_active_materialization_jobs(session: Session, candidate_id: uuid.UUID) -> int:
    envelope_ids = select(ClipMaterialization.id).where(
        ClipMaterialization.clip_candidate_id == candidate_id,
        ClipMaterialization.is_current.is_(True),
    )
    return int(
        session.scalar(
            select(func.count())
            .select_from(ProcessingJob)
            .where(
                ProcessingJob.kind == JobKind.MATERIALIZATION,
                ProcessingJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
                ProcessingJob.materialization_id.in_(envelope_ids),
            )
        )
        or 0
    )


def test_concurrent_enqueue_shares_one_envelope_and_job(engine: Engine, monkeypatch: Any) -> None:
    candidate_id = _seed_ready(engine, monkeypatch)
    dispatched: list[tuple[str, str]] = []
    monkeypatch.setattr(
        queue_module, "_dispatch", lambda cid, jid, **kwargs: dispatched.append((cid, jid))
    )

    barrier = threading.Barrier(2)
    outcomes: list[MaterializationQueueOutcome | None] = [None, None]
    errors: list[Exception] = []

    def worker(index: int) -> None:
        session = Session(engine)
        try:
            barrier.wait(timeout=30)
            outcomes[index] = queue_materialization(session, candidate_id, priority="CANDIDATE")
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
    first, second = outcomes
    assert first is not None and second is not None
    assert first.materialization_id == second.materialization_id
    assert first.job_id == second.job_id
    assert len(dispatched) == 1
    with Session(engine) as session:
        assert count_active_materialization_jobs(session, candidate_id) == 1


def test_broker_failure_is_recovered_by_the_next_request(engine: Engine, monkeypatch: Any) -> None:
    candidate_id = _seed_ready(engine, monkeypatch)
    attempts = {"count": 0}

    def flaky(cid: str, jid: str, **kwargs: Any) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("broker unavailable")

    monkeypatch.setattr(queue_module, "_dispatch", flaky)

    with Session(engine) as session:
        first = queue_materialization(session, candidate_id, priority="CANDIDATE")
        assert first.queued is True
        assert first.job_id is not None
        session.commit()
        job = session.get(ProcessingJob, first.job_id)
        assert job is not None and job.error_code == DISPATCH_FAILED_MARKER
        envelope = session.get(ClipMaterialization, first.materialization_id)
        assert envelope is not None and DISPATCH_FAILED_MARKER in envelope.reason_codes

    # A fresh request redispatchs the SAME job/request and creates no new job.
    with Session(engine) as session:
        second = queue_materialization(session, candidate_id, priority="CANDIDATE")
        session.commit()
        assert second.job_id == first.job_id
        assert second.materialization_id == first.materialization_id
        job = session.get(ProcessingJob, first.job_id)
        assert job is not None and job.error_code is None
        assert count_active_materialization_jobs(session, candidate_id) == 1
    assert attempts["count"] == 2


def test_repeated_broker_failure_keeps_the_marker(engine: Engine, monkeypatch: Any) -> None:
    candidate_id = _seed_ready(engine, monkeypatch)

    def always_fails(cid: str, jid: str, **kwargs: Any) -> None:
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(queue_module, "_dispatch", always_fails)

    with Session(engine) as session:
        first = queue_materialization(session, candidate_id, priority="CANDIDATE")
        session.commit()
    with Session(engine) as session:
        second = queue_materialization(session, candidate_id, priority="CANDIDATE")
        session.commit()
        assert second.job_id == first.job_id
        job = session.get(ProcessingJob, first.job_id)
        assert job is not None and job.error_code == DISPATCH_FAILED_MARKER
        assert count_active_materialization_jobs(session, candidate_id) == 1


def test_new_generation_job_and_stale_worker_rejection(engine: Engine, monkeypatch: Any) -> None:
    candidate_id = _seed_ready(engine, monkeypatch)
    monkeypatch.setattr(queue_module, "_dispatch", lambda cid, jid, **kwargs: None)

    storage = StorageService(get_settings().storage_root)

    # 1. Queue and execute the CANDIDATE generation.
    with Session(engine) as session:
        candidate_outcome = queue_materialization(session, candidate_id, priority="CANDIDATE")
        session.commit()
        assert candidate_outcome.job_id is not None
        job_a = candidate_outcome.job_id
        env_a = candidate_outcome.materialization_id
    with Session(engine) as session:
        executor = build_materialization_executor(
            session, storage, get_settings(), writing=DeterministicWritingProvider()
        )
        executor.set_active_job(job_a)
        executor.execute(str(candidate_id), priority="CANDIDATE")
        session.commit()
        job_row = session.get(ProcessingJob, job_a)
        assert job_row is not None
        assert job_row.status is JobStatus.SUCCEEDED
        assert job_row.claim_version == 1
        assert env_a is not None

    # 2. Queue a genuinely different FINAL_CLIP generation as a NEW job.
    with Session(engine) as session:
        final_outcome = queue_materialization(session, candidate_id, priority="FINAL_CLIP")
        session.commit()
        assert final_outcome.job_id is not None
        assert final_outcome.job_id != job_a
        assert final_outcome.materialization_id != env_a
    with Session(engine) as session:
        executor = build_materialization_executor(
            session, storage, get_settings(), writing=DeterministicWritingProvider()
        )
        executor.set_active_job(final_outcome.job_id)
        executor.execute(str(candidate_id), priority="FINAL_CLIP")
        session.commit()
        job_b = session.get(ProcessingJob, final_outcome.job_id)
        assert job_b is not None and job_b.status is JobStatus.SUCCEEDED
        # Both distinct jobs own their own generation with independent local
        # claim counters (both are 1); no global ordering is compared.
        assert job_b.claim_version == 1
        current = session.scalars(
            select(ClipMaterialization).where(
                ClipMaterialization.clip_candidate_id == candidate_id,
                ClipMaterialization.is_current.is_(True),
            )
        ).all()
        assert len(current) == 1
        assert current[0].id == final_outcome.materialization_id

    # 3. The superseded CANDIDATE worker cannot mutate any row.
    with Session(engine) as session:
        snapshot = session.get(ClipMaterialization, env_a)
        assert snapshot is not None
        before = snapshot.output_fingerprint
        stale_owner = MaterializationOwner(
            materialization_id=env_a,
            job_id=job_a,
            job_claim_version=1,
            materialization_claim_version=1,
        )
        assert guarded_update(session, stale_owner, {"output_fingerprint": "forged"}) is False
        session.rollback()
    with Session(engine) as session:
        after = session.get(ClipMaterialization, env_a)
        assert after is not None and after.output_fingerprint == before
        current = session.scalars(
            select(ClipMaterialization).where(
                ClipMaterialization.clip_candidate_id == candidate_id,
                ClipMaterialization.is_current.is_(True),
            )
        ).one()
        assert current.id == final_outcome.materialization_id


def test_envelope_is_created_before_provider_work(engine: Engine, monkeypatch: Any) -> None:
    candidate_id = _seed_ready(engine, monkeypatch)

    def inspect_envelope(cid: str, jid: str, **kwargs: Any) -> None:
        # At dispatch time the durable envelope and job binding must already exist.
        with Session(engine) as session:
            job = session.get(ProcessingJob, uuid.UUID(jid))
            assert job is not None and job.materialization_id is not None
            envelope = session.get(ClipMaterialization, job.materialization_id)
            assert envelope is not None
            assert envelope.active_job_id == job.id
            assert envelope.lifecycle is MaterializationLifecycle.QUEUED

    monkeypatch.setattr(queue_module, "_dispatch", inspect_envelope)
    with Session(engine) as session:
        outcome = queue_materialization(session, candidate_id, priority="CANDIDATE")
        session.commit()
    assert outcome.materialization_id is not None


def test_input_fingerprint_distinguishes_generations(engine: Engine, monkeypatch: Any) -> None:
    candidate_id = _seed_ready(engine, monkeypatch)
    with Session(engine) as session:
        inputs_candidate = assemble_inputs(session, candidate_id, priority="CANDIDATE")
        inputs_final = assemble_inputs(session, candidate_id, priority="FINAL_CLIP")
        assert input_fingerprint(inputs_candidate) != input_fingerprint(inputs_final)
