"""Deterministic Stage 5.2 publication/ownership race tests (PostgreSQL-gated).

Controlled synchronization (events and committed transactions), never timing
sleeps. Each test is bound to a uniquely named disposable database and cleans it
up afterwards; the supplied URL is only a connection base.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from stage51_support import seed_stage51
from stage52_support import install_fake_stream_probe
from test_stage52_remediation import _executor, _FakeRunner, _plan_ready

from app.core.enums import JobStatus, RenderExecutionLifecycle
from app.db.base import Base
from app.models import ClipCandidate, ProcessingJob
from app.models.render_execution import RenderExecution
from app.render.execution.queue import queue_render_execution
from app.render.execution.types import TechnicalQCResult

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not _URL, reason="CLIPFACTORY_TEST_POSTGRES_URL is required for PostgreSQL race tests"
)

_DATABASE = f"clipfactory_stage52_races_{uuid.uuid4().hex[:12]}"


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    assert _URL is not None
    admin_url = str(make_url(_URL).set(database="postgres").render_as_string(hide_password=False))
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{_DATABASE}" WITH (FORCE)'))
            connection.execute(text(f'CREATE DATABASE "{_DATABASE}"'))
    finally:
        admin.dispose()
    target = str(make_url(_URL).set(database=_DATABASE).render_as_string(hide_password=False))
    engine = create_engine(target)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()
        cleanup = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        try:
            with cleanup.connect() as connection:
                connection.execute(text(f'DROP DATABASE IF EXISTS "{_DATABASE}" WITH (FORCE)'))
        finally:
            cleanup.dispose()


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.render.execution.queue._dispatch", lambda *args: None)
    monkeypatch.setattr("app.composition.queue._dispatch", lambda *args: None)
    install_fake_stream_probe(monkeypatch)


def _seed(engine: Engine, monkeypatch: Any) -> tuple[uuid.UUID, uuid.UUID]:
    with Session(engine) as session:
        fixture = seed_stage51(session, monkeypatch)
        _plan_ready(session, fixture)
        candidate_id = fixture.stage50.selection.candidate.id
        outcome = queue_render_execution(session, candidate_id)
        session.commit()
        assert outcome.job_id is not None
        return outcome.render_execution_id, outcome.job_id


def _pass_qc(**_: Any) -> TechnicalQCResult:
    return TechnicalQCResult(status="PASS", checks=(), reason_codes=(), policy_version="test")


def _run(engine: Engine, row_id: uuid.UUID, job_id: uuid.UUID, qc: Any) -> Any:
    session = Session(engine)
    try:
        executor = _executor(session, _FakeRunner(), qc_checker=qc)
        executor.set_active_job(job_id)
        try:
            executor.execute(row_id)
        except BaseException:  # noqa: BLE001 - outcome asserted via persisted state
            pass
        return executor
    finally:
        session.close()


def test_final_publication_blocks_on_concurrent_candidate_invalidation(
    engine: Engine, monkeypatch: Any
) -> None:
    row_id, job_id = _seed(engine, monkeypatch)

    def invalidating_qc(*args: Any, **kwargs: Any) -> TechnicalQCResult:
        other = Session(engine)
        try:
            candidate = other.get(ClipCandidate, _candidate_id(engine, row_id))
            assert candidate is not None
            candidate.is_current = False
            other.commit()
        finally:
            other.close()
        return _pass_qc()

    _run(engine, row_id, job_id, invalidating_qc)
    with Session(engine) as check:
        row = check.get(RenderExecution, row_id)
        job = check.get(ProcessingJob, job_id)
        assert row is not None and row.lifecycle is not RenderExecutionLifecycle.COMPLETE
        assert row.cache_eligible is False
        assert job is not None and job.status is not JobStatus.SUCCEEDED


def test_final_publication_blocks_on_concurrent_cancellation(
    engine: Engine, monkeypatch: Any
) -> None:
    row_id, job_id = _seed(engine, monkeypatch)

    def cancelling_qc(*args: Any, **kwargs: Any) -> TechnicalQCResult:
        other = Session(engine)
        try:
            job = other.get(ProcessingJob, job_id)
            assert job is not None
            job.status = JobStatus.CANCELLED
            other.commit()
        finally:
            other.close()
        return _pass_qc()

    _run(engine, row_id, job_id, cancelling_qc)
    with Session(engine) as check:
        row = check.get(RenderExecution, row_id)
        job = check.get(ProcessingJob, job_id)
        assert row is not None and row.lifecycle is RenderExecutionLifecycle.CANCELLED
        assert row.active_job_id is None
        assert job is not None and job.status is JobStatus.CANCELLED


def test_final_publication_blocks_on_superseded_claim(engine: Engine, monkeypatch: Any) -> None:
    row_id, job_id = _seed(engine, monkeypatch)

    def superseding_qc(*args: Any, **kwargs: Any) -> TechnicalQCResult:
        other = Session(engine)
        try:
            job = other.get(ProcessingJob, job_id)
            assert job is not None
            job.status = JobStatus.RUNNING
            job.claim_version = int(job.claim_version) + 5
            other.commit()
        finally:
            other.close()
        return _pass_qc()

    executor = _run(engine, row_id, job_id, superseding_qc)
    assert executor.skipped_duplicate is True
    with Session(engine) as check:
        row = check.get(RenderExecution, row_id)
        assert row is not None and row.lifecycle is not RenderExecutionLifecycle.COMPLETE


def test_concurrent_invalidation_serializes_with_publication_lock(
    engine: Engine, monkeypatch: Any
) -> None:
    row_id, job_id = _seed(engine, monkeypatch)
    candidate_id = _candidate_id(engine, row_id)

    reached = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    errors: list[BaseException] = []

    def gated_qc(*args: Any, **kwargs: Any) -> TechnicalQCResult:
        reached.set()
        assert release.wait(timeout=30)
        return _pass_qc()

    def worker() -> None:
        try:
            _run(engine, row_id, job_id, gated_qc)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
        finally:
            completed.set()

    invalidator = Session(engine)
    thread = threading.Thread(target=worker, daemon=True)
    try:
        # Hold the candidate row lock, then invalidate and commit under it.
        invalidator.execute(
            select(ClipCandidate).where(ClipCandidate.id == candidate_id).with_for_update()
        ).first()
        thread.start()
        assert reached.wait(timeout=30)
        candidate = invalidator.get(ClipCandidate, candidate_id)
        assert candidate is not None
        candidate.is_current = False
        invalidator.commit()
        release.set()
        assert completed.wait(timeout=60)
    finally:
        invalidator.close()
    assert not errors, errors
    with Session(engine) as check:
        row = check.get(RenderExecution, row_id)
        job = check.get(ProcessingJob, job_id)
        assert row is not None and row.lifecycle is not RenderExecutionLifecycle.COMPLETE
        assert job is not None and job.status is not JobStatus.SUCCEEDED


def _candidate_id(engine: Engine, row_id: uuid.UUID) -> uuid.UUID:
    with Session(engine) as session:
        row = session.get(RenderExecution, row_id)
        assert row is not None
        return row.clip_candidate_id
