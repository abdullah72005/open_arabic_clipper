"""Stage 6 runtime corrections: application-session semantics, reactivation,
deferred recovery, and upstream revalidation.

These regressions reproduce the PostgreSQL probes: with the production session
settings (``autoflush=False``/``expire_on_commit=False``) the old code loaded a
cached QUEUED job, lost JSON checkpoint updates, replayed deferred responses
forever, and wrote output after upstream went stale.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from test_stage6_component_recovery import (
    AcceptingCritiqueProvider,
    ScriptedWritingProvider,
    _ready_composition,
    accepting_critique_result,
)
from test_stage6_materialization_core import make_inputs

from app.core.enums import JobKind, JobStatus, MaterializationOutcome
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization import queue as queue_module
from app.materialization.components import ComponentProviders, prepare_materialization
from app.materialization.executor import build_materialization_executor
from app.materialization.policy import DEFAULT_CONFIG
from app.materialization.providers import WritingRequest, WritingResult
from app.materialization.queue import queue_materialization
from app.materialization.repository import (
    claim_materialization,
    get_or_create_envelope,
    load_component,
    merge_component_checkpoint,
    reserve_attempt,
)
from app.models import ClipCandidate, ClipMaterialization, ProcessingJob
from app.services.storage import StorageService


def _owned_envelope(engine: Engine) -> tuple[Session, Any, Any, Any]:
    session = Session(engine)
    inputs = replace(make_inputs(), candidate_id=str(uuid.uuid4()), source_id=str(uuid.uuid4()))
    envelope = get_or_create_envelope(session, inputs, {"priority": "CANDIDATE"})
    job = ProcessingJob(
        source_video_id=uuid.UUID(inputs.source_id),
        kind=JobKind.MATERIALIZATION,
        materialization_id=envelope.id,
        status=JobStatus.RUNNING,
        claim_version=1,
    )
    session.add(job)
    session.flush()
    envelope.active_job_id = job.id
    session.commit()
    owner = claim_materialization(session, envelope.id, job.id, expected_job_claim_version=1)
    assert owner is not None
    session.commit()
    return session, inputs, owner, envelope


# 1. Application-session semantics ------------------------------------------


def test_sequential_component_checkpoints_preserve_both(monkeypatch: Any, tmp_path: Any) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'seq.sqlite3'}")
    Base.metadata.create_all(engine)
    session, _inputs, owner, _envelope = _owned_envelope(engine)
    try:
        # Hold a stale ORM reference while writing two components sequentially.
        stale = session.get(ClipMaterialization, owner.materialization_id)
        assert stale is not None
        assert merge_component_checkpoint(
            session, owner, "SUPPORTING", {"dependency_fingerprint": "s", "status": "ACCEPTED"}
        )
        assert merge_component_checkpoint(
            session, owner, "TIMELINE", {"dependency_fingerprint": "t", "status": "ACCEPTED"}
        )
        fresh = session.get(ClipMaterialization, owner.materialization_id, populate_existing=True)
        assert fresh is not None
        assert "SUPPORTING" in fresh.component_checkpoints
        assert "TIMELINE" in fresh.component_checkpoints
    finally:
        session.close()
        engine.dispose()


def test_sequential_attempts_are_all_persisted(monkeypatch: Any, tmp_path: Any) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'att.sqlite3'}")
    Base.metadata.create_all(engine)
    session, _inputs, owner, _envelope = _owned_envelope(engine)
    try:
        stale = session.get(ClipMaterialization, owner.materialization_id)
        assert stale is not None
        first = reserve_attempt(session, owner, "SCRIPT.WRITE", "d", "semantic", "ROUTINE")
        second = reserve_attempt(session, owner, "SCRIPT.CRITIQUE", "d", "critique", None)
        fresh = session.get(ClipMaterialization, owner.materialization_id, populate_existing=True)
        assert fresh is not None
        ids = {entry["attempt_id"] for entry in fresh.call_ledger["entries"]}
        assert {first["attempt_id"], second["attempt_id"]} <= ids
    finally:
        session.close()
        engine.dispose()


def test_production_session_wrapper_runs_the_job(monkeypatch: Any, tmp_path: Any) -> None:
    from stage51_support import seed_stage51

    from app.workers import tasks

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'prod.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(tasks, "create_session_factory", lambda: factory)
    monkeypatch.setattr(queue_module, "_dispatch", lambda *a, **k: None)

    session = factory()
    fixture = seed_stage51(session, monkeypatch, narration="NONE")
    candidate = fixture.stage50.selection.candidate
    _ready_composition(session, fixture)
    outcome = queue_materialization(session, candidate.id, priority="CANDIDATE")
    session.commit()
    assert outcome.job_id is not None
    job_id = str(outcome.job_id)
    candidate_id = str(candidate.id)
    session.close()

    tasks.run_materialization_job.apply(
        args=[candidate_id, job_id, "CANDIDATE", None, "OFF", False]
    ).get()

    verify = factory()
    try:
        job = verify.get(ProcessingJob, uuid.UUID(job_id))
        assert job is not None
        assert job.status is JobStatus.SUCCEEDED
        envelope = verify.get(ClipMaterialization, job.materialization_id)
        assert envelope is not None and envelope.is_current is True
    finally:
        verify.close()
        engine.dispose()


# 3. Deferred is not accepted ------------------------------------------------


class ToggleWritingProvider(ScriptedWritingProvider):
    provider_name = "toggle"

    def __init__(self) -> None:
        super().__init__()
        self.deferred = True

    def write(self, request: WritingRequest) -> WritingResult:
        self.calls.append(request)
        if self.deferred:
            return WritingResult(
                copies={},
                provider_identity=self.runtime_identity(),
                deferred=True,
                reason_code="DEFERRED_PROVIDER",
            )
        return WritingResult(copies=dict(self._copies), provider_identity=self.runtime_identity())


class ToggleCritiqueProvider(AcceptingCritiqueProvider):
    provider_name = "toggle-critique"

    def __init__(self) -> None:
        super().__init__()
        self.deferred = True

    def critique(self, request: Any) -> Any:
        from app.materialization.providers import CritiqueResult

        self.calls.append(request)
        if self.deferred:
            return CritiqueResult(
                accepted={},
                provider_identity=self.runtime_identity(),
                deferred=True,
                reason_code="DEFERRED_PROVIDER",
            )
        return accepting_critique_result(request, self.runtime_identity())


def test_deferred_script_is_not_replayed_and_recovers(tmp_path: Any) -> None:
    from app.materialization.components import script_dependency_fingerprint

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'defer.sqlite3'}")
    Base.metadata.create_all(engine)
    session, inputs, owner, _envelope = _owned_envelope(engine)
    try:
        writing = ToggleWritingProvider()
        critique = ToggleCritiqueProvider()
        context = _context(session, owner, inputs, writing, critique)
        dep = script_dependency_fingerprint(context, critique_provider=critique)

        # Run 1: deferred writing is not a reusable accepted component.
        prepare_materialization(context)
        assert len(writing.calls) == 1
        assert load_component(session, owner.materialization_id, "SCRIPT", dep) is None
        stored = session.get(ClipMaterialization, owner.materialization_id, populate_existing=True)
        assert stored is not None
        assert stored.component_checkpoints["SCRIPT"]["status"] == "DEFERRED"
        assert "writing" not in stored.component_checkpoints["SCRIPT"]

        # Run 2: writing recovers and is stored; critique is deferred.
        writing.deferred = False
        prepare_materialization(_context(session, owner, inputs, writing, critique))
        assert len(writing.calls) == 2
        assert len(critique.calls) == 1
        assert load_component(session, owner.materialization_id, "SCRIPT", dep) is None

        # Run 3: the accepted writing is reused; healthy critique accepts.
        critique.deferred = False
        prepare_materialization(_context(session, owner, inputs, writing, critique))
        assert len(writing.calls) == 2
        assert len(critique.calls) == 2
        accepted = load_component(session, owner.materialization_id, "SCRIPT", dep)
        assert accepted is not None and accepted["status"] == "ACCEPTED"
    finally:
        session.close()
        engine.dispose()


def _context(session: Session, owner: Any, inputs: Any, writing: Any, critique: Any) -> Any:
    from app.materialization.components import ComponentContext

    return ComponentContext(
        session=session,
        owner=owner,
        inputs=inputs,
        config=DEFAULT_CONFIG,
        storage=StorageService(get_settings().storage_root),
        providers=ComponentProviders(writing=writing, critique=critique),
        allow_hosted=True,
    )


# 4. Upstream revalidation ---------------------------------------------------


class MutatingWritingProvider(ScriptedWritingProvider):
    provider_name = "mutating"

    def __init__(self, engine: Engine, candidate_id: uuid.UUID) -> None:
        super().__init__()
        self._engine = engine
        self._candidate_id = candidate_id

    def write(self, request: WritingRequest) -> WritingResult:
        result = super().write(request)
        with Session(self._engine) as other:
            candidate = other.get(ClipCandidate, self._candidate_id)
            assert candidate is not None
            candidate.is_current = False
            other.commit()
        return result


def test_upstream_staleness_during_writing_blocks_output(monkeypatch: Any, tmp_path: Any) -> None:
    from stage51_support import seed_stage51

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'stale.sqlite3'}")
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        fixture = seed_stage51(session, monkeypatch, narration="NONE")
        candidate = fixture.stage50.selection.candidate
        _ready_composition(session, fixture)
        job = ProcessingJob(source_video_id=candidate.source_video_id, kind=JobKind.MATERIALIZATION)
        session.add(job)
        session.commit()

        writing = MutatingWritingProvider(engine, candidate.id)
        critique = AcceptingCritiqueProvider()
        executor = build_materialization_executor(
            session,
            fixture.stage50.storage,
            get_settings(),
            writing=writing,
            critique=critique,
        )
        executor.set_active_job(job.id)
        executor.execute(str(candidate.id))
        session.expire_all()

        job = session.get(ProcessingJob, job.id)
        assert job is not None and job.status is JobStatus.SUCCEEDED
        envelope = session.scalars(
            select(ClipMaterialization).where(ClipMaterialization.clip_candidate_id == candidate.id)
        ).first()
        assert envelope is not None
        assert envelope.output_fingerprint == ""
        assert envelope.semantic_outcome is MaterializationOutcome.UPSTREAM_REVALIDATION_REQUIRED
        assert envelope.lifecycle.value == "BLOCKED"
    finally:
        session.close()
        engine.dispose()
