"""Stage 6 durable component gateway: reservation, checkpoint recovery, replay.

These regressions drive the real ``prepare_materialization`` component path with
an owned envelope on a shared file-backed SQLite database so an independent
provider session can observe the committed reservation. They prove that a crash
cannot erase budget history, that accepted subcomponents resume without repeating
providers, that cancellation and exhausted budgets stop before the next call, and
that the real executor resume path reuses accepted script work.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session
from test_stage6_materialization_core import make_inputs

from app.core.enums import JobKind, JobStatus
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.components import (
    ComponentContext,
    ComponentProviders,
    prepare_materialization,
    script_dependency_fingerprint,
)
from app.materialization.fingerprints import stable_digest
from app.materialization.policy import DEFAULT_CONFIG
from app.materialization.providers import (
    CritiqueRequest,
    CritiqueResult,
    WritingRequest,
    WritingResult,
)
from app.materialization.repository import (
    MaterializationOwner,
    claim_materialization,
    get_or_create_envelope,
    ledger_counts,
    load_component,
)
from app.models import ClipMaterialization, ProcessingJob
from app.services.storage import StorageService

NOVEL_COPY = "The mechanism is that mentorship loss compounds"


def _scripted_copies() -> dict[int, str]:
    return {1: NOVEL_COPY}


class ScriptedWritingProvider:
    """Hosted-looking provider returning fixed copies and recording invocations."""

    provider_name = "scripted"

    def __init__(self, copies: Mapping[int, str] | None = None) -> None:
        self._copies = dict(copies or _scripted_copies())
        self.calls: list[WritingRequest] = []

    def available(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "scripted", "model": "m", "api_version": "v1"}

    def write(self, request: WritingRequest) -> WritingResult:
        self.calls.append(request)
        return WritingResult(copies=dict(self._copies), provider_identity=self.runtime_identity())

    def release(self) -> None:
        return None


class InspectingWritingProvider(ScriptedWritingProvider):
    """Asserts a committed SCRIPT.WRITE reservation exists before it is invoked."""

    provider_name = "inspecting"

    def __init__(self, engine: Engine, envelope_id: uuid.UUID) -> None:
        super().__init__()
        self._engine = engine
        self._envelope_id = envelope_id
        self.saw_reserved = False
        self.raise_after = True

    def write(self, request: WritingRequest) -> WritingResult:
        self.calls.append(request)
        with Session(self._engine) as observer:
            row = observer.get(ClipMaterialization, self._envelope_id)
            assert row is not None
            entries = (row.call_ledger or {}).get("entries", [])
            reserved = [
                entry
                for entry in entries
                if isinstance(entry, Mapping)
                and entry.get("component") == "SCRIPT.WRITE"
                and entry.get("status") == "RESERVED"
            ]
            assert reserved, "the reservation must be committed before provider work"
            self.saw_reserved = True
        if self.raise_after:
            raise RuntimeError("simulated crash after provider invocation")
        return WritingResult(copies=dict(self._copies), provider_identity=self.runtime_identity())


class AcceptingCritiqueProvider:
    provider_name = "accepting-critique"

    def __init__(self, *, fail_first: bool = False) -> None:
        self.calls: list[CritiqueRequest] = []
        self._fail_first = fail_first

    def available(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "accepting-critique", "model": "m", "api_version": "v1"}

    def critique(self, request: CritiqueRequest) -> CritiqueResult:
        self.calls.append(request)
        if self._fail_first and len(self.calls) == 1:
            raise RuntimeError("simulated crash during critique")
        return CritiqueResult(accepted={1: True}, provider_identity=self.runtime_identity())

    def release(self) -> None:
        return None


class CountingNarrationSynthesizer:
    provider_name = "counting-tts"

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def available(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "counting-tts", "model": "m", "api_version": "v1"}

    def synthesize(self, request: Any):
        from app.materialization.narration import FakeNarrationSynthesizer

        self.calls.append(request)
        return FakeNarrationSynthesizer().synthesize(request)

    def release(self) -> None:
        return None


@pytest.fixture  # type: ignore[untyped-decorator]
def component_fixture(
    tmp_path: Path,
) -> Iterator[tuple[Engine, Session, Any, MaterializationOwner]]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'components.sqlite3'}")
    Base.metadata.create_all(engine)
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
    owner = claim_materialization(
        session, envelope.id, job.id, expected_job_claim_version=1
    )
    assert owner is not None
    session.commit()
    try:
        yield engine, session, inputs, owner
    finally:
        session.close()
        engine.dispose()


def _context(
    session: Session,
    owner: MaterializationOwner,
    inputs: Any,
    providers: ComponentProviders,
    *,
    allow_hosted: bool = True,
    cancelled=None,
) -> ComponentContext:
    return ComponentContext(
        session=session,
        owner=owner,
        inputs=inputs,
        config=DEFAULT_CONFIG,
        storage=StorageService(get_settings().storage_root),
        providers=providers,
        allow_hosted=allow_hosted,
        cancelled=cancelled or (lambda: False),
    )


def test_reserved_attempt_is_committed_before_provider_and_stays_uncertain(
    component_fixture: Any,
) -> None:
    engine, session, inputs, owner = component_fixture
    provider = InspectingWritingProvider(engine, owner.materialization_id)
    context = _context(
        session,
        owner,
        inputs,
        ComponentProviders(writing=provider, critique=AcceptingCritiqueProvider()),
    )
    with pytest.raises(RuntimeError):
        prepare_materialization(context)
    session.rollback()
    assert provider.saw_reserved is True
    assert len(provider.calls) == 1

    # A fresh session observes the durable UNCERTAIN attempt and consumed budget.
    with Session(engine) as observer:
        row = observer.get(ClipMaterialization, owner.materialization_id)
        assert row is not None
        counts = ledger_counts(row.call_ledger)
        assert counts["semantic"] == 1
        assert counts["uncertain"] == 1


def test_accepted_writing_before_critique_crash_resumes_writing(
    component_fixture: Any,
) -> None:
    engine, session, inputs, owner = component_fixture
    writing = ScriptedWritingProvider()
    critique = AcceptingCritiqueProvider(fail_first=True)
    context = _context(
        session,
        owner,
        inputs,
        ComponentProviders(writing=writing, critique=critique),
    )
    with pytest.raises(RuntimeError):
        prepare_materialization(context)
    session.rollback()
    assert len(writing.calls) == 1
    assert len(critique.calls) == 1

    resume_context = _context(
        session,
        owner,
        inputs,
        ComponentProviders(writing=writing, critique=critique),
    )
    prepare_materialization(resume_context)
    # The completed writing response is reused; only critique re-executes.
    assert len(writing.calls) == 1
    assert len(critique.calls) == 2


def test_accepted_script_replay_uses_load_component_and_resume(
    component_fixture: Any,
) -> None:
    engine, session, inputs, owner = component_fixture
    writing = ScriptedWritingProvider()
    critique = AcceptingCritiqueProvider()
    context = _context(
        session, owner, inputs, ComponentProviders(writing=writing, critique=critique)
    )
    prepared = prepare_materialization(context)
    assert prepared.authored_blocks
    accepted_blocks = prepared.authored_blocks

    script_dep = script_dependency_fingerprint(context, critique_provider=critique)
    before = len(writing.calls) + len(critique.calls)
    recovered = load_component(session, owner.materialization_id, "SCRIPT", script_dep)
    assert recovered is not None
    assert recovered["status"] == "ACCEPTED"
    assert recovered["result"]["authored_blocks"] == accepted_blocks
    assert len(writing.calls) + len(critique.calls) == before

    # Re-running the whole component path reuses every accepted subcomponent.
    prepare_materialization(
        _context(session, owner, inputs, ComponentProviders(writing=writing, critique=critique))
    )
    assert len(writing.calls) == 1
    assert len(critique.calls) == 1


def test_cancellation_between_writing_and_critique_stops_next_call(
    component_fixture: Any,
) -> None:
    engine, session, inputs, owner = component_fixture
    state = {"cancelled": False}

    class FlippingProvider(ScriptedWritingProvider):
        def write(self, request: WritingRequest) -> WritingResult:
            result = super().write(request)
            state["cancelled"] = True
            return result

    writing = FlippingProvider()
    critique = AcceptingCritiqueProvider()
    context = _context(
        session,
        owner,
        inputs,
        ComponentProviders(writing=writing, critique=critique),
        cancelled=lambda: state["cancelled"],
    )
    prepare_materialization(context)
    assert len(writing.calls) == 1
    assert len(critique.calls) == 0


def test_exhausted_semantic_budget_defers_without_invoking(
    component_fixture: Any,
) -> None:
    engine, session, inputs, owner = component_fixture
    row = session.get(ClipMaterialization, owner.materialization_id)
    assert row is not None
    row.call_ledger = {
        "version": "stage6-attempts-v2",
        "entries": [
            {
                "attempt_id": str(uuid.uuid4()),
                "component": "SCRIPT.WRITE",
                "dependency_fingerprint": "x",
                "phase": "INITIAL",
                "ordinal": index,
                "kind": "semantic",
                "tier": "ROUTINE",
                "status": "RESPONDED",
                "provider_identity": {},
                "response_fingerprint": None,
            }
            for index in range(1, DEFAULT_CONFIG.max_raw_semantic_calls + 1)
        ],
    }
    session.commit()

    writing = ScriptedWritingProvider()
    critique = AcceptingCritiqueProvider()
    prepared = prepare_materialization(
        _context(session, owner, inputs, ComponentProviders(writing=writing, critique=critique))
    )
    assert writing.calls == []
    assert critique.calls == []
    assert prepared.authored_blocks == []


def test_accepted_synthesis_is_reused_on_resume(component_fixture: Any) -> None:
    engine, session, inputs, owner = component_fixture
    profile = replace(inputs.profile, narration_allowed=True)
    narration_inputs = replace(
        inputs,
        narration={"need": "RECOMMENDED", "requirements": {}},
        profile=profile,
    )
    # Rebind the envelope identity for the narration-enabled generation.
    envelope = get_or_create_envelope(session, narration_inputs, {"priority": "CANDIDATE"})
    job = ProcessingJob(
        source_video_id=uuid.UUID(narration_inputs.source_id),
        kind=JobKind.MATERIALIZATION,
        materialization_id=envelope.id,
        status=JobStatus.RUNNING,
        claim_version=1,
    )
    session.add(job)
    session.flush()
    envelope.active_job_id = job.id
    session.commit()
    owner = claim_materialization(
        session, envelope.id, job.id, expected_job_claim_version=1
    )
    assert owner is not None

    writing = ScriptedWritingProvider()
    critique = AcceptingCritiqueProvider()
    synthesizer = CountingNarrationSynthesizer()
    context = _context(
        session,
        owner,
        narration_inputs,
        ComponentProviders(writing=writing, critique=critique, narration=synthesizer),
    )
    first = prepare_materialization(context)
    assert first.narration_references
    first_synthesis_calls = len(synthesizer.calls)
    assert first_synthesis_calls >= 1

    prepare_materialization(
        _context(
            session,
            owner,
            narration_inputs,
            ComponentProviders(writing=writing, critique=critique, narration=synthesizer),
        )
    )
    assert len(synthesizer.calls) == first_synthesis_calls


def test_narration_fidelity_is_separately_checkpointed_and_reused(
    component_fixture: Any,
) -> None:
    """Local fidelity validation is a separate, idempotently-reused checkpoint."""

    engine, session, inputs, owner = component_fixture
    profile = replace(inputs.profile, narration_allowed=True)
    narration_inputs = replace(
        inputs,
        narration={"need": "RECOMMENDED", "requirements": {}},
        profile=profile,
    )
    envelope = get_or_create_envelope(session, narration_inputs, {"priority": "CANDIDATE"})
    job = ProcessingJob(
        source_video_id=uuid.UUID(narration_inputs.source_id),
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

    writing = ScriptedWritingProvider()
    critique = AcceptingCritiqueProvider()
    synthesizer = CountingNarrationSynthesizer()
    validated = {"n": 0}

    def accept(_audio: Any) -> Mapping[str, object]:
        validated["n"] += 1
        return {"status": "ACCEPTED", "findings": [], "asr_identity": {"model": "fake"}}

    context = _context(
        session,
        owner,
        narration_inputs,
        ComponentProviders(
            writing=writing,
            critique=critique,
            narration=synthesizer,
            narration_fidelity=accept,
        ),
    )
    first = prepare_materialization(context)
    assert first.narration_references
    synthesis_calls = len(synthesizer.calls)
    assert synthesis_calls >= 1
    assert validated["n"] == 1

    # Replaying reuses both the accepted synthesis and the accepted validation.
    prepare_materialization(
        _context(
            session,
            owner,
            narration_inputs,
            ComponentProviders(
                writing=writing,
                critique=critique,
                narration=synthesizer,
                narration_fidelity=accept,
            ),
        )
    )
    assert len(synthesizer.calls) == synthesis_calls
    assert validated["n"] == 1


def test_rejected_fidelity_does_not_accept_the_narration(
    component_fixture: Any,
) -> None:
    engine, session, inputs, owner = component_fixture
    profile = replace(inputs.profile, narration_allowed=True)
    narration_inputs = replace(
        inputs,
        narration={"need": "RECOMMENDED", "requirements": {}},
        profile=profile,
    )
    envelope = get_or_create_envelope(session, narration_inputs, {"priority": "CANDIDATE"})
    job = ProcessingJob(
        source_video_id=uuid.UUID(narration_inputs.source_id),
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

    def reject(_audio: Any) -> Mapping[str, object]:
        return {"status": "REJECTED", "findings": ["MEANING_MISMATCH"]}

    prepared = prepare_materialization(
        _context(
            session,
            owner,
            narration_inputs,
            ComponentProviders(
                writing=ScriptedWritingProvider(),
                critique=AcceptingCritiqueProvider(),
                narration=CountingNarrationSynthesizer(),
                narration_fidelity=reject,
            ),
        )
    )
    assert not prepared.narration_references


def test_component_dependency_fingerprint_is_stable(component_fixture: Any) -> None:
    _engine, session, inputs, owner = component_fixture
    context = _context(
        session,
        owner,
        inputs,
        ComponentProviders(writing=ScriptedWritingProvider(), critique=AcceptingCritiqueProvider()),
    )
    first = script_dependency_fingerprint(context, critique_provider=context.providers.critique)
    second = script_dependency_fingerprint(context, critique_provider=context.providers.critique)
    assert first == second
    assert first == stable_digest(
        {
            "component": "SCRIPT",
            "validation_version": DEFAULT_CONFIG.validation_version,
            "plan_output_fingerprint": inputs.plan_output_fingerprint,
            "final_clip_output_fingerprint": inputs.final_clip.output_fingerprint,
            "blocks": [dict(block) for block in inputs.blocks],
            "plan_target_context": dict(inputs.plan_target_context),
            "profile_semantic_context": {
                "target_market": inputs.profile.target_market,
                "output_language_policy": inputs.profile.output_language_policy,
                "register_intent": inputs.profile.register_intent,
            },
            "narration": dict(inputs.narration),
            "operator_authored_copy": [dict(item) for item in inputs.operator_authored_copy],
            "verification_dependencies": [dict(item) for item in inputs.verification_dependencies],
            "writing_identity": context.providers.writing.runtime_identity(),
            "critique_identity": context.providers.critique.runtime_identity(),
        }
    )


def _ready_composition(session: Session, fixture: Any) -> None:
    from stage51_support import FakeDetector, FakeFrameSampler, FakeSceneCutDetector

    from app.composition.executor import build_visual_composition_executor
    from app.composition.queue import get_or_create_plan_row

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


def test_executor_resume_reuses_accepted_script(monkeypatch: Any) -> None:
    from stage51_support import seed_stage51

    from app.materialization.executor import build_materialization_executor

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        fixture = seed_stage51(session, monkeypatch, narration="NONE")
        candidate = fixture.stage50.selection.candidate
        _ready_composition(session, fixture)

        writing = ScriptedWritingProvider()
        critique = AcceptingCritiqueProvider()

        job = ProcessingJob(source_video_id=candidate.source_video_id, kind=JobKind.MATERIALIZATION)
        session.add(job)
        session.commit()
        executor = build_materialization_executor(
            session,
            fixture.stage50.storage,
            get_settings(),
            writing=writing,
            critique=critique,
        )
        executor.set_active_job(job.id)
        executor.execute(str(candidate.id))
        session.refresh(job)
        assert job.status is JobStatus.SUCCEEDED
        assert len(writing.calls) >= 1
        first_writing = len(writing.calls)
        first_critique = len(critique.calls) if critique.calls else 0
        envelope = session.scalars(
            select(ClipMaterialization).where(ClipMaterialization.is_current.is_(True))
        ).one()

        # Resume: a new job for the SAME generation reuses accepted script work.
        resume_job = ProcessingJob(
            source_video_id=candidate.source_video_id,
            kind=JobKind.MATERIALIZATION,
        )
        session.add(resume_job)
        session.commit()
        resume = build_materialization_executor(
            session,
            fixture.stage50.storage,
            get_settings(),
            writing=writing,
            critique=critique,
        )
        resume.set_active_job(resume_job.id)
        resume.execute(str(candidate.id))
        session.refresh(resume_job)
        assert resume_job.status is JobStatus.SUCCEEDED
        assert resume_job.materialization_id == envelope.id
        assert len(writing.calls) == first_writing
        assert (len(critique.calls) if critique.calls else 0) == first_critique
