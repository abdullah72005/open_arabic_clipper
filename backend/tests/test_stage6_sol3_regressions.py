"""Regressions for the third SOL-review Stage 6 defect set.

1. Complete ownership/currentness protection in persistence.
2. Durable, replay-safe call budgets.
3. Arabic narration fidelity via order/occurrence-aware alignment.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from test_stage6_materialization_core import make_inputs

from app.core.enums import (
    CandidateDisposition,
    ContentType,
    JobKind,
    JobStatus,
    OriginalityRisk,
    RightsRisk,
)
from app.db.base import Base
from app.materialization.inputs import MaterializationInputs
from app.materialization.narration import (
    FakeNarrationSynthesizer,
    transcription_matches,
)
from app.materialization.policy import DEFAULT_CONFIG, CallLedger
from app.materialization.providers import (
    FakeCritiqueProvider,
    FakeWritingProvider,
    materialize_script,
)
from app.materialization.service import (
    MaterializationStaleError,
    persist_materialization,
    run_materialization,
)
from app.models import ClipCandidate, ClipMaterialization, ProcessingJob, SourceVideo

NOVEL_COPY = "The mechanism is that mentorship loss compounds"


def _inputs(**overrides: object) -> MaterializationInputs:
    inputs = make_inputs()
    blocks = list(inputs.blocks)
    blocks[1] = {**blocks[1], "grounding_refs": ["block:0"]}
    base = replace(inputs, blocks=tuple(blocks))
    return replace(base, **overrides)


def _run(inputs: MaterializationInputs, ledger: CallLedger | None = None):
    return run_materialization(
        inputs,
        provider=FakeWritingProvider(copies={1: NOVEL_COPY}),
        critique_provider=FakeCritiqueProvider(accepted={1: True}),
        narration_synthesizer=FakeNarrationSynthesizer(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        ledger=ledger or CallLedger(),
    )


def _seed_candidate(session: Session) -> tuple[SourceVideo, ClipCandidate]:
    source = SourceVideo(source_uri="/tmp/stage6c.mp4", content_hash="stage6c-hash")
    session.add(source)
    session.flush()
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key="stage6c",
        disposition=CandidateDisposition.CANDIDATE,
        start_time=0.0,
        end_time=10.0,
        start_segment_index=0,
        end_segment_index=0,
        primary_content_type=ContentType.STORY,
        rights_risk=RightsRisk.LOW,
        originality_risk=OriginalityRisk.NOT_INDICATED,
    )
    session.add(candidate)
    session.flush()
    return source, candidate


# 1. Complete ownership/currentness protection ------------------------------


def test_changed_input_supersession_cannot_be_reverted_by_historical_token() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        source, candidate = _seed_candidate(session)
        inputs_a = _inputs(source_id=str(source.id), candidate_id=str(candidate.id))
        changed = replace(inputs_a, priority="FINAL_CLIP")
        assert changed.plan_id == inputs_a.plan_id

        row_a = persist_materialization(session, inputs_a, _run(inputs_a), claim_version=1)
        assert row_a.is_current
        row_b = persist_materialization(session, changed, _run(changed), claim_version=2)
        assert row_b.is_current
        assert row_a.id != row_b.id

        # A historical token must never demote the newer current generation.
        with pytest.raises(MaterializationStaleError):
            persist_materialization(session, inputs_a, _run(inputs_a), claim_version=1)
        current = session.scalars(
            select(ClipMaterialization).where(ClipMaterialization.is_current.is_(True))
        ).one()
        assert current.id == row_b.id


def test_equal_claim_with_changed_generation_is_rejected() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        source, candidate = _seed_candidate(session)
        inputs_a = _inputs(source_id=str(source.id), candidate_id=str(candidate.id))
        changed = replace(inputs_a, priority="FINAL_CLIP")
        persist_materialization(session, inputs_a, _run(inputs_a), claim_version=2)
        with pytest.raises(MaterializationStaleError):
            persist_materialization(session, changed, _run(changed), claim_version=2)


def test_worker_mutation_requires_claim_and_running_job() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        source, candidate = _seed_candidate(session)
        inputs = _inputs(source_id=str(source.id), candidate_id=str(candidate.id))
        job = ProcessingJob(
            source_video_id=source.id,
            kind=JobKind.MATERIALIZATION,
            status=JobStatus.QUEUED,
        )
        session.add(job)
        session.flush()

        # A worker mutation without its claim token is rejected.
        with pytest.raises(MaterializationStaleError):
            persist_materialization(session, inputs, _run(inputs), active_job_id=job.id)
        # A claim token whose job is not RUNNING is rejected.
        with pytest.raises(MaterializationStaleError):
            persist_materialization(
                session, inputs, _run(inputs), active_job_id=job.id, claim_version=1
            )
        # A RUNNING job with the exact claim is accepted.
        job.status = JobStatus.RUNNING
        job.claim_version = 1
        session.commit()
        row = persist_materialization(
            session, inputs, _run(inputs), active_job_id=job.id, claim_version=1
        )
        assert row.active_job_id == job.id


# 2. Durable, replay-safe call budgets --------------------------------------


def test_reserve_is_idempotent_and_consumes_budget_once() -> None:
    ledger = CallLedger()
    assert ledger.reserve("scope:write:ROUTINE", kind="semantic", tier="ROUTINE") is True
    assert ledger.reserve("scope:write:ROUTINE", kind="semantic", tier="ROUTINE") is False
    assert ledger.semantic == 1
    assert len(ledger.ledger_entries) == 1


def test_merge_unions_attempts_and_never_blind_adds() -> None:
    first = CallLedger()
    first.reserve("gen1:write:ROUTINE", kind="semantic", tier="ROUTINE")
    first.reserve("gen1:critique", kind="critique")
    second = CallLedger()
    second.reserve("gen1:write:ROUTINE", kind="semantic", tier="ROUTINE")
    second.reserve("gen1:critique", kind="critique")
    # Same generation replay: union by identity, count stays two.
    assert first.merge(second).semantic == 2
    assert len(first.merge(second).ledger_entries) == 2

    other = CallLedger()
    other.reserve("gen2:write:ROUTINE", kind="semantic", tier="ROUTINE")
    other.reserve("gen2:critique", kind="critique")
    merged = first.merge(other)
    # Genuinely distinct attempts union in exactly once each.
    assert merged.semantic == 4
    assert len(merged.ledger_entries) == 4


def test_crashed_attempt_stays_consumed_and_is_not_reinvoked() -> None:
    inputs = _inputs()
    ledger = CallLedger()
    provider = FakeWritingProvider(copies={1: NOVEL_COPY})
    scope = "fixed-scope"
    materialize_script(
        provider,
        inputs,
        ledger=ledger,
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        attempt_scope=scope,
        critique_provider=FakeCritiqueProvider(accepted={1: True}),
    )
    first_calls = len(provider.calls)
    assert first_calls == 1

    # Simulate a re-run after a crash between reservation and response: the same
    # scope must not re-invoke the provider and must stay budget-consumed.
    replay = materialize_script(
        provider,
        inputs,
        ledger=ledger,
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        attempt_scope=scope,
        critique_provider=FakeCritiqueProvider(accepted={1: True}),
    )
    assert len(provider.calls) == first_calls
    assert replay.deferred is True
    assert "ATTEMPT_ALREADY_RESERVED" in replay.reason_codes
    assert ledger.semantic == 2


def test_exhausted_semantic_budget_defers_without_invoking() -> None:
    inputs = _inputs()
    ledger = CallLedger()
    for index in range(DEFAULT_CONFIG.max_raw_semantic_calls):
        ledger.reserve(f"pre:{index}", kind="semantic", tier="ROUTINE")
    provider = FakeWritingProvider(copies={1: NOVEL_COPY})
    script = materialize_script(
        provider,
        inputs,
        ledger=ledger,
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        attempt_scope="scope",
    )
    assert provider.calls == []
    assert "SEMANTIC_BUDGET_EXHAUSTED" in script.reason_codes


def test_synthesis_ceiling_is_enforced_across_generation() -> None:
    inputs = _inputs()
    ledger = CallLedger()
    for index in range(DEFAULT_CONFIG.max_raw_synthesis_calls):
        ledger.reserve(f"pre:synth:{index}", kind="synthesis")
    result = _run(inputs, ledger=ledger)
    assert result.narration == ()
    assert ledger.synthesis == DEFAULT_CONFIG.max_raw_synthesis_calls


def test_persisted_budget_does_not_double_count_on_identical_resubmit() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        source, candidate = _seed_candidate(session)
        inputs = _inputs(source_id=str(source.id), candidate_id=str(candidate.id))
        row = persist_materialization(session, inputs, _run(inputs), claim_version=1)
        first_semantic = row.call_ledger["semantic"]
        assert first_semantic == 2
        row2 = persist_materialization(session, inputs, _run(inputs), claim_version=1)
        assert row2.call_ledger["semantic"] == first_semantic
        assert len(row2.call_ledger["entries"]) == first_semantic


# 3. Arabic narration fidelity ----------------------------------------------


def test_arabic_role_reversal_is_rejected() -> None:
    expected = "أحمد قابل محمد اليوم في المكتب بعد انتهاء الاجتماع"
    actual = "محمد قابل أحمد اليوم في المكتب بعد انتهاء الاجتماع"
    assert not transcription_matches(expected, actual)


def test_changed_number_attribution_is_rejected() -> None:
    expected = "أحمد اشترى 3 كتب ومحمد اشترى 5 أقلام"
    actual = "أحمد اشترى 5 كتب ومحمد اشترى 3 أقلام"
    assert not transcription_matches(expected, actual)


def test_valid_equivalent_arabic_speech_passes() -> None:
    assert transcription_matches(
        "أحمد قابل محمد اليوم في المكتب",
        "احمد قابل محمد اليوم في المكتب",
    )
    assert transcription_matches(
        "أحمد ذهب إلى المكتب",
        "أحمد ذهب امم إلى المكتب",
    )
