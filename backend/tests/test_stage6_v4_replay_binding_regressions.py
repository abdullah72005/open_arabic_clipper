"""Stage 6 V4 replay-binding regressions (F1/F2, offline contract only).

These tests drive the real ``CheckpointedCritiqueProvider`` over a durable
SQLite envelope. They prove that a critique can only be replayed for the exact
bounded request facts it reviewed, and that an old judgment is never relabelled
with new delivery/timing facts. Provider responses are local stubs: simulated
QC, not live semantic or human acceptance. No TTS, hosted, Qwen or worker path
is invoked.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from test_stage6_gemini_boundary import NOVEL_COPY, _inputs_with_draft
from test_stage6_v4_followup_regressions import _complete_block

from app.core.enums import JobKind, JobStatus
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.acceptance import AssessmentContext, assess_materialization
from app.materialization.components import (
    CheckpointedCritiqueProvider,
    ComponentContext,
    ComponentProviders,
    DurableRequestGateway,
    script_dependency_fingerprint,
)
from app.materialization.fingerprints import input_fingerprint
from app.materialization.policy import DEFAULT_CONFIG, CallLedger
from app.materialization.providers import (
    CRITIQUE_ACCEPTED,
    CRITIQUE_DEFERRED,
    CRITIQUE_REJECTED,
    BlockCritique,
    CritiqueDimensionFinding,
    CritiqueRequest,
    CritiqueResult,
    DeterministicWritingProvider,
    materialize_script,
)
from app.materialization.repository import (
    MaterializationOwner,
    claim_materialization,
    get_or_create_envelope,
    merge_component_checkpoint,
)
from app.materialization.service import assemble_result
from app.models import ClipMaterialization, ProcessingJob
from app.services.storage import StorageService


class RecordingCritique:
    """Local stub producing complete categorical evidence; records raw calls."""

    provider_name = "recorded"

    def __init__(self) -> None:
        self.calls: list[CritiqueRequest] = []
        self.result: CritiqueResult | None = None

    def available(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "recorded", "model": "m", "api_version": "v1"}

    def critique(self, request: CritiqueRequest) -> CritiqueResult:
        self.calls.append(request)
        blocks: dict[int, BlockCritique] = {}
        for target in request.targets:
            refs = target.offered_grounding_refs or ("block:0",)
            blocks[target.block_index] = _complete_block(
                index=target.block_index, copy=target.copy, refs=refs
            )
        self.result = CritiqueResult(blocks=blocks, provider_identity=self.runtime_identity())
        return self.result

    def release(self) -> None:
        return None


def _enabled_inputs() -> Any:
    base = _inputs_with_draft(NOVEL_COPY)
    return replace(
        base,
        candidate_id=str(uuid.uuid4()),
        source_id=str(uuid.uuid4()),
        narration={"need": "RECOMMENDED", "requirements": {}},
        profile=replace(base.profile, narration_allowed=True),
    )


@pytest.fixture  # type: ignore[untyped-decorator]
def replay_envelope(
    tmp_path: Path,
) -> Iterator[tuple[Session, MaterializationOwner, Any]]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'replay.sqlite3'}")
    Base.metadata.create_all(engine)
    session = Session(engine)
    inputs = _enabled_inputs()
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
    try:
        yield session, owner, inputs
    finally:
        session.close()
        engine.dispose()


def _component_context(
    session: Session, owner: MaterializationOwner, inputs: Any, critique: RecordingCritique
) -> ComponentContext:
    return ComponentContext(
        session=session,
        owner=owner,
        inputs=inputs,
        config=DEFAULT_CONFIG,
        storage=StorageService(get_settings().storage_root),
        providers=ComponentProviders(writing=DeterministicWritingProvider(), critique=critique),
        allow_hosted=True,
    )


def _run_script(
    session: Session,
    owner: MaterializationOwner,
    inputs: Any,
    critique: RecordingCritique,
    *,
    narration_available: bool,
) -> tuple[Any, CheckpointedCritiqueProvider, str]:
    context = _component_context(session, owner, inputs, critique)
    dependency = script_dependency_fingerprint(context, critique_provider=critique)
    wrapper = CheckpointedCritiqueProvider(critique, DurableRequestGateway(context), dependency)
    script = materialize_script(
        DeterministicWritingProvider(),
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        attempt_scope=input_fingerprint(inputs),
        critique_provider=wrapper,
        narration_available=narration_available,
    )
    return script, wrapper, dependency


def _script_critique_entry(session: Session, owner: MaterializationOwner) -> dict[str, Any]:
    row = session.get(ClipMaterialization, owner.materialization_id)
    assert row is not None
    entry = (row.component_checkpoints or {}).get("SCRIPT")
    assert isinstance(entry, Mapping)
    critique = entry.get("critique")
    assert isinstance(critique, Mapping)
    return dict(critique)


def _length_pass_findings(script: Any) -> list[Any]:
    return [
        finding
        for finding in script.findings
        if "CRITIQUE_LENGTH_NECESSITY_PASS" in finding.reason_codes
    ]


# --- F1: legacy boolean verdict agrees with the decision --------------------


def test_explicit_negative_boolean_rejects() -> None:
    result = CritiqueResult(accepted={1: False})
    assert result.decision(1) is False
    assert result.verdict(1) == CRITIQUE_REJECTED


def test_positive_boolean_never_approves() -> None:
    result = CritiqueResult(accepted={1: True})
    assert result.decision(1) is None
    assert result.verdict(1) == CRITIQUE_DEFERRED


def test_missing_boolean_is_unknown() -> None:
    result = CritiqueResult()
    assert result.decision(1) is None
    assert result.verdict(1) is None


def test_structured_result_verdict_precedes_legacy_boolean() -> None:
    accepting = BlockCritique(
        block_index=1,
        findings=tuple(
            CritiqueDimensionFinding(dimension, "PASS")
            for dimension in (
                "GROUNDING_ENTAILMENT",
                "SUBSTANTIVE_ADDED_VALUE",
                "INTENT_FULFILLMENT",
                "SEMANTIC_FIDELITY",
                "UNSUPPORTED_FACTS",
                "CODE_SWITCH_PRESERVATION",
                "REDUNDANCY",
                "DIALECT_REGISTER_FIT",
                "NATURALNESS",
                "LENGTH_NECESSITY",
            )
        ),
    )
    accepting = replace(
        accepting,
        findings=tuple(
            replace(
                finding,
                copy_quote=NOVEL_COPY,
                grounding_refs=("block:0",),
                added_relation="The copy names the mechanism.",
            )
            if finding.dimension == "INTENT_FULFILLMENT"
            else finding
            for finding in accepting.findings
        ),
    )
    result = CritiqueResult(accepted={1: False}, blocks={1: accepting})
    assert result.decision(1) is True
    assert result.verdict(1) == CRITIQUE_ACCEPTED

    rejecting = replace(
        accepting,
        findings=tuple(
            replace(finding, level="FAIL")
            if finding.dimension == "GROUNDING_ENTAILMENT"
            else finding
            for finding in accepting.findings
        ),
    )
    rejected = CritiqueResult(accepted={1: True}, blocks={1: rejecting})
    assert rejected.decision(1) is False
    assert rejected.verdict(1) == CRITIQUE_REJECTED


# --- F2: replay is bound to the exact reviewed request facts ---------------


def test_silent_then_narration_available_replay_defers_without_resend(
    replay_envelope: tuple[Session, MaterializationOwner, Any],
) -> None:
    session, owner, inputs = replay_envelope
    critique = RecordingCritique()

    first, first_wrapper, _dep = _run_script(
        session, owner, inputs, critique, narration_available=False
    )
    assert first.accepted is True
    assert first_wrapper.reused is False
    assert len(critique.calls) == 1
    reviewed = critique.calls[0].targets[0].proposed_delivery
    assert reviewed["delivery"] == "STANDALONE_CARD"
    assert reviewed["planned_duration_seconds"] == pytest.approx(3.357143, abs=1e-6)
    entry = _script_critique_entry(session, owner)
    assert entry["request_fingerprint"]
    assert entry["reviewed_targets"][0]["proposed_delivery"]["delivery"] == "STANDALONE_CARD"

    second, second_wrapper, _dep = _run_script(
        session, owner, inputs, critique, narration_available=True
    )
    assert second_wrapper.reused is False
    assert second.deferred is True
    assert second.authored_blocks == ()
    # No automatic provider resend for an incompatible replay.
    assert len(critique.calls) == 1
    assert "CRITIQUE_CHECKPOINT_REQUEST_MISMATCH" in second.reason_codes
    assert _length_pass_findings(second) == []


def test_narration_available_then_silent_replay_defers_without_resend(
    replay_envelope: tuple[Session, MaterializationOwner, Any],
) -> None:
    session, owner, inputs = replay_envelope
    critique = RecordingCritique()

    first, _wrapper, _dep = _run_script(session, owner, inputs, critique, narration_available=True)
    assert first.accepted is True
    assert critique.calls[0].targets[0].proposed_delivery["delivery"] == "NARRATION"
    assert len(critique.calls) == 1

    second, second_wrapper, _dep = _run_script(
        session, owner, inputs, critique, narration_available=False
    )
    assert second_wrapper.reused is False
    assert second.deferred is True
    assert second.authored_blocks == ()
    assert len(critique.calls) == 1
    assert "CRITIQUE_CHECKPOINT_REQUEST_MISMATCH" in second.reason_codes
    assert _length_pass_findings(second) == []


def test_exact_match_replay_reuses_checkpoint_with_zero_provider_calls(
    replay_envelope: tuple[Session, MaterializationOwner, Any],
) -> None:
    session, owner, inputs = replay_envelope
    critique = RecordingCritique()

    first, _wrapper, _dep = _run_script(session, owner, inputs, critique, narration_available=False)
    assert first.accepted is True
    first_calls = len(critique.calls)

    second, second_wrapper, _dep = _run_script(
        session, owner, inputs, critique, narration_available=False
    )
    assert second_wrapper.reused is True
    assert second.accepted is True
    assert len(critique.calls) == first_calls
    passes = _length_pass_findings(second)
    assert len(passes) == 1
    evidence = passes[0].evidence
    assert evidence["reviewed_delivery"] == "STANDALONE_CARD"
    assert evidence["reviewed_duration_seconds"] == pytest.approx(3.357143, abs=1e-6)
    assert evidence["copy"] == NOVEL_COPY


@pytest.mark.parametrize("bound", [True, False])
def test_legacy_or_malformed_positive_checkpoint_defers_without_resend(
    replay_envelope: tuple[Session, MaterializationOwner, Any], bound: bool
) -> None:
    session, owner, inputs = replay_envelope
    critique = RecordingCritique()
    _first, _wrapper, dependency = _run_script(
        session, owner, inputs, critique, narration_available=False
    )
    entry = _script_critique_entry(session, owner)
    calls_after_recording = len(critique.calls)
    entry["response"] = {
        "accepted": [{"index": 1, "value": True}],
        "blocks": [],
        "provider_identity": {"provider": "legacy"},
        "notes": "",
        "deferred": False,
        "reason_code": None,
    }
    if not bound:
        entry.pop("request_fingerprint", None)
        entry.pop("reviewed_targets", None)
    assert entry["dependency_fingerprint"] == dependency
    merge_component_checkpoint(session, owner, "SCRIPT", {"critique": entry})

    script, wrapper, _dep = _run_script(session, owner, inputs, critique, narration_available=False)
    assert wrapper.reused is False
    assert script.deferred is True
    assert script.authored_blocks == ()
    assert len(critique.calls) == calls_after_recording
    expected = "MALFORMED_CRITIQUE_EVIDENCE" if bound else "CRITIQUE_CHECKPOINT_UNBOUND"
    assert expected in script.reason_codes


def test_legacy_complete_but_unbound_checkpoint_cannot_authorize(
    replay_envelope: tuple[Session, MaterializationOwner, Any],
) -> None:
    session, owner, inputs = replay_envelope
    critique = RecordingCritique()
    _first, _wrapper, _dep = _run_script(
        session, owner, inputs, critique, narration_available=False
    )
    entry = _script_critique_entry(session, owner)
    calls_after_recording = len(critique.calls)
    # A complete old positive response without the request binding is legacy.
    entry.pop("request_fingerprint", None)
    merge_component_checkpoint(session, owner, "SCRIPT", {"critique": entry})

    script, wrapper, _dep = _run_script(session, owner, inputs, critique, narration_available=False)
    assert wrapper.reused is False
    assert script.deferred is True
    assert script.authored_blocks == ()
    assert len(critique.calls) == calls_after_recording
    assert "CRITIQUE_CHECKPOINT_UNBOUND" in script.reason_codes


def test_stale_silent_card_judgment_does_not_admit_the_render(
    replay_envelope: tuple[Session, MaterializationOwner, Any],
) -> None:
    session, owner, inputs = replay_envelope
    critique = RecordingCritique()
    _first, _wrapper, _dep = _run_script(
        session, owner, inputs, critique, narration_available=False
    )
    stale, _wrapper, _dep = _run_script(session, owner, inputs, critique, narration_available=True)
    assert stale.deferred is True

    assembled = assemble_result(
        inputs,
        script=stale,
        narration=(),
        supporting=(),
        config=DEFAULT_CONFIG,
        ledger=CallLedger(),
        tts_available=False,
        narration_required=True,
    )
    context = dataclasses.replace(
        AssessmentContext.from_dict(assembled.assessment_context),
        render_complete=True,
        render_qc_status="PASS",
        measured_loudness_lufs=-16.0,
        measured_true_peak_dbtp=-2.0,
    )
    final = assess_materialization(
        context, output_duration=assembled.timeline.output_duration, config=DEFAULT_CONFIG
    )
    assert final.outcome not in {"READY_FOR_HUMAN_REVIEW", "MATERIALIZED_WITH_CAUTION"}
    assert final.publication_ready is False
    assert len(critique.calls) == 1
