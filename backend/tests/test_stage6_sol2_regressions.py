"""Regressions for the second SOL-review Stage 6 defect set."""

from __future__ import annotations

from dataclasses import replace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from test_stage6_materialization_core import make_inputs

from app.core.enums import (
    CandidateDisposition,
    ContentType,
    MaterializationOutcome,
    OriginalityRisk,
    PlanStatus,
    RefinementPriority,
    RefinementStatus,
    RightsRisk,
    StrategyDisposition,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.db.base import Base
from app.materialization import service
from app.materialization.acceptance import AssessmentContext, assess_materialization
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
from app.models import CandidateRefinement, ClipCandidate, ClipMaterialization, SourceVideo
from app.models.transformation_eligibility import (
    TransformationEligibilityAnalysis,
    TransformationStrategyCandidate,
)
from app.models.transformation_plan import TransformationPlan, TransformationPlanSet

NOVEL_COPY = "The mechanism is that mentorship loss compounds"


def _inputs(**overrides: object) -> MaterializationInputs:
    inputs = make_inputs()
    blocks = list(inputs.blocks)
    blocks[1] = {**blocks[1], "grounding_refs": ["block:0"]}
    base = replace(inputs, blocks=tuple(blocks))
    return replace(base, **overrides)


# 3. Hard failures are never clearable by a favorable critique --------------


def test_invalid_grounding_reference_is_never_cleared_by_critique() -> None:
    inputs = _inputs()
    blocks = list(inputs.blocks)
    # A non-essential SOURCE_AS_EVIDENCE block with an invalid reference.
    blocks[1] = {
        **blocks[1],
        "block_type": "ORIGINAL_VALUE",
        "substantive_value_kind": "SOURCE_AS_EVIDENCE",
        "grounding_refs": ["block:999"],
    }
    inputs = replace(inputs, blocks=tuple(blocks))
    script = materialize_script(
        FakeWritingProvider(copies={1: NOVEL_COPY}),
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        critique_provider=FakeCritiqueProvider(accepted={1: True}),
    )
    assert script.authored_blocks == ()
    assert any("INVALID_GROUNDING_REFERENCE" in code for code in script.reason_codes)


# 4. Final readiness requires complete, in-policy evidence ------------------


def test_qc_pass_with_missing_measurement_defers() -> None:
    assessment = assess_materialization(
        AssessmentContext(render_qc_status="PASS", measured_loudness_lufs=-16.0),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.DEFERRED_EVIDENCE.value


def test_out_of_policy_loudness_is_assessed_as_unmet() -> None:
    assessment = assess_materialization(
        AssessmentContext(
            render_qc_status="PASS",
            measured_loudness_lufs=-9.0,
            measured_true_peak_dbtp=-2.0,
        ),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.REJECTED_QUALITY.value
    assert "LOUDNESS_OUT_OF_POLICY" in assessment.reason_codes


def test_true_peak_over_ceiling_is_assessed_as_unmet() -> None:
    assessment = assess_materialization(
        AssessmentContext(
            render_qc_status="PASS",
            measured_loudness_lufs=-16.0,
            measured_true_peak_dbtp=0.5,
        ),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.REJECTED_QUALITY.value
    assert "TRUE_PEAK_EXCEEDED" in assessment.reason_codes


def test_unknown_critical_fidelity_defers() -> None:
    assessment = assess_materialization(
        AssessmentContext(
            render_qc_status="PASS",
            measured_loudness_lufs=-16.0,
            measured_true_peak_dbtp=-2.0,
            critical_fidelity_unknown=True,
        ),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.DEFERRED_EVIDENCE.value


# 5. Generated narration fulfills the contract narration slot ---------------


def test_generated_narration_fulfills_required_contract_slot() -> None:
    slots = (
        {
            "slot_id": "narration-1",
            "slot_kind": "AUTHORED_NARRATION",
            "block_index": 1,
            "block_type": "ORIGINAL_VALUE",
            "required": True,
            "reason_code": "REQUIRED_NARRATION",
            "payload": {},
        },
    )
    inputs = _inputs(
        execution_slots=slots,
        narration={"need": "RECOMMENDED", "requirements": {"placement_block_index": 1}},
        profile=replace(make_inputs().profile, narration_allowed=True),
    )
    result = run_materialization(
        inputs,
        provider=FakeWritingProvider(copies={1: NOVEL_COPY}),
        critique_provider=FakeCritiqueProvider(accepted={1: True}),
        narration_synthesizer=FakeNarrationSynthesizer(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        ledger=CallLedger(),
    )
    assert result.narration
    slot = next(item for item in result.slot_resolutions if item.slot_kind == "AUTHORED_NARRATION")
    assert slot.resolution == "FULFILLED"


# 6. ASR/script: contracted negation and reversed attribution ---------------


def test_asr_rejects_contracted_negation() -> None:
    assert not transcription_matches("John didn't agree", "John did agree")


def test_asr_rejects_reversed_speaker_attribution() -> None:
    assert not transcription_matches("John told Mary", "Mary told John")


def test_asr_accepts_equivalent_order() -> None:
    assert transcription_matches("John did not agree", "john did not agree")


# 1. Frozen FINAL_CLIP readiness -------------------------------------------


def test_usable_final_rejects_non_ready_status() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for index, status in enumerate(
            (
                RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW,
                RefinementStatus.CANDIDATE_REFINED,
            )
        ):
            source = SourceVideo(source_uri=f"/tmp/x{index}.mp4", content_hash=f"h{index}")
            session.add(source)
            session.flush()
            candidate = ClipCandidate(
                source_video_id=source.id,
                candidate_key=f"c{index}",
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
            session.add(
                CandidateRefinement(
                    source_video_id=source.id,
                    clip_candidate_id=candidate.id,
                    priority=RefinementPriority.FINAL_CLIP,
                    status=status,
                    coarse_start=0.0,
                    coarse_end=10.0,
                    context_start=0.0,
                    context_end=10.0,
                    refined_start=0.0,
                    refined_end=10.0,
                    final_transcript="some text",
                    word_timestamps=[],
                )
            )
            session.flush()
            assert service._usable_final(session, candidate.id) is None


def test_plan_target_context_is_read_from_the_plan_set() -> None:
    """assemble_inputs must use the authoritative plan-set context, not the plan row."""

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        source = SourceVideo(source_uri="/tmp/ctx.mp4", content_hash="ctx-hash")
        session.add(source)
        session.flush()
        candidate = ClipCandidate(
            source_video_id=source.id,
            candidate_key="ctx",
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
        analysis = TransformationEligibilityAnalysis(
            source_video_id=source.id, clip_candidate_id=candidate.id
        )
        session.add(analysis)
        session.flush()
        strategy = TransformationStrategyCandidate(
            analysis_id=analysis.id,
            strategy_key="k",
            strategy_type=TransformationStrategyType.SOURCE_AS_EVIDENCE,
            disposition=StrategyDisposition.RECOMMENDED,
            intensity=TransformationIntensity.MINIMAL,
            rank=1,
        )
        session.add(strategy)
        session.flush()
        plan_set = TransformationPlanSet(
            source_video_id=source.id,
            clip_candidate_id=candidate.id,
            transformation_analysis_id=analysis.id,
            target_context={
                "target_market": "EGYPT",
                "output_language_policy": "SOURCE_LANGUAGE",
                "register_intent": "SOURCE_COMPATIBLE",
            },
        )
        session.add(plan_set)
        session.flush()
        plan = TransformationPlan(
            plan_set_id=plan_set.id,
            strategy_candidate_id=strategy.id,
            plan_key="pk",
            status=PlanStatus.PLAN_GENERATED,
            strategy_type=TransformationStrategyType.SOURCE_AS_EVIDENCE,
            intensity=TransformationIntensity.MINIMAL,
            blocks=[],
        )
        session.add(plan)
        session.commit()

        loaded = session.get(TransformationPlan, plan.id)
        assert loaded is not None
        resolved_set = session.get(TransformationPlanSet, loaded.plan_set_id)
        assert resolved_set is not None
        assert service._plan_target_context(resolved_set)["target_market"] == "EGYPT"
        # The plan row has no target_context attribute; the plan set is the only
        # authoritative source.
        assert not hasattr(loaded, "target_context")


# 7. Persistence preserves budgets and fences stale workers -----------------


def _seed_candidate(session: Session) -> tuple[SourceVideo, ClipCandidate]:
    source = SourceVideo(source_uri="/tmp/stage6b.mp4", content_hash="stage6b-hash")
    session.add(source)
    session.flush()
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key="stage6b",
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


def _run(inputs: MaterializationInputs):
    return run_materialization(
        inputs,
        provider=FakeWritingProvider(copies={1: NOVEL_COPY}),
        critique_provider=FakeCritiqueProvider(accepted={1: True}),
        narration_synthesizer=FakeNarrationSynthesizer(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        ledger=CallLedger(),
    )


def test_persistence_preserves_repairs_and_budget_across_retries() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        source, candidate = _seed_candidate(session)
        inputs = _inputs(source_id=str(source.id), candidate_id=str(candidate.id))
        row = persist_materialization(session, inputs, _run(inputs), claim_version=1)
        row.repair_count = 1
        session.commit()
        # Identical work re-submitted must not reset the repair allowance.
        row2 = persist_materialization(session, inputs, _run(inputs), claim_version=1)
        assert row2.id == row.id
        assert row2.repair_count == 1
        assert row2.call_ledger["semantic"] >= 2


def test_persistence_rejects_stale_worker() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        source, candidate = _seed_candidate(session)
        inputs = _inputs(source_id=str(source.id), candidate_id=str(candidate.id))
        persist_materialization(session, inputs, _run(inputs), claim_version=2)
        with pytest.raises(MaterializationStaleError):
            persist_materialization(session, inputs, _run(inputs), claim_version=1)
        # The authoritative current row is unchanged.
        current = session.scalars(
            select(ClipMaterialization).where(ClipMaterialization.is_current.is_(True))
        ).one()
        assert current.claim_version == 2
