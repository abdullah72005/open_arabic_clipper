"""End-to-end offline proof that suggestion classification is kind-independent.

Drives the actual production contracts for the target SOURCE_AS_EVIDENCE
candidate: Stage 4.1 planning -> Stage 4.2 governance -> Stage 4.3 selection ->
Stage 5.0 render contract -> Stage 6 writing/critique. All provider seams are
fakes; no worker, database server, ffprobe, ffmpeg, TTS, Qwen, or Gemini runs.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage41_support import (
    FakePlanningProvider,
    build_words,
    seed_stage41,
)
from stage42_support import FakeGovernanceProvider, FakeGovernanceSettings, install_stage42_settings
from stage50_support import FakeProber, managed_source

from app.core.enums import (
    GovernancePlanStatus,
    NarrationNeed,
    PlanBlockType,
    RefinementPriority,
    RefinementStatus,
    SemanticProviderMode,
    SourceExcerptRole,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.inputs import (
    MaterializationInputs,
    ProfileContext,
    VisualPlanEvidence,
)
from app.materialization.policy import DEFAULT_CONFIG as STAGE6_CONFIG
from app.materialization.policy import CallLedger
from app.materialization.providers import (
    FakeCritiqueProvider,
    FakeWritingProvider,
    build_writing_request,
    materialize_script,
)
from app.materialization.scripting import validate_authored_block
from app.materialization.service import (
    _contract_slots,
    _final_clip_evidence,
    _merge_blocks,
    _source_media_evidence,
)
from app.media.ffprobe import MediaMetadata
from app.models import CandidateRefinement, Transcript
from app.render.service import create_render_contract
from app.transformation.executor import build_transformation_executor
from app.transformation.governance.executor import build_transformation_governance_executor
from app.transformation.governance.queue import get_or_create_governance_set
from app.transformation.planning.executor import TransformationPlanningExecutor
from app.transformation.planning.policy import DEFAULT_CONFIG as STAGE41_CONFIG
from app.transformation.planning.queue import get_or_create_plan_set, list_plans
from app.transformation.planning.request import OPERATOR_REQUEST_VERSION
from app.transformation.planning.types import (
    NarrationRequirement,
    PlanProviderBlock,
    PlanProviderPlan,
)
from app.transformation.selection.service import read_selection, select_transformation_plan

CANDIDATE_ID = "4b7e08bb-67f1-42b6-8433-958bc57d6bce"
WINDOW_START = 259.160
WINDOW_END = 271.000
SOURCE_EXCERPT = (
    "إن شاء الله خير لو أنت عادة مع صحابتك وفيه واحد جديد داخل عليك وإيه أكتر حاجة "
    "ممكن تقفلك منه؟ الاستظراف أه أو يخش مثلا يفرد نفسه أو يكون تانك أو يعود يتكلم "
    "عن نفسه كتير"
)
SUGGESTION = "اقتراح للحوار: بعد ما تحكي موقفك، اسأل اللي قدامك عن تجربة شبهه وسيب له مساحة يرد."
SEMANTIC_INTENT = (
    "Offer an authored, source-linked conversational suggestion: after describing "
    "your own situation, invite the other person to share a similar experience and "
    "leave room for their answer."
)
WHY_UNAVAILABLE = (
    "The source names self-focused behaviour but does not offer a concrete conversational action."
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _prober() -> FakeProber:
    return FakeProber(
        MediaMetadata(
            duration_seconds=360.0,
            video_codec="h264",
            width=1920,
            height=1080,
            frames_per_second=30.0,
            audio_codec="aac",
            audio_sample_rate=48_000,
        )
    )


def _materialization_inputs(
    session: Session,
    candidate: Any,
    plan: Any,
    contract: Any,
    final: CandidateRefinement,
) -> MaterializationInputs:
    merged_blocks, bindings = _merge_blocks(contract, plan)
    selection = read_selection(session, candidate.id)
    assert selection is not None and selection.effective
    profile_payload = dict(plan.plan_set.target_context or {})
    return MaterializationInputs(
        candidate_id=str(candidate.id),
        candidate_key=candidate.candidate_key,
        source_id=str(candidate.source_video_id),
        priority="CANDIDATE",
        selection_id=str(selection.row.id),
        selection_output_fingerprint=selection.row.output_fingerprint,
        selected_plan_fingerprint=plan.plan_output_fingerprint,
        plan_id=str(plan.id),
        plan_output_fingerprint=plan.plan_output_fingerprint,
        strategy_type=plan.strategy_type.value,
        intensity=plan.intensity.value,
        blocks=merged_blocks,
        hero_block_index=int(plan.hero_block_index),
        narration={
            "need": plan.narration_need,
            "requirements": dict(plan.narration_requirements or {}),
        },
        verification_dependencies=(),
        original_value_kinds=tuple(plan.original_value_kinds or []),
        required_context=tuple(plan.required_context or []),
        preservation_constraints=tuple(plan.preservation_constraints or []),
        plan_target_context=profile_payload,
        contract_id=str(contract.id),
        contract_status=contract.status.value,
        contract_input_fingerprint=contract.input_fingerprint,
        contract_output_fingerprint=contract.output_fingerprint,
        execution_slots=_contract_slots(contract),
        contract_source_bindings=bindings,
        final_clip=_final_clip_evidence(final),
        visual_plan=VisualPlanEvidence(
            visual_plan_id="offline-visual",
            output_fingerprint="offline-visual-output",
            plan_ready=True,
            ass_relative_path="offline/captions.ass",
            ass_sha256="a" * 64,
            ass_policy_version="offline",
            caption_source_fingerprint=contract.caption_source_fingerprint,
            framing_fingerprint="offline-framing",
            source_media_fingerprint=contract.source_media_fingerprint,
        ),
        source_media=_source_media_evidence(contract),
        profile=ProfileContext(
            profile_id=None,
            profile_key="default",
            target_market=str(profile_payload.get("target_market", "UNSPECIFIED")),
            output_language_policy=str(
                profile_payload.get("output_language_policy", "SOURCE_LANGUAGE")
            ),
            register_intent=str(profile_payload.get("register_intent", "SOURCE_COMPATIBLE")),
            narration_allowed=False,
        ),
        source_dialect={"profile": "EGYPTIAN", "confidence": 0.85},
        source_language="ar",
    )


def test_source_as_evidence_authored_thesis_suggestion_passes_every_boundary(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    words = build_words(SOURCE_EXCERPT, start=WINDOW_START, span=WINDOW_END - WINDOW_START)
    governance_provider = FakeGovernanceProvider(auto=True)
    settings = FakeGovernanceSettings(
        planning_mode="adaptive",
        governance_mode="adaptive",
        governance_provider=governance_provider,
    )
    install_stage42_settings(monkeypatch, settings)

    seed = seed_stage41(
        session,
        candidate_id=CANDIDATE_ID,
        settings=settings,
        strategy_type=TransformationStrategyType.SOURCE_AS_EVIDENCE,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
        transcript=SOURCE_EXCERPT,
        transcript_duration=360.0,
        refined=(WINDOW_START, WINDOW_END),
        candidate_start=WINDOW_START,
        candidate_end=WINDOW_END,
        source_segments=[
            {"start": 0.0, "end": WINDOW_START, "text": "Intro"},
            {
                "start": WINDOW_START,
                "end": WINDOW_END,
                "text": SOURCE_EXCERPT,
                "corrected_text": SOURCE_EXCERPT,
            },
            {"start": WINDOW_END, "end": 360.0, "text": "Outro"},
        ],
        words=words,
    )
    source, candidate, planning_refinement, analysis, strategies = seed
    assert candidate.id == uuid.UUID(CANDIDATE_ID)
    assert candidate.start_time == WINDOW_START
    assert candidate.end_time == WINDOW_END
    assert planning_refinement.coarse_start == WINDOW_START
    assert planning_refinement.coarse_end == WINDOW_END
    transcript_row = session.query(Transcript).filter_by(source_video_id=source.id).one()
    target_segment = transcript_row.segments[1]
    assert (target_segment["start"], target_segment["end"]) == (WINDOW_START, WINDOW_END)
    assert target_segment["text"] == SOURCE_EXCERPT
    assert strategies[0].strategy_type.value == "SOURCE_AS_EVIDENCE"

    # Plan against the exact FINAL_CLIP refinement the production path uses.
    final = CandidateRefinement(
        source_video_id=source.id,
        clip_candidate_id=candidate.id,
        priority=RefinementPriority.FINAL_CLIP,
        status=RefinementStatus.FINAL_TRANSCRIPT_READY,
        coarse_start=planning_refinement.coarse_start,
        coarse_end=planning_refinement.coarse_end,
        context_start=planning_refinement.context_start,
        context_end=planning_refinement.context_end,
        refined_start=WINDOW_START,
        refined_end=WINDOW_END,
        automatic_transcript=SOURCE_EXCERPT,
        final_transcript=SOURCE_EXCERPT,
        word_timestamps=words,
        confidence=0.95,
        quality_level="FINAL_CLIP",
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.85,
        output_fingerprint="integrated-final-fp",
    )
    session.add(final)
    session.flush()
    analysis.refinement_id = final.id
    analysis.refinement_priority = RefinementPriority.FINAL_CLIP.value
    analysis.refinement_quality_level = "FINAL_CLIP"
    session.flush()
    analysis.input_fingerprint = build_transformation_executor(session, settings).input_fingerprint(
        candidate
    )
    session.commit()

    strategy = strategies[0]
    provider_plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        confidence=0.7,
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                purpose="Hero source moment",
                source_role=SourceExcerptRole.HERO,
                use_full_window=False,
                word_start_index=0,
                word_end_index=len(words) - 1,
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.ORIGINAL_VALUE,
                purpose="Authored conversational suggestion",
                estimated_duration=4.0,
                substantive_value_kind=SubstantiveValueKind.AUTHORED_THESIS,
                semantic_intent=SEMANTIC_INTENT,
                why_unavailable=WHY_UNAVAILABLE,
                grounding_refs=("block:0",),
                draft_line=SUGGESTION,
            ),
        ),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )

    plan_set = get_or_create_plan_set(session, candidate, analysis)
    planning_provider = FakePlanningProvider([provider_plan])
    settings = FakeGovernanceSettings(
        planning_provider=planning_provider,
        governance_provider=governance_provider,
        planning_mode="adaptive",
        governance_mode="adaptive",
    )
    install_stage42_settings(monkeypatch, settings)
    planning = TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=planning_provider,
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=SemanticProviderMode.ADAPTIVE,
        config=STAGE41_CONFIG,
        lease_factory=settings.heavy_model_lease_factory(),
    )
    plan_set.operator_request = {
        "version": OPERATOR_REQUEST_VERSION,
        "source_start": WINDOW_START,
        "source_end": WINDOW_END,
        "editorial_intent": None,
    }
    session.flush()
    planning.execute(plan_set.id)
    session.refresh(plan_set)

    plans = list_plans(session, plan_set.id)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.blocks[1]["substantive_value_kind"] == "AUTHORED_THESIS"
    assert plan.blocks[1]["draft_line"] == SUGGESTION
    assert provider_plan.blocks[0].use_full_window is False
    assert plan.blocks[0]["word_start_index"] == 0
    assert plan.blocks[0]["word_end_index"] == len(words) - 1
    assert plan.blocks[0]["source_start"] == WINDOW_START
    assert plan.blocks[0]["source_end"] == WINDOW_END

    governance_set = get_or_create_governance_set(session, candidate, plan_set)
    build_transformation_governance_executor(session, settings).execute(governance_set.id)
    session.refresh(governance_set)
    governance_rows = [
        row for row in governance_set.results if row.transformation_plan_id == plan.id
    ]
    assert len(governance_rows) == 1
    assert governance_rows[0].status in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }
    assert governance_rows[0].eligible_for_stage4_3 is True

    selection = select_transformation_plan(session, candidate.id)
    assert selection is not None and selection.effective
    assert selection.row.selected_plan_id == plan.id
    assert selection.row.status.value in {
        "PLAN_SELECTED",
        "PLAN_SELECTED_WITH_CAUTION",
    }

    storage, _source_path = managed_source(source)
    contract_view = create_render_contract(
        session,
        candidate.id,
        settings=get_settings(),
        storage=storage,
        prober=_prober(),
    )
    assert contract_view is not None
    contract = contract_view.row
    assert contract.contract_ready is True
    assert contract.status.value == "MATERIALIZATION_REQUIRED"

    inputs = _materialization_inputs(session, candidate, plan, contract, final)
    request = build_writing_request(inputs, config=STAGE6_CONFIG)
    target = next(item for item in request.targets if item.block_index == 1)
    assert target.substantive_value_kind == "AUTHORED_THESIS"
    assert target.claim_class == "AUTHORED_SOURCE_LINKED_SUGGESTION"

    # The label is not acceptance: this exact copy requires the complete
    # semantic critique before it can be accepted.
    validation = validate_authored_block(
        {**inputs.blocks[1], "copy": SUGGESTION},
        inputs,
        STAGE6_CONFIG,
    )
    assert validation.claim is not None
    assert validation.claim.claim_class == "AUTHORED_SOURCE_LINKED_SUGGESTION"
    assert validation.requires_semantic_review is True
    assert validation.accepted is False

    critique_provider = FakeCritiqueProvider(accepted={1: True})
    script = materialize_script(
        FakeWritingProvider(copies={1: SUGGESTION}),
        inputs,
        ledger=CallLedger(),
        config=STAGE6_CONFIG,
        allow_hosted=True,
        critique_provider=critique_provider,
    )
    assert len(critique_provider.calls) == 1
    critique_target = critique_provider.calls[0].targets[0]
    assert critique_target.block_index == 1
    assert critique_target.claim_class == "AUTHORED_SOURCE_LINKED_SUGGESTION"
    assert script.accepted
    assert len(script.authored_blocks) == 1
    assert script.authored_blocks[0].substantive_value_kind == "AUTHORED_THESIS"
    assert script.authored_blocks[0].claim_class == "AUTHORED_SOURCE_LINKED_SUGGESTION"
    assert governance_provider.calls == 1
