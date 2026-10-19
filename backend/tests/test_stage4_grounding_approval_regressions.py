"""Stage 4.1/4.2/4.3 authoritative grounding and approval regressions."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage42_support import (
    make_critique,
    make_inputs,
    make_plan,
    original_block,
    source_block,
)
from stage43_support import (
    FakeGovernanceSettings,
    install_selection_settings,
    make_result_spec,
    seed_selection_fixture,
)

from app.core.enums import (
    ContentType,
    NarrationNeed,
    PlanBlockType,
    SourceExcerptRole,
    SourceMomentStructure,
    SubstantiveValueKind,
    TransformationSelectionStatus,
)
from app.db.base import Base
from app.transformation.governance.policy import DEFAULT_CONFIG as STAGE42_CONFIG
from app.transformation.governance.validation import (
    SEMANTIC_AVAILABLE,
    apply_semantic_review,
    build_plan_governance,
    evaluate_plan,
    revalidate_integrity,
)
from app.transformation.grounding import (
    GROUNDING_DECLARATION_MALFORMED,
    GROUNDING_EVIDENCE_UNRESOLVED,
    GROUNDING_REQUIRED_MISSING,
    resolve_reference,
)
from app.transformation.planning.policy import DEFAULT_CONFIG as STAGE41_CONFIG
from app.transformation.planning.providers import build_planning_request, parse_plan_results
from app.transformation.planning.types import (
    NarrationRequirement,
    PlanningContext,
    PlanningInputs,
    PlanProviderBlock,
    PlanProviderPlan,
    WordEvidence,
)
from app.transformation.planning.validation import (
    deterministic_provider_plan,
    validate_provider_plan,
)
from app.transformation.selection.service import (
    _mandatory_failures,
    evaluate_without_persisting,
)


def _words() -> tuple[WordEvidence, ...]:
    return tuple(
        WordEvidence(index=index, text=f"w{index}", start=100.0 + index, end=101.0 + index)
        for index in range(10)
    )


def _inputs() -> PlanningInputs:
    words = _words()
    return PlanningInputs(
        candidate_id="candidate",
        candidate_key="candidate-key",
        source_id="source",
        content_type=ContentType.STORY,
        source_moment_structure=SourceMomentStructure.CLAIM,
        source_moment={"hook_index": None, "payoff_index": None},
        transcript=" ".join(word.text for word in words),
        transcript_confidence=0.9,
        refined_start=100.0,
        refined_end=110.0,
        words=words,
        word_coverage_sufficient=True,
        context_segments=(),
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
        code_switch={},
        entities=(),
        unresolved_spans=(),
        idea_summary="idea",
        topic_summary="topic",
        hooks=(),
        rights_risk="LOW",
        originality_risk="NOT_INDICATED",
        rights_status="OWNED",
        media_origin="PODCAST_INTERVIEW",
        provenance_snapshot={},
        stage3_risk={"moment_density_score": 0.1},
        stage40_assessments={},
        stage40_platform_risk={},
        stage40_analysis_id="analysis",
        stage40_input_fingerprint="in",
        stage40_output_fingerprint="out",
        stage40_policy_version="stage4.0-v1",
        stage40_strategies=(),
        planning_context=PlanningContext(),
        refinement_priority="CANDIDATE",
        refinement_quality_level="CANDIDATE",
        refinement_status="CANDIDATE_REFINED",
        refinement_output_fingerprint="ref-out",
        mandatory_evidence={},
    )


def _strategy(
    *, external: str = "NOT_REQUIRED", strategy_type: str = "ANALYSIS"
) -> dict[str, object]:
    return {
        "id": "strategy-1",
        "strategy_key": "candidate:ANALYSIS",
        "strategy_type": strategy_type,
        "intensity": "MODERATE",
        "rank": 1,
        "strategy_fingerprint": "strategy-fp",
        "external_verification_requirement": external,
        "confidence": 0.8,
        "added_value_focus": "Explain the negotiation underneath the dish swap",
        "substantive_value_kind": "SYNTHESIS",
    }


def _value_block(
    *,
    grounding: object = ("block:0",),
    dependency_ids: tuple[str, ...] = (),
    block_type: PlanBlockType = PlanBlockType.ORIGINAL_VALUE,
) -> PlanProviderBlock:
    return PlanProviderBlock(
        block_type=block_type,
        estimated_duration=4.0,
        substantive_value_kind=SubstantiveValueKind.SYNTHESIS,
        semantic_intent="Explain the negotiation underneath the dish swap.",
        why_unavailable="The raw excerpt states the swap but not the negotiation dynamic.",
        grounding_refs=grounding if isinstance(grounding, tuple) else (),
        dependency_ids=dependency_ids,
    )


def _plan(
    *,
    grounding: object = ("block:0",),
    value_block: PlanProviderBlock | None = None,
    extra: tuple[PlanProviderBlock, ...] = (),
) -> PlanProviderPlan:
    return PlanProviderPlan(
        strategy_id="strategy-1",
        strategy_key="candidate:ANALYSIS",
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                source_role=SourceExcerptRole.HERO,
                word_start_index=2,
                word_end_index=6,
            ),
            value_block if value_block is not None else _value_block(grounding=grounding),
            *extra,
        ),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )


def _validate(plan: PlanProviderPlan, *, strategy: dict[str, object] | None = None) -> Any:
    return validate_provider_plan(
        plan,
        strategy or _strategy(),
        _inputs(),
        STAGE41_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="provider-fp",
    )


@pytest.mark.parametrize("grounding", [None, ()])
def test_missing_or_empty_required_grounding_rejected(grounding: object) -> None:
    result = _validate(_plan(grounding=grounding))
    assert result.plan is None
    assert any(reason.startswith(GROUNDING_REQUIRED_MISSING) for reason in result.reasons)


def test_blank_required_grounding_rejected_as_malformed() -> None:
    result = _validate(_plan(grounding=("",)))
    assert result.plan is None
    assert any(reason.startswith(GROUNDING_DECLARATION_MALFORMED) for reason in result.reasons)


def test_historical_unresolved_declaration_is_rejected() -> None:
    result = _validate(_plan(grounding=("793.32",)))
    assert result.plan is None
    assert any(reason.startswith(GROUNDING_EVIDENCE_UNRESOLVED) for reason in result.reasons)


def test_unresolved_declaration_is_reported_with_block_and_declaration() -> None:
    result = _validate(_plan(grounding=("793.32",)))
    assert result.plan is None
    reason = next(item for item in result.reasons if item.startswith(GROUNDING_EVIDENCE_UNRESOLVED))
    assert "block=1" in reason
    assert "793.32" in reason


@pytest.mark.parametrize(
    "declarations",
    [
        "block:0",
        [123],
        [None],
        [{"kind": "SOURCE_EXCERPT", "block_index": 0}],
        ["words:2-6", 123],
    ],
)
def test_provider_parser_preserves_malformed_declarations(declarations: object) -> None:
    request = build_planning_request(_strategy(), _inputs(), STAGE41_CONFIG)
    content = {
        "plans": [
            {
                "strategy_id": "strategy-1",
                "strategy_key": "candidate:ANALYSIS",
                "confidence": 0.7,
                "blocks": [
                    {
                        "block_type": "SOURCE_EXCERPT",
                        "source_role": "HERO",
                        "word_start_index": 2,
                        "word_end_index": 6,
                    },
                    {
                        "block_type": "ORIGINAL_VALUE",
                        "estimated_duration": 4.0,
                        "substantive_value_kind": "SYNTHESIS",
                        "semantic_intent": "Explain the negotiation underneath the dish swap.",
                        "why_unavailable": "The raw excerpt states the swap but not the dynamic.",
                        "grounding_refs": declarations,
                    },
                ],
            }
        ]
    }
    parsed = parse_plan_results(content, [request])["candidate:ANALYSIS"]
    parsed_plan = parsed.plans[0]
    assert parsed_plan.blocks[1].grounding_refs_invalid is True
    result = _validate(parsed_plan)
    assert result.plan is None
    assert any(reason.startswith(GROUNDING_DECLARATION_MALFORMED) for reason in result.reasons)


def test_valid_citation_does_not_hide_another_malformed_declaration() -> None:
    result = _validate(_plan(grounding=("words:2-6", "words:bad")))
    assert result.plan is None
    assert any(reason.startswith(GROUNDING_DECLARATION_MALFORMED) for reason in result.reasons)


@pytest.mark.parametrize(
    "declaration",
    [
        "block:0",
        "words:2-6",
        "source:102-106",
        "w2 w3 w4 w5 w6",
    ],
)
def test_valid_supported_string_declarations_resolve(declaration: str) -> None:
    result = _validate(_plan(grounding=(declaration,)))
    assert result.plan is not None, result.reasons


@pytest.mark.parametrize(
    "declaration",
    [
        {"kind": "SOURCE_EXCERPT", "block_index": 0},
        {"kind": "WORD_RANGE", "word_start_index": 2, "word_end_index": 6},
        {"kind": "SOURCE_SPAN", "source_start": 102.0, "source_end": 106.0},
        {"kind": "QUOTE", "quote": "w2 w3 w4 w5 w6"},
    ],
)
def test_valid_object_declarations_resolve_in_shared_parser(
    declaration: object,
) -> None:
    source_blocks = [
        {
            "index": 0,
            "block_type": "SOURCE_EXCERPT",
            "source_text": "w2 w3 w4 w5 w6",
            "source_start": 102.0,
            "source_end": 107.0,
            "word_start_index": 2,
            "word_end_index": 6,
            "planning_words": [
                {
                    "index": index,
                    "text": f"w{index}",
                    "start": 100.0 + index,
                    "end": 101.0 + index,
                }
                for index in range(2, 7)
            ],
        }
    ]
    text, reason = resolve_reference(declaration, source_blocks)
    assert reason is None
    assert text


@pytest.mark.parametrize(
    "declaration",
    [
        "block:1",
        "block:99",
        "words:1-6",
        "words:2-7",
        "words:99",
        "source:99-103",
        "source:102-120",
    ],
)
def test_non_source_unknown_partial_and_out_of_scope_are_rejected(
    declaration: str,
) -> None:
    result = _validate(_plan(grounding=(declaration,)))
    assert result.plan is None
    assert any(reason.startswith(GROUNDING_EVIDENCE_UNRESOLVED) for reason in result.reasons)


def test_reversed_word_range_is_malformed() -> None:
    result = _validate(_plan(grounding=("words:6-2",)))
    assert result.plan is None
    assert any(reason.startswith(GROUNDING_DECLARATION_MALFORMED) for reason in result.reasons)


def test_external_verification_linked_block_may_omit_source_grounding() -> None:
    value = _value_block(grounding=(), dependency_ids=("claim-1",))
    placeholder = PlanProviderBlock(
        block_type=PlanBlockType.FACT_VERIFICATION_PLACEHOLDER,
        claim_dependency="claim-1",
        verification_rationale="external fact requires verification",
        intended_use="support the claim",
        must_verify_before_execution=True,
        dependent_block_ids=(1,),
    )
    result = _validate(
        _plan(value_block=value, extra=(placeholder,)),
        strategy=_strategy(external="REQUIRES_EXTERNAL_FACT_VERIFICATION"),
    )
    assert result.plan is not None, result.reasons
    assert result.plan.status.value == "PLAN_GENERATED_WITH_VERIFICATION_REQUIRED"


def test_arbitrary_dependency_id_does_not_waive_source_grounding() -> None:
    value = _value_block(grounding=(), dependency_ids=("claim-1",))
    linked_value = _value_block(grounding=("block:0",), dependency_ids=("claim-1",))
    placeholder = PlanProviderBlock(
        block_type=PlanBlockType.FACT_VERIFICATION_PLACEHOLDER,
        claim_dependency="claim-1",
        verification_rationale="external fact requires verification",
        intended_use="support the claim",
        must_verify_before_execution=True,
        dependent_block_ids=(2,),
    )
    plan = PlanProviderPlan(
        strategy_id="strategy-1",
        strategy_key="candidate:ANALYSIS",
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                source_role=SourceExcerptRole.HERO,
                word_start_index=2,
                word_end_index=6,
            ),
            value,
            linked_value,
            placeholder,
        ),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )
    result = _validate(plan, strategy=_strategy(external="REQUIRES_EXTERNAL_FACT_VERIFICATION"))
    assert result.plan is None
    assert any(reason.startswith(GROUNDING_REQUIRED_MISSING) for reason in result.reasons)


def test_non_substantive_block_does_not_require_grounding() -> None:
    transition = PlanProviderBlock(
        block_type=PlanBlockType.TRANSITION,
        purpose="Structural transition",
        estimated_duration=0.5,
    )
    result = _validate(_plan(extra=(transition,)))
    assert result.plan is not None, result.reasons


def test_deterministic_provider_never_uses_strategy_label_as_evidence() -> None:
    assert (
        deterministic_provider_plan(
            _strategy(strategy_type="SOURCE_LED_MINIMAL"), _inputs(), STAGE41_CONFIG
        )
        is None
    )


def test_stage42_persisted_missing_grounding_fails_integrity() -> None:
    plan = make_plan(
        blocks=[source_block(0), original_block(1, grounding=())],
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    ok, reasons = revalidate_integrity(plan, STAGE42_CONFIG)
    assert not ok
    assert any(reason.startswith(GROUNDING_REQUIRED_MISSING) for reason in reasons)


def test_stage42_favorable_critique_never_clears_grounding_integrity() -> None:
    plan = make_plan(
        blocks=[source_block(0), original_block(1, grounding=())],
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    inputs = make_inputs([plan])
    evaluation = evaluate_plan(plan, inputs, STAGE42_CONFIG, provider_mode_deterministic=False)
    apply_semantic_review(evaluation, make_critique(plan), SEMANTIC_AVAILABLE)
    assert evaluation.integrity_ok is False
    assert evaluation.hard_failures
    assert evaluation.verification.get("claim_state") != "GROUNDED_IN_SOURCE"
    governance = build_plan_governance(evaluation, input_fingerprint="in", output_fingerprint="out")
    assert governance.status.value == "REJECTED_BY_GOVERNOR"
    assert governance.eligible_for_stage4_3 is False


def test_stage42_valid_source_grounding_remains_integrity_valid() -> None:
    plan = make_plan()
    ok, reasons = revalidate_integrity(plan, STAGE42_CONFIG)
    assert ok, reasons


def test_stage42_valid_reference_does_not_establish_entailment() -> None:
    plan = make_plan(
        blocks=[
            source_block(0, text="The source speaker discussed the weather and sports."),
            original_block(1, intent="The merger closes next Monday", grounding=("block:0",)),
        ],
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    evaluation = evaluate_plan(
        plan, make_inputs([plan]), STAGE42_CONFIG, provider_mode_deterministic=True
    )
    assert evaluation.integrity_ok is True
    assert evaluation.verification.get("claim_state") != "GROUNDED_IN_SOURCE"


def test_stage43_live_guard_excludes_legacy_ungrounded_approval(
    sqlite_engine: Engine, monkeypatch: Any
) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        settings = FakeGovernanceSettings()
        install_selection_settings(monkeypatch, settings)
        fixture = seed_selection_fixture(
            session, settings=settings, result_specs=[make_result_spec()]
        )
        row = fixture.plans[0]
        row.blocks = [
            {**dict(block), "grounding_refs": []}
            if dict(block).get("block_type") == "ORIGINAL_VALUE"
            else dict(block)
            for block in (row.blocks or [])
        ]
        session.flush()
        failures = _mandatory_failures(session, fixture.candidate, [row])
        assert any(
            reason.startswith(GROUNDING_REQUIRED_MISSING) for reason in failures[str(row.id)]
        )
        decision, _freshness = evaluate_without_persisting(session, fixture.candidate.id)
        assert decision.status is not TransformationSelectionStatus.PLAN_SELECTED
        assert decision.status is not TransformationSelectionStatus.PLAN_SELECTED_WITH_CAUTION
