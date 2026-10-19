"""Stage 4.1/4.2/4.3 authoritative grounding and approval regressions."""

from __future__ import annotations

from types import SimpleNamespace
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
from app.materialization.grounding import parse_grounding_refs, resolve_grounding
from app.materialization.inputs import FinalClipEvidence
from app.materialization.inputs import WordEvidence as FinalWordEvidence
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


def _span_words() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "index": index,
            "text": f"w{index}",
            "start": 100.0 + index,
            "end": 101.0 + index,
        }
        for index in range(2, 7)
    )


def _span_source_block(
    *,
    index: int = 0,
    start: float = 102.0,
    duration: float = 5.0,
    words: tuple[dict[str, object], ...] | None = None,
    text: str = "w2 w3 w4 w5 w6",
) -> dict[str, object]:
    chosen = _span_words() if words is None else words
    first = chosen[0]["index"]
    last = chosen[-1]["index"]
    return {
        "index": index,
        "block_type": "SOURCE_EXCERPT",
        "source_role": "HERO",
        "source_start": start,
        "source_end": start + duration,
        "source_text": text,
        "planning_source_text": text,
        "planning_source_start": start,
        "planning_source_end": start + duration,
        "word_start_index": first,
        "word_end_index": last,
        "planning_word_start_index": first,
        "planning_word_end_index": last,
        "planning_words": list(chosen),
    }


def _stage6_inputs(
    *,
    blocks: tuple[dict[str, object], ...] | None = None,
    final_words: tuple[dict[str, object], ...] | None = None,
    final_start: float = 102.0,
    final_end: float = 107.0,
) -> SimpleNamespace:
    words = _span_words() if final_words is None else final_words
    final = FinalClipEvidence(
        refinement_id="ref",
        priority="FINAL_CLIP",
        quality_level="FINAL_CLIP",
        status="FINAL_TRANSCRIPT_READY",
        output_fingerprint="final-fp",
        text=" ".join(str(word["text"]) for word in words),
        words=tuple(
            FinalWordEvidence(
                index=int(word["index"]),
                text=str(word["text"]),
                start=float(word["start"]),
                end=float(word["end"]),
            )
            for word in words
        ),
        start=final_start,
        end=final_end,
    )
    return SimpleNamespace(blocks=blocks or (_span_source_block(),), final_clip=final)


def _stage6_resolution(declaration: str, **kwargs: Any) -> Any:
    return resolve_grounding(_stage6_inputs(**kwargs), parse_grounding_refs([declaration]))


@pytest.mark.parametrize(
    "declaration",
    [
        "words:0-100000000",
        "words:100000000-200000000",
        {
            "kind": "WORD_RANGE",
            "word_start_index": 0,
            "word_end_index": 100000000,
        },
        "words:100000000-100000004",
    ],
)
def test_huge_word_ranges_fail_before_allocation(declaration: object) -> None:
    text, reason = resolve_reference(declaration, (_span_source_block(),))
    assert text is None
    assert reason == "word_range_not_fully_covered_by_selected_evidence"


@pytest.mark.parametrize(
    "declaration",
    ["words:0-100000000", "words:100000000-100000004"],
)
def test_huge_word_ranges_fail_at_stage41(declaration: str) -> None:
    result = _validate(_plan(grounding=(declaration,)))
    assert result.plan is None
    assert any(reason.startswith(GROUNDING_EVIDENCE_UNRESOLVED) for reason in result.reasons)


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("word:3", "w3"),
        ("words:3-5", "w3 w4 w5"),
        ("words:2-6", "w2 w3 w4 w5 w6"),
    ],
)
def test_valid_word_ranges_resolve_exactly(declaration: str, expected: str) -> None:
    text, reason = resolve_reference(declaration, (_span_source_block(),))
    assert reason is None
    assert text == expected


@pytest.mark.parametrize(
    "declaration",
    [
        "words:1-3",
        "words:2-7",
        "words:99",
        "words:0-2",
    ],
)
def test_missing_or_partial_word_ranges_do_not_expand_evidence(
    declaration: str,
) -> None:
    text, reason = resolve_reference(declaration, (_span_source_block(),))
    assert text is None
    assert reason == "word_range_not_fully_covered_by_selected_evidence"


def test_valid_word_range_spans_supported_selected_excerpts() -> None:
    first = _span_source_block(
        words=_span_words()[:2],
        start=102.0,
        duration=2.0,
        text="w2 w3",
    )
    second = _span_source_block(
        index=1,
        words=_span_words()[2:],
        start=104.0,
        duration=3.0,
        text="w4 w5 w6",
    )
    text, reason = resolve_reference("words:2-6", (first, second))
    assert reason is None
    assert text == "w2 w3 w4 w5 w6"


def test_sparse_word_indexes_do_not_fill_gaps() -> None:
    first = _span_source_block(
        words=(_span_words()[0], _span_words()[1]),
        start=102.0,
        duration=2.0,
        text="w2 w3",
    )
    trailing = {
        "index": 7,
        "text": "w7",
        "start": 107.0,
        "end": 108.0,
    }
    second = _span_source_block(
        index=1,
        words=(*_span_words()[3:], trailing),
        start=104.0,
        duration=4.0,
        text="w5 w6 w7",
    )
    text, reason = resolve_reference("words:2-6", (first, second))
    assert text is None
    assert reason == "word_range_not_fully_covered_by_selected_evidence"


@pytest.mark.parametrize(
    "declaration",
    [
        "source:102.2-102.3",
        "source:102.2-106.7",
    ],
)
def test_historical_time_references_rejected_at_stage41(declaration: str) -> None:
    result = _validate(_plan(grounding=(declaration,)))
    assert result.plan is None
    assert any(reason.startswith(GROUNDING_EVIDENCE_UNRESOLVED) for reason in result.reasons)


@pytest.mark.parametrize(
    "declaration",
    [
        "source:102.2-102.3",
        "source:102.2-106.7",
        "source:102-106",
        "source:102-107",
        "source:103.5",
        "source:99.5",
        "source:102.5-106.5",
    ],
)
def test_stage4_and_stage6_time_reference_parity(declaration: str) -> None:
    stage4_text, stage4_reason = resolve_reference(declaration, (_span_source_block(),))
    stage6 = _stage6_resolution(declaration)
    assert (stage4_reason is None) == (not stage6.unresolved)
    if stage4_reason is None:
        assert stage4_text == stage6.text


def test_stage4_and_stage6_agree_on_point_in_word_gap() -> None:
    words = list(_span_words())
    words[2] = {**words[2], "start": 104.5, "end": 105.5}
    blocks = (_span_source_block(words=tuple(words)),)
    stage4_text, stage4_reason = resolve_reference("source:104.2", blocks)
    stage6 = _stage6_resolution("source:104.2", blocks=blocks, final_words=tuple(words))
    assert stage4_text is None
    assert stage4_reason is not None
    assert stage6.unresolved


def test_time_reference_across_uncovered_selected_scope_gap_is_rejected() -> None:
    first = _span_source_block(
        words=(_span_words()[0], _span_words()[1]),
        start=102.0,
        duration=2.0,
        text="w2 w3",
    )
    second = _span_source_block(
        index=1,
        words=_span_words()[3:],
        start=106.0,
        duration=2.0,
        text="w5 w6",
    )
    blocks = (first, second)
    stage4_text, stage4_reason = resolve_reference("source:103-107", blocks)
    stage6 = _stage6_resolution("source:103-107", blocks=blocks)
    assert stage4_text is None
    assert stage4_reason == "source_span_out_of_selected_scope"
    assert stage6.unresolved


def test_missing_word_evidence_rejects_narrow_time_references() -> None:
    block = {
        key: value
        for key, value in _span_source_block().items()
        if key
        not in {
            "word_start_index",
            "word_end_index",
            "planning_word_start_index",
            "planning_word_end_index",
            "planning_words",
        }
    }
    for declaration in ("source:102-103", "source:103.5", "source:102.2-106.7"):
        text, reason = resolve_reference(declaration, (block,))
        assert text is None
        assert reason is not None


def test_valid_whole_block_reference_survives_without_word_evidence() -> None:
    block = {
        key: value
        for key, value in _span_source_block().items()
        if key
        not in {
            "word_start_index",
            "word_end_index",
            "planning_word_start_index",
            "planning_word_end_index",
            "planning_words",
        }
    }
    text, reason = resolve_reference("source:102-107", (block,))
    assert reason is None
    assert text == "w2 w3 w4 w5 w6"
    block_text, block_reason = resolve_reference("block:0", (block,))
    assert block_reason is None
    assert block_text == text


def test_later_final_clip_changes_are_a_legitimate_downstream_rejection() -> None:
    blocks = (_span_source_block(),)
    stage4_text, stage4_reason = resolve_reference("source:102-106", blocks)
    stage6 = _stage6_resolution("source:102-106", final_words=_span_words()[:3])
    assert stage4_reason is None
    assert stage4_text == "w2 w3 w4 w5"
    assert stage6.unresolved
    assert "final_clip_range_not_covered" in " ".join(stage6.unresolved)


@pytest.mark.parametrize(
    "declaration",
    [
        "source:102.2-102.3",
        "source:102.2-106.7",
    ],
)
def test_stage42_invalid_time_reference_fails_integrity(declaration: str) -> None:
    source = source_block(0, start=102.0, duration=5.0)
    source.update(
        {
            "source_text": "w2 w3 w4 w5 w6",
            "word_start_index": 2,
            "word_end_index": 6,
            "planning_word_start_index": 2,
            "planning_word_end_index": 6,
            "planning_words": list(_span_words()),
        }
    )
    plan = make_plan(
        blocks=[source, original_block(1, grounding=(declaration,))],
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    ok, reasons = revalidate_integrity(plan, STAGE42_CONFIG)
    assert not ok
    assert any(reason.startswith(GROUNDING_EVIDENCE_UNRESOLVED) for reason in reasons)


def test_stage42_favorable_critique_never_clears_invalid_time_reference() -> None:
    source = source_block(0, start=102.0, duration=5.0)
    source.update(
        {
            "source_text": "w2 w3 w4 w5 w6",
            "word_start_index": 2,
            "word_end_index": 6,
            "planning_word_start_index": 2,
            "planning_word_end_index": 6,
            "planning_words": list(_span_words()),
        }
    )
    plan = make_plan(
        blocks=[source, original_block(1, grounding=("source:102.2-106.7",))],
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


def test_stage43_live_guard_excludes_persisted_invalid_time_reference(
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
            {**dict(block), "grounding_refs": ["source:20.6-44.4"]}
            if dict(block).get("block_type") == "ORIGINAL_VALUE"
            else dict(block)
            for block in (row.blocks or [])
        ]
        session.flush()
        failures = _mandatory_failures(session, fixture.candidate, [row])
        assert any(
            reason.startswith(GROUNDING_EVIDENCE_UNRESOLVED) for reason in failures[str(row.id)]
        )
        decision, _freshness = evaluate_without_persisting(session, fixture.candidate.id)
        assert decision.status is not TransformationSelectionStatus.PLAN_SELECTED
        assert decision.status is not TransformationSelectionStatus.PLAN_SELECTED_WITH_CAUTION
