"""Focused Stage 4.1/4.2/4.3/5.0/6 regressions for confirmed integration defects.

Fixtures reproduce the bounded historical evidence recorded by the isolated
Stage 6 acceptance attempt (2026-10-18). They are deterministic and never call a
provider or media process.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from app.core.enums import (
    ContentType,
    NarrationNeed,
    PlanBlockType,
    SourceExcerptRole,
    SourceMomentStructure,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.materialization.grounding import (
    parse_grounding_refs,
    resolve_grounding,
)
from app.materialization.inputs import (
    FinalClipEvidence,
    MaterializationInputs,
    ProfileContext,
    SourceMediaEvidence,
    VisualPlanEvidence,
    WordEvidence,
)
from app.materialization.policy import DEFAULT_CONFIG as STAGE6_CONFIG
from app.materialization.scripting import validate_authored_block
from app.materialization.types import AuthoredBlock
from app.transformation.governance.policy import DEFAULT_CONFIG as STAGE42_CONFIG
from app.transformation.governance.types import PlanEvidence
from app.transformation.governance.validation import revalidate_integrity
from app.transformation.hook_payoff import (
    HOOK_EVIDENCE_UNRESOLVED,
    HOOK_NOT_COVERED,
    PAYOFF_NOT_COVERED,
    MandatoryEvidence,
    coverage_reasons,
    resolve_mandatory_evidence,
)
from app.transformation.planning.policy import DEFAULT_CONFIG as STAGE41_CONFIG
from app.transformation.planning.types import (
    NarrationRequirement,
    PlanningContext,
    PlanningInputs,
    PlanProviderBlock,
    PlanProviderPlan,
)
from app.transformation.planning.types import (
    WordEvidence as PlanningWordEvidence,
)
from app.transformation.planning.validation import validate_provider_plan

# Historical source segment evidence (authoritative transcript segment indexes).
_HISTORICAL = {
    "1827c524": {
        "hook_index": 343,
        "hook": (583.78, 586.2),
        "refined": (583.78, 620.74),
        "hero": (587.14, 590.74),
    },
    "64e7807c": {
        "hook_index": 101,
        "hook": (177.78, 178.7),
        "refined": (173.48, 221.2),
        "hero": (191.86, 196.2),
    },
    "ac0670c4": {
        "hook_index": 443,
        "hook": (766.08, 766.96),
        "refined": (765.58, 801.48),
        "hero": (786.34, 788.02),
    },
    "a725e62d": {
        "hook_index": 71,
        "hook": (142.52, 143.54),
        "refined": (142.5, 181.92),
        "hero": (142.5, 144.28),
    },
}


def _segments(hook_index: int, hook: tuple[float, float]) -> list[dict[str, object]]:
    segments = [
        {"start": float(i), "end": float(i) + 1.0, "text": f"segment-{i}"} for i in range(476)
    ]
    segments[hook_index] = {
        "start": hook[0],
        "end": hook[1],
        "text": "historical hook segment",
    }
    return segments


def _words(
    refined: tuple[float, float], hero: tuple[float, float]
) -> tuple[PlanningWordEvidence, ...]:
    start, end = refined
    hero_start, hero_end = hero
    words: list[PlanningWordEvidence] = []
    # Eight words before the historical hero selection, eleven within it, then
    # one bounded trailing word. This reproduces the selected plan's index space.
    for index in range(8):
        words.append(
            PlanningWordEvidence(
                index=index,
                text=f"pre{index}",
                start=start + (hero_start - start) * index / 8,
                end=start + (hero_start - start) * (index + 1) / 8,
            )
        )
    for offset in range(11):
        index = 8 + offset
        words.append(
            PlanningWordEvidence(
                index=index,
                text=f"hero{offset}",
                start=hero_start + (hero_end - hero_start) * offset / 11,
                end=hero_start + (hero_end - hero_start) * (offset + 1) / 11,
            )
        )
    words.append(
        PlanningWordEvidence(index=19, text="tail", start=hero_end, end=min(end, hero_end + 1.0))
    )
    return tuple(words)


def _planning_inputs(case: str) -> PlanningInputs:
    data = _HISTORICAL[case]
    hook_index = int(data["hook_index"])
    hook = tuple(data["hook"])  # type: ignore[arg-type]
    refined = tuple(data["refined"])  # type: ignore[arg-type]
    hero = tuple(data["hero"])  # type: ignore[arg-type]
    segments = _segments(hook_index, hook)  # type: ignore[arg-type]
    mandatory = resolve_mandatory_evidence(
        "cafc529e-b0d8-401b-b439-2721a57de73c",
        {"hook_index": hook_index, "payoff_index": None},
        segments,
    )
    return PlanningInputs(
        candidate_id=case,
        candidate_key=f"candidate-{case}",
        source_id="cafc529e-b0d8-401b-b439-2721a57de73c",
        content_type=ContentType.STORY,
        source_moment_structure=SourceMomentStructure.CLAIM,
        source_moment={"hook_index": hook_index, "payoff_index": None},
        transcript=" ".join(word.text for word in _words(refined, hero)),
        transcript_confidence=0.8,
        refined_start=refined[0],
        refined_end=refined[1],
        words=_words(refined, hero),
        word_coverage_sufficient=True,
        context_segments=(),
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
        code_switch={},
        entities=(),
        unresolved_spans=(),
        idea_summary="idea",
        topic_summary="topic",
        hooks=({"type": "CONTRADICTION", "text": "hook"},),
        rights_risk="LOW",
        originality_risk="NOT_INDICATED",
        rights_status="OWNED",
        media_origin="PODCAST_INTERVIEW",
        provenance_snapshot={},
        stage3_risk={"moment_density_score": 0.7},
        stage40_assessments={},
        stage40_platform_risk={},
        stage40_analysis_id="analysis",
        stage40_input_fingerprint="in",
        stage40_output_fingerprint="out",
        stage40_policy_version="stage4.0-v1",
        stage40_strategies=(),
        planning_context=PlanningContext(),
        refinement_priority="FINAL_CLIP",
        refinement_quality_level="FINAL_CLIP",
        refinement_status="FINAL_TRANSCRIPT_READY",
        refinement_output_fingerprint="ref-out",
        mandatory_evidence=mandatory.as_dict(),
    )


def _provider_plan() -> PlanProviderPlan:
    return PlanProviderPlan(
        strategy_id="strategy-1",
        strategy_key="candidate:ANALYSIS",
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                source_role=SourceExcerptRole.HERO,
                word_start_index=8,
                word_end_index=18,
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.ORIGINAL_VALUE,
                estimated_duration=2.0,
                substantive_value_kind=SubstantiveValueKind.SYNTHESIS,
                semantic_intent="Explain the relational negotiation underneath the dish swap.",
                why_unavailable="The raw exchange states the swap but not its negotiation dynamic.",
                grounding_refs=("words:8-18",),
            ),
        ),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )


def _strategy() -> dict[str, object]:
    return {
        "id": "strategy-1",
        "strategy_key": "candidate:ANALYSIS",
        "strategy_type": TransformationStrategyType.ANALYSIS.value,
        "intensity": "MODERATE",
        "rank": 1,
        "strategy_fingerprint": "strategy-fp",
        "external_verification_requirement": "NOT_REQUIRED",
        "confidence": 0.8,
    }


# --- Stage 4.1 -----------------------------------------------------------------


@pytest.mark.parametrize("case", ["1827c524", "64e7807c", "ac0670c4"])
def test_historical_missing_hook_cases_fail_stage41_validation(case: str) -> None:
    result = validate_provider_plan(
        _provider_plan(),
        _strategy(),
        _planning_inputs(case),
        STAGE41_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="provider-fp",
    )
    assert result.plan is None
    assert HOOK_NOT_COVERED in result.reasons
    assert PAYOFF_NOT_COVERED not in result.reasons


def test_valid_hook_coverage_still_generates_a_plan() -> None:
    result = validate_provider_plan(
        _provider_plan(),
        _strategy(),
        _planning_inputs("a725e62d"),
        STAGE41_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="provider-fp",
    )
    assert result.plan is not None
    assert result.plan.hook_payoff_evidence["hook_index"] == 71
    assert result.plan.hook_payoff_evidence["payoff_index"] is None


def test_optional_absent_evidence_is_not_a_failure() -> None:
    evidence = MandatoryEvidence()
    assert coverage_reasons(None, None, evidence, []) == ()


def test_unresolvable_declared_hook_is_not_satisfied() -> None:
    evidence = MandatoryEvidence(hook_declared=True, hook_error="UNRESOLVED")
    reasons = coverage_reasons(999, None, evidence, [(0.0, 1000.0)])
    assert HOOK_EVIDENCE_UNRESOLVED in reasons


# --- Stage 4.2 / 4.3 -----------------------------------------------------------


def _governance_plan(case: str, *, payoff_index: int | None = None) -> PlanEvidence:
    data = _HISTORICAL[case]
    hero = tuple(data["hero"])  # type: ignore[arg-type]
    return PlanEvidence(
        plan_id="plan-1",
        plan_key="plan-key",
        plan_output_fingerprint="plan-fp",
        provider_input_fingerprint="provider-fp",
        strategy_id="strategy-1",
        strategy_key="candidate:ANALYSIS",
        strategy_type="ANALYSIS",
        strategy_fingerprint="strategy-fp",
        strategy_rank=1,
        intensity="MODERATE",
        status="PLAN_GENERATED",
        generation_rank=1,
        is_current=True,
        planner_confidence=0.8,
        blocks=(
            {
                "index": 0,
                "block_type": "SOURCE_EXCERPT",
                "source_role": "HERO",
                "source_start": hero[0],
                "source_end": hero[1],
                "estimated_duration": hero[1] - hero[0],
            },
        ),
        hero_block_index=0,
        hero_source_start=hero[0],
        hero_source_end=hero[1],
        hero_appearance_time=0.0,
        narration={"need": "NONE", "requirements": {}},
        verification_dependencies=(),
        derived_durations={},
        hook_payoff_evidence={
            "hook_index": int(data["hook_index"]),
            "payoff_index": payoff_index,
        },
        original_value_kinds=(),
        original_value_reasons=(),
        preservation_constraints=(),
        required_context=(),
        degraded_rules=(),
        stage40_risk={},
        source_dialect={},
        target_audience={},
        strategy_snapshot={},
    )


@pytest.mark.parametrize("case", ["1827c524", "64e7807c", "ac0670c4"])
def test_stage42_integrity_blocks_legacy_missing_hook(case: str) -> None:
    data = _HISTORICAL[case]
    segments = _segments(int(data["hook_index"]), tuple(data["hook"]))  # type: ignore[arg-type]
    mandatory = resolve_mandatory_evidence(
        "cafc529e-b0d8-401b-b439-2721a57de73c",
        {"hook_index": int(data["hook_index"]), "payoff_index": None},
        segments,
    )
    ok, reasons = revalidate_integrity(
        _governance_plan(case), STAGE42_CONFIG, mandatory_evidence=mandatory.as_dict()
    )
    assert not ok
    assert HOOK_NOT_COVERED in reasons
    assert PAYOFF_NOT_COVERED not in reasons


def test_stage42_reports_payoff_and_both_distinctly() -> None:
    data = _HISTORICAL["1827c524"]
    segments = _segments(int(data["hook_index"]), tuple(data["hook"]))  # type: ignore[arg-type]
    mandatory = resolve_mandatory_evidence(
        "source",
        {"hook_index": 343, "payoff_index": 344},
        segments,
    )
    ok, reasons = revalidate_integrity(
        _governance_plan("1827c524", payoff_index=344),
        STAGE42_CONFIG,
        mandatory_evidence=mandatory.as_dict(),
    )
    assert not ok
    assert HOOK_NOT_COVERED in reasons
    assert PAYOFF_NOT_COVERED in reasons


# --- Stage 5.0 -----------------------------------------------------------------


def test_historical_hook_failure_is_classified_as_hook_at_stage5() -> None:
    from app.render.compatibility import evaluate_compatibility
    from app.render.policy import HOOK_NOT_COVERED as RENDER_HOOK_NOT_COVERED
    from app.render.policy import MATERIAL_TIMING_CHANGE
    from app.render.types import ClipWord
    from app.render.types import FinalClipEvidence as RenderFinalClipEvidence

    data = _HISTORICAL["1827c524"]
    segments = _segments(343, tuple(data["hook"]))  # type: ignore[arg-type]
    final = RenderFinalClipEvidence(
        refinement_id="ref",
        priority="FINAL_CLIP",
        quality_level="FINAL_CLIP",
        status="FINAL_TRANSCRIPT_READY",
        output_fingerprint="final-fp",
        final_transcript="hero words at the later moment",
        words=tuple(
            ClipWord(index=i, text=token, start=587.14 + i, end=587.14 + i + 0.9)
            for i, token in enumerate("hero words at the later moment".split())
        ),
        refined_start=587.14,
        refined_end=590.74,
    )
    block = {
        "index": 0,
        "block_type": "SOURCE_EXCERPT",
        "source_start": 587.14,
        "source_end": 590.74,
        "source_text": final.final_transcript,
    }
    result = evaluate_compatibility(
        blocks=[block],
        hero_block_index=0,
        hook_payoff_evidence={"hook_index": 343, "payoff_index": None},
        final=final,
        source_segments=segments,
        planning_refined_start=583.78,
        planning_refined_end=620.74,
        exact_identity_match=False,
        caption_source_fingerprint="caption",
        planning_refinement_id="plan-ref",
        planning_refinement_priority="FINAL_CLIP",
        planning_refinement_quality_level="FINAL_CLIP",
        planning_output_fingerprint="plan-fp",
    )
    assert result.outcome == MATERIAL_TIMING_CHANGE
    assert RENDER_HOOK_NOT_COVERED in result.reason_codes
    assert "PAYOFF_NOT_COVERED" not in result.reason_codes


# --- Stage 6 grounding ---------------------------------------------------------


def _materialization_inputs() -> MaterializationInputs:
    hero_text = "أوه نزلي أكي نزلك إيه؟"
    support_text = (
        "حاجة مش عارفة أكيد ما أبعنبش وما جايبيني عامل موسك ده جبنة وخيار أوه طب تيجي مبدل؟ ماشي خد"
    )
    final_text = f"{hero_text} {support_text}"
    hero_tokens = hero_text.split()
    support_tokens = support_text.split()
    hero_step = (144.28 - 142.5) / len(hero_tokens)
    support_step = (150.66 - 144.28) / len(support_tokens)
    timed: list[tuple[str, float, float]] = []
    for index, token in enumerate(hero_tokens):
        timed.append((token, 142.5 + index * hero_step, 142.5 + (index + 1) * hero_step))
    for index, token in enumerate(support_tokens):
        timed.append((token, 144.28 + index * support_step, 144.28 + (index + 1) * support_step))
    final_words = tuple(
        WordEvidence(index=i, text=token, start=start, end=end)
        for i, (token, start, end) in enumerate(timed)
    )
    planning_words = tuple(
        {"index": i, "text": token, "start": start, "end": end}
        for i, (token, start, end) in enumerate(timed)
    )
    blocks = (
        {
            "index": 0,
            "block_type": "SOURCE_EXCERPT",
            "source_role": "HERO",
            "source_start": 142.5,
            "source_end": 144.28,
            "word_start_index": 5,
            "word_end_index": 9,
            "source_text": hero_text,
            "planning_source_text": hero_text,
            "planning_source_start": 142.5,
            "planning_source_end": 144.28,
            "planning_word_start_index": 3,
            "planning_word_end_index": 7,
            "planning_words": planning_words[3:8],
        },
        {
            "index": 1,
            "block_type": "SOURCE_EXCERPT",
            "source_role": "SUPPORT",
            "source_start": 144.28,
            "source_end": 150.66,
            "word_start_index": 10,
            "word_end_index": 28,
            "source_text": support_text,
            "planning_source_text": support_text,
            "planning_source_start": 144.28,
            "planning_source_end": 150.66,
            "planning_word_start_index": 8,
            "planning_word_end_index": 26,
            "planning_words": planning_words[8:27],
        },
        {
            "index": 2,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "semantic_intent": "Highlight the negotiation dynamic underneath the dish swap.",
            "why_unavailable": "The raw exchange states the swap but not the negotiation dynamic.",
            "grounding_refs": ["words:3-7", "words:8-26"],
            "word_start_index": None,
            "word_end_index": None,
        },
    )
    return MaterializationInputs(
        candidate_id="a725e62d-bec3-445d-93b4-76815febfde4",
        candidate_key="candidate",
        source_id="cafc529e-b0d8-401b-b439-2721a57de73c",
        priority="CANDIDATE",
        selection_id="selection",
        selection_output_fingerprint="selection-fp",
        selected_plan_fingerprint="plan-fp",
        plan_id="plan",
        plan_output_fingerprint="plan-output",
        strategy_type="ANALYSIS",
        intensity="MODERATE",
        blocks=blocks,
        hero_block_index=0,
        narration={"need": "NONE", "requirements": {}},
        verification_dependencies=(),
        original_value_kinds=("SYNTHESIS",),
        required_context=(),
        preservation_constraints=(),
        plan_target_context={},
        contract_id="contract",
        contract_status="MATERIALIZATION_REQUIRED",
        contract_input_fingerprint="contract-in",
        contract_output_fingerprint="contract-out",
        execution_slots=(),
        contract_source_bindings=(
            {
                "block_index": 0,
                "final_clip_start": 142.5,
                "final_clip_end": 144.28,
                "final_clip_word_start": 5,
                "final_clip_word_end": 9,
                "final_clip_text": hero_text,
            },
            {
                "block_index": 1,
                "final_clip_start": 144.28,
                "final_clip_end": 150.66,
                "final_clip_word_start": 10,
                "final_clip_word_end": 28,
                "final_clip_text": support_text,
            },
        ),
        final_clip=FinalClipEvidence(
            refinement_id="ref",
            priority="FINAL_CLIP",
            quality_level="FINAL_CLIP",
            status="FINAL_TRANSCRIPT_READY",
            output_fingerprint="ref-out",
            text=final_text,
            words=final_words,
            start=142.5,
            end=150.66,
        ),
        visual_plan=VisualPlanEvidence(
            visual_plan_id="vis",
            output_fingerprint="vis-out",
            plan_ready=True,
            ass_relative_path="captions.ass",
            ass_sha256="a" * 64,
            ass_policy_version="v1",
            caption_source_fingerprint="caption",
            framing_fingerprint="framing",
            source_media_fingerprint="media",
        ),
        source_media=SourceMediaEvidence(
            relative_path="source.webm",
            content_hash="hash",
            size_bytes=1,
            mtime_ns=1,
            duration=900.0,
            frame_rate=Fraction(25, 1),
            display_width=1920,
            display_height=1080,
            encoded_width=1920,
            encoded_height=1080,
            sample_rate=48000,
            channels=2,
        ),
        profile=ProfileContext(
            profile_id="profile",
            profile_key="stage6-acceptance-no-narration",
            target_market="UNSPECIFIED",
            output_language_policy="SOURCE_LANGUAGE",
            register_intent="SOURCE_COMPATIBLE",
            narration_allowed=False,
        ),
    )


def test_historical_words_references_survive_and_resolve_exactly() -> None:
    inputs = _materialization_inputs()
    raw = ["words:3-7", "words:8-26"]
    refs = parse_grounding_refs(raw)
    assert [ref.raw for ref in refs] == raw
    resolution = resolve_grounding(inputs, refs)
    assert resolution.unresolved == ()
    assert "أوه نزلي" in resolution.text
    assert "جبنة وخيار" in resolution.text


def test_planning_and_final_clip_index_spaces_are_not_confused() -> None:
    inputs = _materialization_inputs()
    refs = parse_grounding_refs(["words:3-7"])
    resolution = resolve_grounding(inputs, refs)
    assert resolution.unresolved == ()
    assert resolution.text == inputs.blocks[0]["source_text"]


def test_grounding_refs_round_trip_through_authored_block_and_ledger() -> None:
    inputs = _materialization_inputs()
    result = validate_authored_block(
        {
            "block_index": 2,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "The swap exposes a negotiation about the assigned dish.",
            "grounding_refs": ["words:3-7", "words:8-26"],
        },
        inputs,
        STAGE6_CONFIG,
    )
    assert result.claim is not None
    assert [ref.raw for ref in result.claim.grounding_refs] == [
        "words:3-7",
        "words:8-26",
    ]
    ledger = result.claim.as_dict()
    reparsed = parse_grounding_refs(ledger["grounding_refs"])
    assert [ref.raw for ref in reparsed] == ["words:3-7", "words:8-26"]
    authored = AuthoredBlock(
        block_index=2,
        block_type="ORIGINAL_VALUE",
        slot_id="block-2",
        substantive_value_kind="SYNTHESIS",
        copy="copy",
        adds_beyond_source="value",
        claim_class="SOURCE_GROUNDED_INFERENCE",
        grounding_refs=result.claim.grounding_refs,
    )
    assert [ref["raw"] for ref in authored.as_dict()["grounding_refs"]] == [
        "words:3-7",
        "words:8-26",
    ]


@pytest.mark.parametrize(
    ("declaration", "reason"),
    [
        ("words:3-", "malformed_word_range"),
        ("words:3.5-7", "malformed_word_range"),
        ("words:7-3", "reversed_word_range"),
        ("words:-1-3", "negative_word_index"),
        ("block:abc", "unsupported_reference_label"),
        ("strategy:foo", "unsupported_reference_label"),
        (123, "unsupported_reference_type"),
        ({"kind": "WORD_RANGE"}, "missing_or_invalid_word_start_index"),
        ({"kind": "BOGUS"}, "unsupported_reference_kind"),
        ({"kind": "SOURCE_EXCERPT", "block_index": "0"}, "missing_or_invalid_block_index"),
        (
            {"kind": "SOURCE_SPAN", "source_start": float("nan"), "source_end": 2.0},
            "missing_or_non_finite_source_start",
        ),
        ({"kind": "FINAL_CLIP", "claim_id": 4}, "claim_id_not_string"),
    ],
)
def test_malformed_and_unsupported_declarations_fail_explicitly(
    declaration: object, reason: str
) -> None:
    refs = parse_grounding_refs([declaration])
    assert len(refs) == 1
    assert refs[0].kind == "INVALID"
    assert refs[0].invalid_reason == reason
    resolution = resolve_grounding(_materialization_inputs(), refs)
    assert resolution.unresolved


def test_malformed_container_is_not_an_empty_reference_list() -> None:
    refs = parse_grounding_refs("words:3-7")
    assert len(refs) == 1
    assert refs[0].kind == "INVALID"
    assert refs[0].invalid_reason == "invalid_reference_container"


def test_mixed_valid_and_invalid_references_fail() -> None:
    refs = parse_grounding_refs(["words:3-7", "words:bad"])
    assert [ref.kind for ref in refs] == ["WORD_RANGE", "INVALID"]
    result = validate_authored_block(
        {
            "block_index": 2,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "The swap exposes a negotiation about the assigned dish.",
            "grounding_refs": ["words:3-7", "words:bad"],
        },
        _materialization_inputs(),
        STAGE6_CONFIG,
    )
    assert any("INVALID_GROUNDING_REFERENCE" in finding.reason_codes for finding in result.findings)
    assert not result.accepted


def test_partial_overlap_and_out_of_scope_do_not_escape() -> None:
    inputs = _materialization_inputs()
    for ref in ("words:2-7", "words:3-30", "words:0-26"):
        resolution = resolve_grounding(inputs, parse_grounding_refs([ref]))
        assert resolution.unresolved


def test_nonexistent_and_non_source_blocks_fail() -> None:
    inputs = _materialization_inputs()
    for ref in ("block:99", "block:2"):
        resolution = resolve_grounding(inputs, parse_grounding_refs([ref]))
        assert resolution.unresolved


def test_valid_object_forms_resolve_exactly() -> None:
    inputs = _materialization_inputs()
    refs = parse_grounding_refs(
        [
            {"kind": "SOURCE_EXCERPT", "block_index": 0},
            {"kind": "WORD_RANGE", "word_start_index": 3, "word_end_index": 7},
            {"kind": "SOURCE_SPAN", "source_start": 142.5, "source_end": 144.28},
            {"kind": "FINAL_CLIP"},
            {"kind": "QUOTE", "quote": inputs.blocks[0]["source_text"]},
        ]
    )
    assert all(ref.kind != "INVALID" for ref in refs)
    resolution = resolve_grounding(inputs, refs)
    assert resolution.unresolved == ()
    assert resolution.text


def test_valid_grounding_does_not_approve_an_invented_number() -> None:
    result = validate_authored_block(
        {
            "block_index": 2,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "The swap exposes a 47 percent negotiation about the assigned dish.",
            "grounding_refs": ["words:3-7", "words:8-26"],
        },
        _materialization_inputs(),
        STAGE6_CONFIG,
    )
    assert not result.accepted
    assert any("INVENTED_NUMBER" in finding.reason_codes for finding in result.findings)


def test_valid_grounding_does_not_approve_an_unsupported_essential_claim() -> None:
    result = validate_authored_block(
        {
            "block_index": 2,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "The dish was chosen by a famous chef in Paris.",
            "grounding_refs": ["words:3-7", "words:8-26"],
        },
        _materialization_inputs(),
        STAGE6_CONFIG,
    )
    assert not result.accepted
