"""Stage 5.0 FINAL_CLIP compatibility evaluation (pure, no DB, no provider)."""

from __future__ import annotations

import pytest
from stage50_support import (
    caption_fingerprint_for,
    clip_words,
    make_final,
    source_block,
)

from app.render.compatibility import evaluate_compatibility
from app.render.policy import (
    COMPATIBLE_NON_MATERIAL_CHANGE,
    EXACT_MATCH,
    MATERIAL_SEMANTIC_CHANGE,
    MATERIAL_TIMING_CHANGE,
    SOURCE_SPAN_NO_LONGER_VALID,
    UNRESOLVED_COMPATIBILITY,
)

BASE = "remote work collapsed productivity and promotion rates fell sharply"
BASE_TEXT = (
    "The guest argues that remote work collapsed productivity and promotion rates fell sharply"
)


def evaluate(
    plan_text: str,
    final_text: str,
    *,
    blocks=None,
    final=None,
    hook_payoff=None,
    source_segments=(),
    exact=False,
    planning_start=20.5,
    planning_end=44.5,
):
    final = final if final is not None else make_final(final_text)
    block_list = blocks if blocks is not None else [source_block(plan_text)]
    return evaluate_compatibility(
        blocks=block_list,
        hero_block_index=0,
        hook_payoff_evidence=hook_payoff or {"hook_index": None, "payoff_index": None},
        final=final,
        source_segments=source_segments,
        planning_refined_start=planning_start,
        planning_refined_end=planning_end,
        exact_identity_match=exact,
        caption_source_fingerprint=caption_fingerprint_for(final),
        planning_refinement_id="plan-ref",
        planning_refinement_priority="CANDIDATE",
        planning_refinement_quality_level="CANDIDATE",
        planning_output_fingerprint="plan-fp",
    )


def test_exact_identity_match_passes_structural_binding() -> None:
    result = evaluate(BASE_TEXT, BASE_TEXT, exact=True)
    assert result.outcome == EXACT_MATCH
    assert result.exact_match is True


def test_punctuation_and_normalization_only_change_is_compatible() -> None:
    result = evaluate(BASE_TEXT, BASE_TEXT + " !!!")
    assert result.outcome == COMPATIBLE_NON_MATERIAL_CHANGE


def test_recovered_harmless_english_token_is_compatible() -> None:
    result = evaluate(BASE_TEXT, BASE_TEXT + " indeed")
    assert result.outcome == COMPATIBLE_NON_MATERIAL_CHANGE
    verdict = result.per_block[0]
    assert verdict.recovered_code_switch_tokens


def test_negation_change_is_material_semantic() -> None:
    result = evaluate(BASE_TEXT, BASE_TEXT.replace("collapsed", "did not collapse"))
    assert result.outcome == MATERIAL_SEMANTIC_CHANGE


def test_modality_change_is_material_semantic() -> None:
    result = evaluate("The rate may fall", "The rate will fall")
    assert result.outcome == MATERIAL_SEMANTIC_CHANGE


def test_exclusivity_change_is_material_semantic() -> None:
    result = evaluate("the plan works for everyone", "the plan works only for everyone")
    assert result.outcome == MATERIAL_SEMANTIC_CHANGE


def test_named_entity_change_is_material_semantic() -> None:
    result = evaluate("Google reported growth", "Facebook reported growth")
    assert result.outcome == MATERIAL_SEMANTIC_CHANGE


def test_numeric_change_is_material_semantic() -> None:
    result = evaluate("unemployment rose 20 percent", "unemployment rose 30 percent")
    assert result.outcome == MATERIAL_SEMANTIC_CHANGE


def test_grounding_quote_lost_is_material_semantic() -> None:
    plan = "remote work collapsed productivity and promotion rates fell sharply"
    blocks = [
        source_block(plan),
        {
            "index": 1,
            "block_type": "ORIGINAL_VALUE",
            "purpose": "commentary",
            "semantic_intent": "explain",
            "grounding_refs": ["promotion rates fell sharply"],
        },
    ]
    final = make_final("remote work collapsed productivity and promotion rates dropped")
    result = evaluate(plan, "", blocks=blocks, final=final)
    assert result.outcome == MATERIAL_SEMANTIC_CHANGE


def test_payoff_not_covered_is_material_timing() -> None:
    segments = [
        {"start": 0.0, "end": 10.0, "text": "intro"},
        {"start": 70.0, "end": 80.0, "text": "payoff"},
    ]
    result = evaluate(
        BASE_TEXT,
        BASE_TEXT,
        hook_payoff={"hook_index": None, "payoff_index": 1},
        source_segments=segments,
    )
    assert result.outcome == MATERIAL_TIMING_CHANGE


def test_meaning_critical_unresolved_fails_closed() -> None:
    final = make_final(
        BASE_TEXT,
        unresolved_spans=(
            {
                "meaning_critical": True,
                "resolution_state": "UNRESOLVED",
                "start": 25.0,
                "end": 27.0,
            },
        ),
    )
    result = evaluate(BASE_TEXT, BASE_TEXT, final=final)
    assert result.outcome == UNRESOLVED_COMPATIBILITY


def test_word_evidence_insufficient_fails_closed() -> None:
    final = make_final(BASE_TEXT, words=())
    result = evaluate(BASE_TEXT, BASE_TEXT, final=final)
    assert result.outcome == UNRESOLVED_COMPATIBILITY


def test_source_span_out_of_planning_bounds_is_invalid() -> None:
    blocks = [source_block(BASE_TEXT, source_start=5.0, source_end=6.0)]
    result = evaluate(BASE_TEXT, BASE_TEXT, blocks=blocks)
    assert result.outcome == SOURCE_SPAN_NO_LONGER_VALID


def test_total_alignment_failure_is_invalid() -> None:
    text = "completely unrelated words here now"
    words = clip_words("nothing matches at all in this region", start=21.0, end=30.0)
    block = source_block(
        text,
        source_start=21.0,
        source_end=24.0,
        word_start_index=0,
        word_end_index=2,
    )
    final = make_final("nothing matches at all in this region", words=words)
    result = evaluate(text, "", blocks=[block], final=final)
    assert result.outcome == SOURCE_SPAN_NO_LONGER_VALID


def test_bounded_alignment_rebinds_word_indexes() -> None:
    text = "alpha beta gamma delta epsilon"
    words = clip_words(text, start=21.0, end=30.0)
    block = source_block(
        text,
        source_start=words[1].start,
        source_end=words[3].end,
        word_start_index=1,
        word_end_index=3,
    )
    final = make_final(text, words=words)
    result = evaluate(text, text, blocks=[block], final=final)
    verdict = result.per_block[0]
    assert verdict.structural_valid is True
    assert verdict.rebound_word_start is not None
    assert verdict.rebound_end is not None


@pytest.mark.parametrize(
    "plan,final,expected",
    [
        ("the cat sat on the mat", "the cat sat on the mat", COMPATIBLE_NON_MATERIAL_CHANGE),
        ("the cat sat", "the cat sat quickly", COMPATIBLE_NON_MATERIAL_CHANGE),
    ],
)
def test_small_wording_change_is_compatible(plan: str, final: str, expected: str) -> None:
    result = evaluate(plan, final)
    assert result.outcome == expected
