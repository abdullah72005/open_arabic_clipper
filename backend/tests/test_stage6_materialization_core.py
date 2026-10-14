"""Stage 6 deterministic materialization core (no providers, no media)."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from app.core.enums import MaterializationOutcome, QualityLevel
from app.materialization.acceptance import AssessmentContext, assess_materialization
from app.materialization.fingerprints import (
    input_fingerprint,
    narration_fingerprint,
)
from app.materialization.inputs import (
    FinalClipEvidence,
    MaterializationInputs,
    ProfileContext,
    SourceMediaEvidence,
    VisualPlanEvidence,
    WordEvidence,
)
from app.materialization.policy import DEFAULT_CONFIG
from app.materialization.profiles import semantic_context_matches
from app.materialization.scripting import (
    all_required_slots_fulfilled,
    classify_claim,
    resolve_slots,
    validate_authored_block,
)
from app.materialization.timeline import build_audio_recipe, build_timeline, hero_is_protected
from app.materialization.types import AuthoredBlock, GroundingRef

TRANSCRIPT = (
    "The guest argues that remote work collapsed productivity because managers lost "
    "the ability to mentor junior staff."
)


def make_inputs(**overrides: object) -> MaterializationInputs:
    blocks = (
        {
            "index": 0,
            "block_type": "SOURCE_EXCERPT",
            "source_role": "HERO",
            "source_start": 21.0,
            "source_end": 27.0,
            "word_start_index": 0,
            "word_end_index": 10,
            "source_text": TRANSCRIPT,
        },
        {
            "index": 1,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "semantic_intent": "explain the mechanism",
            "grounding_refs": ["block:0"],
        },
    )
    words = tuple(
        WordEvidence(index=i, text=token, start=21.0 + i * 0.4, end=21.4 + i * 0.4)
        for i, token in enumerate(TRANSCRIPT.split())
    )
    defaults: dict[str, object] = {
        "candidate_id": "cand-1",
        "candidate_key": "cand-key",
        "source_id": "src-1",
        "priority": "CANDIDATE",
        "selection_id": "sel-1",
        "selection_output_fingerprint": "sel-out",
        "selected_plan_fingerprint": "plan-fp",
        "plan_id": "plan-1",
        "plan_output_fingerprint": "plan-out",
        "strategy_type": "SOURCE_AS_EVIDENCE",
        "intensity": "MODERATE",
        "blocks": blocks,
        "hero_block_index": 0,
        "narration": {"need": "NONE", "requirements": {}},
        "verification_dependencies": (),
        "original_value_kinds": ("SYNTHESIS",),
        "required_context": ("remote work debate",),
        "preservation_constraints": ("keep hero intact",),
        "plan_target_context": {
            "target_market": "UNSPECIFIED",
            "output_language_policy": "SOURCE_LANGUAGE",
            "register_intent": "SOURCE_COMPATIBLE",
        },
        "contract_id": "contract-1",
        "contract_status": "MATERIALIZATION_REQUIRED",
        "contract_input_fingerprint": "contract-in",
        "contract_output_fingerprint": "contract-out",
        "execution_slots": (),
        "contract_source_bindings": (
            {
                "block_index": 0,
                "final_clip_start": 21.0,
                "final_clip_end": 27.0,
                "final_clip_word_start": 0,
                "final_clip_word_end": 10,
                "final_clip_text": TRANSCRIPT,
                "source_role": "HERO",
                "is_hero": True,
                "rebind_valid": True,
                "compatibility_outcome": "EXACT_MATCH",
            },
        ),
        "final_clip": FinalClipEvidence(
            refinement_id="ref-1",
            priority="FINAL_CLIP",
            quality_level="FINAL_TRANSCRIPT_READY",
            status="FINAL_TRANSCRIPT_READY",
            output_fingerprint="ref-out",
            text=TRANSCRIPT,
            words=words,
            start=21.0,
            end=27.0,
        ),
        "visual_plan": VisualPlanEvidence(
            visual_plan_id="vis-1",
            output_fingerprint="vis-out",
            plan_ready=True,
            ass_relative_path="candidates/cand-1/visual/captions.ass",
            ass_sha256="ass-sha",
            ass_policy_version="stage5.1-v1",
            caption_source_fingerprint="cap-fp",
            framing_fingerprint="frame-fp",
            source_media_fingerprint="media-fp",
        ),
        "source_media": SourceMediaEvidence(
            relative_path="sources/src-1/original.mp4",
            content_hash="media-hash",
            size_bytes=1000,
            mtime_ns=1,
            duration=60.0,
            frame_rate=Fraction(30, 1),
            display_width=1920,
            display_height=1080,
            encoded_width=1920,
            encoded_height=1080,
            sample_rate=48_000,
            channels=2,
        ),
        "profile": ProfileContext(
            profile_id="prof-1",
            profile_key="default",
            target_market="UNSPECIFIED",
            output_language_policy="SOURCE_LANGUAGE",
            register_intent="SOURCE_COMPATIBLE",
            narration_allowed=False,
        ),
    }
    defaults.update(overrides)
    return MaterializationInputs(**defaults)  # type: ignore[arg-type]


def _authored(index: int, copy: str, **overrides: object) -> AuthoredBlock:
    defaults: dict[str, object] = {
        "block_index": index,
        "block_type": "ORIGINAL_VALUE",
        "slot_id": f"block-{index}",
        "substantive_value_kind": "SYNTHESIS",
        "copy": copy,
        "adds_beyond_source": "explains the causal mechanism",
        "claim_class": "SOURCE_GROUNDED_INFERENCE",
        "grounding_refs": (GroundingRef(kind="SOURCE_EXCERPT", block_index=0),),
        "delivery": "TEXT",
    }
    defaults.update(overrides)
    return AuthoredBlock(**defaults)  # type: ignore[arg-type]


# --- fingerprints ----------------------------------------------------------


def test_input_fingerprint_is_stable_and_priority_sensitive() -> None:
    inputs = make_inputs()
    assert input_fingerprint(inputs) == input_fingerprint(make_inputs())
    assert input_fingerprint(inputs) != input_fingerprint(make_inputs(priority="FINAL_CLIP"))


def test_voice_change_does_not_invalidate_no_narration_output() -> None:
    inputs = make_inputs()
    other_voice = replace(
        inputs,
        profile=replace(inputs.profile, tts_provider="gemini", tts_model="x", tts_voice="Kore"),
    )
    assert input_fingerprint(inputs) == input_fingerprint(other_voice)
    fp = narration_fingerprint("script-fp", [], tts_identity={"voice": "a"})
    fp2 = narration_fingerprint("script-fp", [], tts_identity={"voice": "b"})
    assert fp == fp2


def test_narration_fingerprint_changes_with_voice_when_narrated() -> None:
    chunks = [{"chunk_id": "c1", "text": "hello"}]
    fp = narration_fingerprint("script-fp", chunks, tts_identity={"voice": "a"})
    fp2 = narration_fingerprint("script-fp", chunks, tts_identity={"voice": "b"})
    assert fp != fp2


def test_semantic_context_mismatch_is_detected() -> None:
    inputs = make_inputs()
    ok, mismatches = semantic_context_matches(inputs.profile, inputs.plan_target_context)
    assert ok and mismatches == ()
    profile = replace(inputs.profile, target_market="EGYPT")
    ok, mismatches = semantic_context_matches(profile, inputs.plan_target_context)
    assert not ok and mismatches == ("target_market",)


# --- scripting -------------------------------------------------------------


def test_paraphrase_only_copy_is_rejected() -> None:
    inputs = make_inputs()
    block = _authored(1, TRANSCRIPT)
    result = validate_authored_block(
        {
            "block_index": block.block_index,
            "block_type": block.block_type,
            "substantive_value_kind": block.substantive_value_kind,
            "copy": block.copy,
            "grounding_refs": [ref.as_dict() for ref in block.grounding_refs],
        },
        inputs,
        DEFAULT_CONFIG,
    )
    assert not result.accepted
    assert any("PARAPHRASE_ONLY" in finding.reason_codes for finding in result.findings)


def test_invented_number_is_rejected() -> None:
    inputs = make_inputs()
    result = validate_authored_block(
        {
            "block_index": 1,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "Remote work reduced promotion rates by 42 percent last year",
            "grounding_refs": [{"kind": "SOURCE_EXCERPT", "block_index": 0}],
        },
        inputs,
        DEFAULT_CONFIG,
    )
    assert not result.accepted
    assert any("INVENTED_NUMBER" in finding.reason_codes for finding in result.findings)


def test_presentation_only_copy_is_rejected() -> None:
    inputs = make_inputs()
    result = validate_authored_block(
        {
            "block_index": 1,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "Add captions and zoom the video",
            "grounding_refs": [{"kind": "SOURCE_EXCERPT", "block_index": 0}],
        },
        inputs,
        DEFAULT_CONFIG,
    )
    assert not result.accepted


def test_genuinely_additive_copy_requires_semantic_review() -> None:
    inputs = make_inputs()
    result = validate_authored_block(
        {
            "block_index": 1,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "The mechanism is that junior staff lost informal mentorship",
            "grounding_refs": [{"kind": "SOURCE_EXCERPT", "block_index": 0}],
        },
        inputs,
        DEFAULT_CONFIG,
    )
    assert result.requires_semantic_review
    assert not result.accepted
    assert result.claim is not None
    assert result.claim.claim_class == "SOURCE_GROUNDED_INFERENCE"
    assert result.claim.resolved is False


def test_short_word_is_not_treated_as_protected_mutation() -> None:
    inputs = make_inputs()
    result = validate_authored_block(
        {
            "block_index": 1,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "The mechanism is that mentorship loss compounds over time",
            "grounding_refs": [{"kind": "SOURCE_EXCERPT", "block_index": 0}],
        },
        inputs,
        DEFAULT_CONFIG,
    )
    assert not any("PROTECTED_TOKEN_MUTATED" in finding.reason_codes for finding in result.findings)


def test_mutated_distinctive_source_token_is_rejected() -> None:
    inputs = make_inputs()
    blocks = list(inputs.blocks)
    blocks[0] = {**blocks[0], "source_text": "The researcher cites Google Cloud adoption"}
    inputs = make_inputs(blocks=tuple(blocks))
    result = validate_authored_block(
        {
            "block_index": 1,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "Adoption of Googl Cloud accelerated the shift",
            "grounding_refs": [{"kind": "SOURCE_EXCERPT", "block_index": 0}],
        },
        inputs,
        DEFAULT_CONFIG,
    )
    assert any("PROTECTED_TOKEN_MUTATED" in finding.reason_codes for finding in result.findings)


def test_external_claim_requires_verification() -> None:
    claim = classify_claim(
        "A 2024 study shows remote work collapsed output",
        grounding_text=TRANSCRIPT,
        verification_dependency_id="claim-1",
    )
    assert claim.value == "EXTERNAL_REQUIRES_VERIFICATION"


def test_required_slot_fulfilled_and_deferred() -> None:
    inputs = make_inputs()
    fulfilled = resolve_slots(inputs, [_authored(1, "The mechanism is mentorship loss")])
    assert all_required_slots_fulfilled(fulfilled)
    deferred = resolve_slots(inputs, [])
    assert not all_required_slots_fulfilled(deferred)
    assert deferred[0].resolution == "DEFERRED"


# --- timeline --------------------------------------------------------------


def test_timeline_preserves_hero_and_orders_segments() -> None:
    inputs = make_inputs()
    authored = [_authored(1, "The mechanism is that mentorship loss compounds")]
    build = build_timeline(inputs, authored, [], config=DEFAULT_CONFIG)
    assert hero_is_protected(build.timeline, inputs)
    kinds = [segment.kind for segment in build.timeline.segments]
    assert kinds == ["SOURCE", "AUTHORED_CARD"]
    source = build.timeline.segments[0]
    assert source.is_hero
    assert source.output_duration == 6.0


def test_no_narration_makes_zero_tts_and_no_competing_overlap() -> None:
    inputs = make_inputs()
    authored = [_authored(1, "The mechanism is that mentorship loss compounds")]
    build = build_timeline(inputs, authored, [], config=DEFAULT_CONFIG)
    recipe, conflicts = build_audio_recipe(inputs, build.timeline, [], config=DEFAULT_CONFIG)
    assert conflicts == ()
    assert all(entry.kind != "NARRATION" for entry in recipe.entries)


# --- assessment ------------------------------------------------------------


def test_ready_outcome_requires_complete_favorable_evidence() -> None:
    assessment = assess_materialization(
        AssessmentContext(
            render_qc_status="PASS",
            measured_loudness_lufs=-16.0,
            measured_true_peak_dbtp=-2.0,
        ),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.READY_FOR_HUMAN_REVIEW.value
    assert assessment.human_review_required is True
    assert assessment.publication_ready is False


def test_profile_mismatch_maps_to_upstream_revalidation() -> None:
    assessment = assess_materialization(
        AssessmentContext(profile_mismatch_fields=("target_market",)),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.UPSTREAM_REVALIDATION_REQUIRED.value
    assert assessment.human_review_required is False


def test_required_narration_without_tts_is_tts_unavailable() -> None:
    assessment = assess_materialization(
        AssessmentContext(
            required_narration_required=True,
            narration_accepted=False,
            tts_available=False,
        ),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.TTS_UNAVAILABLE.value


def test_unsupported_essential_claim_blocks_verification() -> None:
    assessment = assess_materialization(
        AssessmentContext(unsupported_essential=True),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.BLOCKED_VERIFICATION.value


def test_missing_required_slot_is_no_output() -> None:
    assessment = assess_materialization(
        AssessmentContext(required_slots_fulfilled=False),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.NO_OUTPUT.value


def test_nonfinite_measurement_is_unmet_not_success() -> None:
    assessment = assess_materialization(
        AssessmentContext(render_qc_status="PASS", measured_loudness_lufs=float("nan")),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.REJECTED_QUALITY.value
    assert any(
        finding.level == QualityLevel.UNMET.value
        and "NONFINITE_MEASUREMENT" in finding.reason_codes
        for finding in assessment.findings
    )


def test_oversize_timeline_is_rejected_not_truncated() -> None:
    assessment = assess_materialization(
        AssessmentContext(), output_duration=200.0, config=DEFAULT_CONFIG
    )
    assert assessment.outcome == MaterializationOutcome.REJECTED_QUALITY.value
