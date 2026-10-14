"""Regressions for the eight SOL-review Stage 6 defects."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_stage6_materialization_core import TRANSCRIPT, make_inputs

from app.core.enums import MaterializationOutcome
from app.materialization.acceptance import AssessmentContext, assess_materialization
from app.materialization.inputs import MaterializationInputs
from app.materialization.narration import transcription_matches, write_narration_asset
from app.materialization.policy import DEFAULT_CONFIG
from app.materialization.scripting import (
    all_required_slots_fulfilled,
    classify_claim,
    resolve_slots,
    validate_authored_block,
)
from app.materialization.supporting import (
    MANIFEST_SCHEMA_VERSION,
    SupportingAsset,
    parse_manifest,
)
from app.materialization.timeline import build_timeline, hero_is_protected
from app.materialization.types import AuthoredBlock, NarrationAudio

_SOURCE_BINDING = {
    "block_index": 0,
    "final_clip_start": 22.5,
    "final_clip_end": 28.5,
    "final_clip_word_start": 0,
    "final_clip_word_end": 10,
    "final_clip_text": TRANSCRIPT,
    "source_role": "HERO",
    "is_hero": True,
    "rebind_valid": True,
    "compatibility_outcome": "EXACT_MATCH",
}


def _inputs_with_binding(**overrides: object) -> MaterializationInputs:
    return make_inputs(contract_source_bindings=(_SOURCE_BINDING,), **overrides)


def _authored(index: int, copy: str, grounding: object = None) -> AuthoredBlock:
    return AuthoredBlock(
        block_index=index,
        block_type="ORIGINAL_VALUE",
        slot_id=f"block-{index}",
        substantive_value_kind="SYNTHESIS",
        copy=copy,
        adds_beyond_source="explains the mechanism",
        claim_class="SOURCE_GROUNDED_INFERENCE",
        grounding_refs=grounding if grounding is not None else (),  # type: ignore[arg-type]
    )


# 1. Invalid grounding references / unrelated factual copy -------------------


def test_invalid_grounding_reference_does_not_fall_back_to_transcript() -> None:
    inputs = make_inputs()
    result = validate_authored_block(
        {
            "block_index": 1,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "The mechanism is that mentorship loss compounds over time",
            "grounding_refs": [{"kind": "SOURCE_EXCERPT", "block_index": 99}],
        },
        inputs,
        DEFAULT_CONFIG,
    )
    assert not result.accepted
    assert result.claim is not None
    assert result.claim.claim_class != "SOURCE_SUPPORTED"
    assert result.claim.resolved is False


def test_unrelated_factual_copy_is_not_a_resolved_inference() -> None:
    inputs = make_inputs()
    result = validate_authored_block(
        {
            "block_index": 1,
            "block_type": "ORIGINAL_VALUE",
            "substantive_value_kind": "SYNTHESIS",
            "copy": "The president resigned after a currency crisis",
            "grounding_refs": [{"kind": "SOURCE_EXCERPT", "block_index": 0}],
        },
        inputs,
        DEFAULT_CONFIG,
    )
    assert not result.accepted
    assert result.claim is not None
    assert result.claim.resolved is False


def test_declared_grounding_without_refs_is_not_grounded() -> None:
    claim = classify_claim(
        "Some novel claim about the source",
        grounding_text="",
        verification_dependency_id=None,
    )
    assert claim.value == "EXTERNAL_REQUIRES_VERIFICATION"


# 2. Timeline uses Stage 5.0 rebound bindings --------------------------------


def test_timeline_uses_contract_binding_times_not_plan_times() -> None:
    inputs = _inputs_with_binding()
    build = build_timeline(inputs, [], [], config=DEFAULT_CONFIG)
    source = build.timeline.segments[0]
    assert source.source_start == 22.5
    assert source.source_end == 28.5


def test_timeline_requires_a_valid_binding_for_source_blocks() -> None:
    inputs = make_inputs(contract_source_bindings=())
    build = build_timeline(inputs, [], [], config=DEFAULT_CONFIG)
    assert not build.timeline.segments or build.conflicts
    assert any(conflict["reason"] == "INVALID_SOURCE_BINDING" for conflict in build.conflicts)


def test_invalid_binding_is_reported_not_silently_used() -> None:
    binding = {**_SOURCE_BINDING, "rebind_valid": False}
    inputs = make_inputs(contract_source_bindings=(binding,))
    build = build_timeline(inputs, [], [], config=DEFAULT_CONFIG)
    assert any(conflict["reason"] == "INVALID_SOURCE_BINDING" for conflict in build.conflicts)


# 3. Measured durations and fractional-duration heroes ----------------------


def test_measured_narration_duration_is_not_capped_at_eight_seconds() -> None:
    inputs = _inputs_with_binding()
    audio = NarrationAudio(
        chunk_id="narr-1",
        block_index=1,
        text="A measured narration",
        relative_path="narration/narr-1.wav",
        sha256="sha",
        size_bytes=1,
        duration_seconds=12.0,
        sample_rate=24_000,
        channels=1,
        container="wav",
    )
    build = build_timeline(
        inputs, [_authored(1, "A measured narration")], [audio], config=DEFAULT_CONFIG
    )
    card = next(seg for seg in build.timeline.segments if seg.kind == "AUTHORED_CARD")
    assert card.output_duration >= 12.0


def test_fractional_duration_hero_is_not_falsely_rejected() -> None:
    binding = {**_SOURCE_BINDING, "final_clip_start": 22.5, "final_clip_end": 28.51}
    inputs = make_inputs(
        contract_source_bindings=(binding,),
        source_media=replace(make_inputs().source_media, frame_rate=Fraction(30, 1)),
    )
    build = build_timeline(inputs, [], [], config=DEFAULT_CONFIG)
    assert hero_is_protected(build.timeline, inputs)


# 4. ASR/script matching ----------------------------------------------------


def test_asr_match_rejects_removed_negation() -> None:
    assert not transcription_matches("John did not agree", "John did agree")


def test_asr_match_rejects_changed_numbers() -> None:
    assert not transcription_matches("the rate is 42 percent", "the rate is 25 percent")


# 5. Required contract narration slots --------------------------------------


def test_required_contract_narration_slot_must_be_fulfilled() -> None:
    slots: tuple[dict[str, object], ...] = (
        {
            "slot_id": "narration-1",
            "slot_kind": "AUTHORED_NARRATION",
            "block_index": None,
            "block_type": "ORIGINAL_VALUE",
            "required": True,
            "reason_code": "REQUIRED_NARRATION",
            "payload": {},
        },
    )
    inputs = make_inputs(execution_slots=slots)
    resolutions = resolve_slots(inputs, [_authored(1, "text")])
    assert not all_required_slots_fulfilled(resolutions)
    assert any(
        item.slot_kind == "AUTHORED_NARRATION" and item.resolution != "FULFILLED"
        for item in resolutions
    )


def test_fulfilled_contract_narration_slot_passes() -> None:
    slots: tuple[dict[str, object], ...] = (
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
    inputs = make_inputs(execution_slots=slots)
    resolutions = resolve_slots(
        inputs, [_authored(1, "text")], narration_satisfied_blocks=frozenset({1})
    )
    assert all_required_slots_fulfilled(resolutions)


# 6. Assessment readiness requires render/QC/loudness evidence ---------------


def test_readiness_requires_render_qc_and_loudness_evidence() -> None:
    assessment = assess_materialization(
        AssessmentContext(), output_duration=30.0, config=DEFAULT_CONFIG
    )
    assert assessment.outcome != MaterializationOutcome.READY_FOR_HUMAN_REVIEW.value
    assert assessment.human_review_required is False


def test_provider_deferral_is_not_ready() -> None:
    assessment = assess_materialization(
        AssessmentContext(
            render_qc_status="PASS",
            measured_loudness_lufs=-16.0,
            measured_true_peak_dbtp=-2.0,
            provider_deferred=True,
        ),
        output_duration=30.0,
        config=DEFAULT_CONFIG,
    )
    assert assessment.outcome == MaterializationOutcome.DEFERRED_PROVIDER.value


def test_unknown_critical_fidelity_is_not_ready() -> None:
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
    assert assessment.outcome != MaterializationOutcome.READY_FOR_HUMAN_REVIEW.value


# 7. Supporting-asset rights require provenance -----------------------------


def test_rights_label_alone_is_not_usable() -> None:
    with pytest.raises(ValueError):
        parse_manifest(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "assets": [
                    {
                        "asset_id": "a1",
                        "relative_path": "supporting/a1.mp4",
                        "sha256": "deadbeef",
                        "rights": "EXPLICIT_COMPATIBLE",
                    }
                ],
            }
        )


def test_complete_provenance_asset_is_usable() -> None:
    asset = SupportingAsset(
        asset_id="a1",
        relative_path="supporting/a1.mp4",
        sha256="a" * 64,
        kind="ABSTRACT_LOOP",
        duration_seconds=8.0,
        width=1920,
        height=1080,
        owner="studio",
        creator="studio",
        origin="internal",
        license="CC0",
        permission="granted",
        allowed_uses=("background",),
        required_attribution="",
        restrictions=(),
        suitability_tags=("calm",),
        rights="EXPLICIT_COMPATIBLE",
    )
    assert asset.usable


def test_missing_attribution_blocks_attribution_required_asset() -> None:
    asset = SupportingAsset(
        asset_id="a1",
        relative_path="supporting/a1.mp4",
        sha256="a" * 64,
        kind="STOCK_BROLL",
        duration_seconds=8.0,
        width=1920,
        height=1080,
        owner="studio",
        creator="studio",
        origin="internal",
        license="CC-BY",
        permission="granted",
        allowed_uses=("background",),
        required_attribution="",
        restrictions=(),
        suitability_tags=("calm",),
        rights="EXPLICIT_COMPATIBLE",
        attribution_required=True,
    )
    assert not asset.usable


# 8. Narration writing must use StorageService ------------------------------


def test_narration_write_rejects_path_traversal(tmp_path: Path) -> None:
    from app.services.storage import StorageService

    storage = StorageService(tmp_path)
    with pytest.raises(Exception):
        write_narration_asset(
            storage, "00000000-0000-0000-0000-0000000000aa", "../../escape.wav", b"RIFF"
        )
