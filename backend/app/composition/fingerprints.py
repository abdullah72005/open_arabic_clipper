"""Canonical deterministic Stage 5.1 visual-composition fingerprints.

The Stage 5.1 input fingerprint covers only visual-composition-, framing-, and
caption-plan-affecting dependencies: the executable Stage 5.0 render contract,
source media identity, display geometry, the selected bound spans, the live
Stage 5.0 caption source, the output profile, safe-zone/detector identity, and
the full versioned Stage 5.1 policy payload. It deliberately excludes TTS
provider/model/voice, future narration audio/text, publishing
title/schedule/metadata, codec/encoder settings, final render artifacts, and
analytics configuration: none of those are Stage 5.1 inputs and can never
invalidate a semantic visual-composition plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.composition.policy import (
    ASS_POLICY_VERSION,
    CAPTION_LAYOUT_POLICY_VERSION,
    DETECTOR_IDENTITY,
    FINGERPRINT_VERSION,
    FRAMING_POLICY_VERSION,
    SAFE_ZONE_PROFILE_VERSION,
    Stage51Config,
    stage51_config_payload,
)
from app.pipeline.fingerprints import canonical_fingerprint


def visual_composition_input_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint("visual-composition-input", FINGERPRINT_VERSION, dict(payload))
    )


def visual_composition_output_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint("visual-composition-output", FINGERPRINT_VERSION, dict(payload))
    )


def analysis_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint("visual-composition-analysis", FINGERPRINT_VERSION, dict(payload))
    )


def framing_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint("visual-composition-framing", FINGERPRINT_VERSION, dict(payload))
    )


def caption_plan_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint("visual-composition-caption-plan", FINGERPRINT_VERSION, dict(payload))
    )


def ass_fingerprint(payload: Mapping[str, object]) -> str:
    return str(canonical_fingerprint("visual-composition-ass", FINGERPRINT_VERSION, dict(payload)))


def source_media_identity_fingerprint(identity: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "visual-composition-source-media-identity", FINGERPRINT_VERSION, dict(identity)
        )
    )


def build_stage51_input_payload(
    *,
    candidate_id: str,
    candidate_key: str,
    candidate_is_current: bool,
    disposition: str,
    analysis_fingerprint: str,
    source_id: str,
    source_media_identity: Mapping[str, object],
    contract_input_fingerprint: str,
    contract_output_fingerprint: str,
    contract_status: str,
    contract_ready: bool,
    contract_is_current: bool,
    contract_live_freshness: str,
    contract_effective: bool,
    selected_plan_id: str | None,
    selection_id: str | None,
    final_refinement_id: str | None,
    display_geometry: Mapping[str, object],
    rotation_degrees: int,
    bound_spans: Sequence[Mapping[str, object]],
    caption_source_fingerprint: str,
    caption_payload: Mapping[str, object],
    output_profile: Mapping[str, object],
    config: Stage51Config,
    detector_identity: Mapping[str, object] | None = None,
    safe_zone_key: str | None = None,
    safe_zone_version: str = SAFE_ZONE_PROFILE_VERSION,
    framing_policy_version: str = FRAMING_POLICY_VERSION,
    caption_layout_policy_version: str = CAPTION_LAYOUT_POLICY_VERSION,
    ass_policy_version: str = ASS_POLICY_VERSION,
    stage51_config: Mapping[str, object] | None = None,
    excluded_runtime_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the exact Stage 5.1 input fingerprint payload.

    ``stage51_config`` overrides the computed ``stage51_config_payload(config)``
    so callers can pin policy/test a specific version. ``excluded_runtime_context``
    is accepted only so call sites can document Stage 5.1-excluded runtime
    inputs (TTS, publishing, codec, final render artifacts, analytics); it is
    never placed in the returned payload.
    """

    resolved_config = (
        dict(stage51_config) if stage51_config is not None else stage51_config_payload(config)
    )
    resolved_detector = (
        dict(detector_identity) if detector_identity is not None else dict(DETECTOR_IDENTITY)
    )
    resolved_safe_zone_key = (
        config.safe_zone_profile_key if safe_zone_key is None else safe_zone_key
    )

    return {
        "candidate": {
            "id": candidate_id,
            "key": candidate_key,
            "is_current": candidate_is_current,
            "disposition": disposition,
            "analysis_fingerprint": analysis_fingerprint,
        },
        "source": {
            "id": source_id,
            "media_identity": dict(source_media_identity),
        },
        "contract": {
            "input_fingerprint": contract_input_fingerprint,
            "output_fingerprint": contract_output_fingerprint,
            "status": contract_status,
            "contract_ready": contract_ready,
            "is_current": contract_is_current,
            "live_freshness": contract_live_freshness,
            "effective": contract_effective,
            "selected_plan_id": selected_plan_id,
            "selection_id": selection_id,
            "final_refinement_id": final_refinement_id,
        },
        "display": {
            "geometry": dict(display_geometry),
            "rotation_degrees": rotation_degrees,
        },
        "spans": [_span_dependency(span) for span in bound_spans],
        "caption": {
            "caption_source_fingerprint": caption_source_fingerprint,
            "payload": dict(caption_payload),
        },
        "output_profile": dict(output_profile),
        "safe_zone": {"key": resolved_safe_zone_key, "version": safe_zone_version},
        "policy": {
            "framing_policy_version": framing_policy_version,
            "caption_layout_policy_version": caption_layout_policy_version,
            "ass_policy_version": ass_policy_version,
            "stage51_config": resolved_config,
        },
        "detector": resolved_detector,
    }


def _span_dependency(span: Mapping[str, object]) -> dict[str, object]:
    return {
        "block_index": span.get("block_index"),
        "start": span.get("start"),
        "end": span.get("end"),
        "word_start_index": span.get("word_start_index"),
        "word_end_index": span.get("word_end_index"),
        "source_role": span.get("source_role"),
        "is_hero": span.get("is_hero"),
    }


__all__ = [
    "analysis_fingerprint",
    "ass_fingerprint",
    "build_stage51_input_payload",
    "caption_plan_fingerprint",
    "framing_fingerprint",
    "source_media_identity_fingerprint",
    "visual_composition_input_fingerprint",
    "visual_composition_output_fingerprint",
]
