"""Canonical deterministic Stage 5.0 fingerprints.

The contract input fingerprint covers every preflight-relevant dependency but
deliberately excludes TTS provider/model/voice, generated narration audio,
caption font/animation, face-tracking output, crop path, B-roll, final codec
tuning, publishing metadata, and any final render artifact hash. Those are not
Stage 5.0 inputs and can never invalidate a semantic render contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.pipeline.fingerprints import canonical_fingerprint
from app.render.policy import (
    RENDER_CONTRACT_FINGERPRINT_VERSION,
    RENDER_CONTRACT_POLICY_VERSION,
    RENDER_CONTRACT_SCHEMA_VERSION,
)


def render_contract_input_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "render-contract-input", RENDER_CONTRACT_FINGERPRINT_VERSION, dict(payload)
        )
    )


def render_contract_output_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "render-contract-output", RENDER_CONTRACT_FINGERPRINT_VERSION, dict(payload)
        )
    )


def caption_source_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "render-caption-source", RENDER_CONTRACT_FINGERPRINT_VERSION, dict(payload)
        )
    )


def source_media_identity_fingerprint(identity: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "render-source-media-identity",
            RENDER_CONTRACT_FINGERPRINT_VERSION,
            dict(identity),
        )
    )


def probe_fingerprint(facts: Mapping[str, object], identity: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "render-source-probe",
            RENDER_CONTRACT_FINGERPRINT_VERSION,
            {"facts": dict(facts), "identity": dict(identity)},
        )
    )


def build_render_contract_input_payload(
    *,
    candidate_id: str,
    candidate_key: str,
    candidate_is_current: bool,
    disposition: str,
    analysis_fingerprint: str,
    source_id: str,
    source_media_identity: Mapping[str, object],
    selection_id: str | None,
    selection_status: str | None,
    selection_with_caution: bool,
    selection_input_fingerprint: str,
    selection_output_fingerprint: str,
    governance_input_fingerprint: str,
    governance_output_fingerprint: str,
    governor_policy_version: str,
    governor_validation_version: str,
    platform_policy_profile_version: str,
    selected_plan_id: str | None,
    selected_plan_fingerprint: str,
    selected_plan_output_fingerprint: str,
    selected_plan_is_current: bool,
    planning_refinement_id: str | None,
    planning_refinement_priority: str,
    planning_refinement_quality_level: str,
    planning_refinement_output_fingerprint: str,
    final_refinement_id: str,
    final_refinement_status: str,
    final_refinement_quality_level: str,
    final_refinement_output_fingerprint: str,
    live_caption_source_fingerprint: str,
    verification_state: str,
    verification_unresolved: bool,
    profile_key: str,
    profile_version: str,
    stage50_config: Mapping[str, object],
) -> dict[str, object]:
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
        "selection": {
            "id": selection_id,
            "status": selection_status,
            "selected_with_caution": selection_with_caution,
            "input_fingerprint": selection_input_fingerprint,
            "output_fingerprint": selection_output_fingerprint,
            "governance_input_fingerprint": governance_input_fingerprint,
            "governance_output_fingerprint": governance_output_fingerprint,
            "governor_policy_version": governor_policy_version,
            "governor_validation_version": governor_validation_version,
            "platform_policy_profile_version": platform_policy_profile_version,
        },
        "selected_plan": {
            "id": selected_plan_id,
            "selected_plan_fingerprint": selected_plan_fingerprint,
            "plan_output_fingerprint": selected_plan_output_fingerprint,
            "is_current": selected_plan_is_current,
        },
        "planning_refinement": {
            "id": planning_refinement_id,
            "priority": planning_refinement_priority,
            "quality_level": planning_refinement_quality_level,
            "output_fingerprint": planning_refinement_output_fingerprint,
        },
        "final_refinement": {
            "id": final_refinement_id,
            "status": final_refinement_status,
            "quality_level": final_refinement_quality_level,
            "output_fingerprint": final_refinement_output_fingerprint,
            "caption_source_fingerprint": live_caption_source_fingerprint,
        },
        "verification": {
            "state": verification_state,
            "unresolved": verification_unresolved,
        },
        "policy": {
            "profile_key": profile_key,
            "profile_version": profile_version,
            "policy_version": RENDER_CONTRACT_POLICY_VERSION,
            "schema_version": RENDER_CONTRACT_SCHEMA_VERSION,
            "fingerprint_version": RENDER_CONTRACT_FINGERPRINT_VERSION,
            "config": dict(stage50_config),
        },
    }


def build_caption_source_payload(
    *,
    final_transcript: str,
    word_timestamps: Sequence[Mapping[str, object]],
    dialect_profile: str | None,
    dialect_confidence: float,
    code_switch_evidence: Mapping[str, object],
    final_refinement_output_fingerprint: str,
) -> dict[str, object]:
    return {
        "final_transcript": final_transcript,
        "word_timestamps": [dict(word) for word in word_timestamps],
        "dialect_profile": dialect_profile,
        "dialect_confidence": dialect_confidence,
        "code_switch_evidence": dict(code_switch_evidence),
        "final_refinement_output_fingerprint": final_refinement_output_fingerprint,
    }


__all__ = [
    "build_caption_source_payload",
    "build_render_contract_input_payload",
    "caption_source_fingerprint",
    "probe_fingerprint",
    "render_contract_input_fingerprint",
    "render_contract_output_fingerprint",
    "source_media_identity_fingerprint",
]
