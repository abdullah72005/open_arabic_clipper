"""Stable Stage 6 fingerprints.

Fingerprints exclude secrets, transient quotas, attempt counters, absolute
deployment paths, and timestamps. They include the full route-relevant identity
so any semantic/profile/asset/timeline/render/policy change invalidates prior
work, while a voice-only change never invalidates a no-narration script.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.materialization.inputs import MaterializationInputs
from app.materialization.policy import (
    FINGERPRINT_VERSION,
    STAGE6_POLICY_VERSION,
    STAGE6_SCHEMA_VERSION,
    STAGE6_VALIDATION_VERSION,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(inputs: MaterializationInputs) -> str:
    """Everything that changes the executable Stage 6 route/scope."""

    return stable_digest(
        {
            "fingerprint_version": FINGERPRINT_VERSION,
            "policy_version": STAGE6_POLICY_VERSION,
            "schema_version": STAGE6_SCHEMA_VERSION,
            "validation_version": STAGE6_VALIDATION_VERSION,
            "candidate_id": inputs.candidate_id,
            "source_id": inputs.source_id,
            "priority": inputs.priority,
            "selection_id": inputs.selection_id,
            "selection_output_fingerprint": inputs.selection_output_fingerprint,
            "selected_plan_fingerprint": inputs.selected_plan_fingerprint,
            "plan_id": inputs.plan_id,
            "plan_output_fingerprint": inputs.plan_output_fingerprint,
            "strategy_type": inputs.strategy_type,
            "intensity": inputs.intensity,
            "blocks": [dict(block) for block in inputs.blocks],
            "hero_block_index": inputs.hero_block_index,
            "narration": dict(inputs.narration),
            "verification_dependencies": [dict(item) for item in inputs.verification_dependencies],
            "original_value_kinds": list(inputs.original_value_kinds),
            "contract_id": inputs.contract_id,
            "contract_status": inputs.contract_status,
            "contract_input_fingerprint": inputs.contract_input_fingerprint,
            "contract_output_fingerprint": inputs.contract_output_fingerprint,
            "execution_slots": [dict(item) for item in inputs.execution_slots],
            "contract_source_bindings": [dict(item) for item in inputs.contract_source_bindings],
            "final_clip": inputs.final_clip.as_dict(),
            "visual_plan": inputs.visual_plan.as_dict(),
            "source_media": inputs.source_media.as_dict(),
            "profile_semantic_context": {
                "target_market": inputs.profile.target_market,
                "output_language_policy": inputs.profile.output_language_policy,
                "register_intent": inputs.profile.register_intent,
                "narration_allowed": inputs.profile.narration_allowed,
            },
            "plan_target_context": dict(inputs.plan_target_context),
            "source_dialect": dict(inputs.source_dialect),
            "source_language": inputs.source_language,
            "operator_authored_copy": [dict(item) for item in inputs.operator_authored_copy],
            "supporting_requested": inputs.supporting_requested,
        }
    )


def script_fingerprint(
    inputs: MaterializationInputs,
    accepted_blocks: Sequence[Mapping[str, object]],
    *,
    prompt_hash: str,
    provider_identity: Mapping[str, object],
) -> str:
    """Script identity: semantic requirements, evidence, profile, provider."""

    return stable_digest(
        {
            "validation_version": STAGE6_VALIDATION_VERSION,
            "plan_output_fingerprint": inputs.plan_output_fingerprint,
            "final_clip_output_fingerprint": inputs.final_clip.output_fingerprint,
            "plan_target_context": dict(inputs.plan_target_context),
            "profile_semantic_context": {
                "target_market": inputs.profile.target_market,
                "output_language_policy": inputs.profile.output_language_policy,
                "register_intent": inputs.profile.register_intent,
            },
            "narration": dict(inputs.narration),
            "blocks": [dict(block) for block in accepted_blocks],
            "operator_authored_copy": [dict(item) for item in inputs.operator_authored_copy],
            "prompt_hash": prompt_hash,
            "provider_identity": dict(provider_identity),
        }
    )


def narration_fingerprint(
    script_fp: str,
    accepted_chunks: Sequence[Mapping[str, object]],
    *,
    tts_identity: Mapping[str, object],
) -> str:
    """Narration identity: exact accepted script plus synthesis identity.

    A no-narration materialization has no accepted chunks, so an unused voice
    change never invalidates it.
    """

    if not accepted_chunks:
        return stable_digest({"script_fingerprint": script_fp, "narration": "NONE"})
    return stable_digest(
        {
            "script_fingerprint": script_fp,
            "chunks": [dict(chunk) for chunk in accepted_chunks],
            "tts_identity": dict(tts_identity),
        }
    )


def supporting_fingerprint(assets: Sequence[Mapping[str, object]]) -> str:
    """Supporting-asset identity: actual bytes/hash plus rights/provenance."""

    return stable_digest({"assets": [dict(asset) for asset in assets]})


def timeline_fingerprint(
    inputs: MaterializationInputs,
    accepted_blocks: Sequence[Mapping[str, object]],
    audio_recipe: Mapping[str, object],
    *,
    segment_identity: Sequence[Mapping[str, object]],
) -> str:
    """Timeline identity: content, bound source map, measured audio, layout."""

    return stable_digest(
        {
            "timeline_schema_version": "stage6-timeline-v1",
            "plan_output_fingerprint": inputs.plan_output_fingerprint,
            "final_clip_output_fingerprint": inputs.final_clip.output_fingerprint,
            "visual_plan_output_fingerprint": inputs.visual_plan.output_fingerprint,
            "source_media": inputs.source_media.as_dict(),
            "accepted_blocks": [dict(block) for block in accepted_blocks],
            "segments": [dict(segment) for segment in segment_identity],
            "audio_recipe": dict(audio_recipe),
            "preservation_constraints": list(inputs.preservation_constraints),
        }
    )


def render_fingerprint(
    timeline_fp: str,
    *,
    ass_sha256: str,
    asset_hashes: Sequence[str],
    mix_recipe: Mapping[str, object],
    delivery_profile_key: str,
) -> str:
    """Render identity: frozen timeline/assets/ASS/mix plus delivery/runtime."""

    return stable_digest(
        {
            "timeline_fingerprint": timeline_fp,
            "ass_sha256": ass_sha256,
            "asset_hashes": list(asset_hashes),
            "mix_recipe": dict(mix_recipe),
            "delivery_profile_key": delivery_profile_key,
        }
    )


def assessment_fingerprint(
    materialization_output_fp: str,
    render_output_fp: str,
    *,
    policy_version: str,
) -> str:
    """Assessment identity: exact materialization/render plus policy."""

    return stable_digest(
        {
            "materialization_output_fingerprint": materialization_output_fp,
            "render_output_fingerprint": render_output_fp,
            "assessment_policy_version": policy_version,
        }
    )


def output_fingerprint(
    inputs: MaterializationInputs,
    *,
    script_fp: str,
    narration_fp: str,
    supporting_fp: str,
    timeline_fp: str,
) -> str:
    """The materialization's own output identity (excludes render)."""

    return stable_digest(
        {
            "input_fingerprint": input_fingerprint(inputs),
            "priority": inputs.priority,
            "script_fingerprint": script_fp,
            "narration_fingerprint": narration_fp,
            "supporting_fingerprint": supporting_fp,
            "timeline_fingerprint": timeline_fp,
        }
    )


__all__ = [
    "assessment_fingerprint",
    "canonical_json",
    "hash_bytes",
    "hash_file",
    "input_fingerprint",
    "narration_fingerprint",
    "output_fingerprint",
    "render_fingerprint",
    "script_fingerprint",
    "stable_digest",
    "supporting_fingerprint",
    "timeline_fingerprint",
]
