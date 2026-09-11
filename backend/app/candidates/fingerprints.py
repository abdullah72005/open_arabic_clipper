"""Stable Stage 3 fingerprints and deterministic candidate identity."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from uuid import UUID

from app.candidates.policy import (
    CLASSIFIER_VERSION,
    HOOK_POLICY_VERSION,
    INPUT_FINGERPRINT_VERSION,
    NOVELTY_POLICY_VERSION,
    OUTPUT_FINGERPRINT_VERSION,
    POLICY_VERSION,
    PROPOSAL_POLICY_VERSION,
    SCORING_VERSION,
    Stage3Config,
)
from app.pipeline.fingerprints import canonical_fingerprint


def candidate_key(
    source_id: UUID | str,
    start_segment_index: int,
    end_segment_index: int,
    span_start: int | None = None,
    span_end: int | None = None,
) -> str:
    """Deterministic candidate identity from source UUID + stable coarse span.

    The span identity is the contiguous atom range within the deterministic atom
    sequence. For non-split segments atom indexes align 1:1 with segment indexes,
    and for oversized segments each bounded sub-window gets a distinct,
    reproducible atom range, so multiple windows from one transcript segment
    never collide.
    """

    start_span = start_segment_index if span_start is None else span_start
    end_span = end_segment_index if span_end is None else span_end
    raw = f"{source_id}:a{start_span}:a{end_span}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def candidate_analysis_input_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint(
        "candidate-analysis-input", INPUT_FINGERPRINT_VERSION, dict(payload)
    )


def provider_input_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint("candidate-provider-input", "1", dict(payload))


def candidate_output_fingerprint(candidates: Sequence[Mapping[str, object]]) -> str:
    """Fingerprint the complete current candidate result set in stable order."""

    ordered = sorted(candidates, key=lambda item: str(item.get("candidate_key", "")))
    return canonical_fingerprint(
        "candidate-analysis-output", OUTPUT_FINGERPRINT_VERSION, {"candidates": ordered}
    )


def stage3_policy_payload() -> dict[str, object]:
    return {
        "policy_version": POLICY_VERSION,
        "proposal_policy_version": PROPOSAL_POLICY_VERSION,
        "scoring_version": SCORING_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "hook_policy_version": HOOK_POLICY_VERSION,
        "novelty_policy_version": NOVELTY_POLICY_VERSION,
    }


def stage3_config_payload(config: Stage3Config) -> dict[str, object]:
    """Every output-affecting Stage 3 limit/threshold participates in invalidation."""

    return {
        "min_window_seconds": config.min_window_seconds,
        "preferred_window_min_seconds": config.preferred_window_min_seconds,
        "preferred_window_max_seconds": config.preferred_window_max_seconds,
        "max_window_seconds": config.max_window_seconds,
        "max_context_seconds": config.max_context_seconds,
        "max_proposals_per_hour": config.max_proposals_per_hour,
        "max_proposals_per_source": config.max_proposals_per_source,
        "max_retained_candidates": config.max_retained_candidates,
        "max_excerpt_characters": config.max_excerpt_characters,
        "retention_threshold": config.retention_threshold,
        "conflict_retention_threshold": config.conflict_retention_threshold,
        "low_transcript_confidence_threshold": config.low_transcript_confidence_threshold,
        "low_boundary_confidence_threshold": config.low_boundary_confidence_threshold,
        "low_word_probability_threshold": config.low_word_probability_threshold,
        "uncertainty_severity_threshold": config.uncertainty_severity_threshold,
        "same_source_duplicate_threshold": config.same_source_duplicate_threshold,
        "cross_source_duplicate_threshold": config.cross_source_duplicate_threshold,
        "novelty_corpus_limit": config.novelty_corpus_limit,
        "max_secondary_content_types": config.max_secondary_content_types,
        "max_provider_candidates": config.max_provider_candidates,
        "provider_candidates_per_request": config.provider_candidates_per_request,
        "max_provider_calls_per_source": config.max_provider_calls_per_source,
        "provider_max_input_characters": config.provider_max_input_characters,
        "provider_max_output_tokens": config.provider_max_output_tokens,
        "provider_context_characters": config.provider_context_characters,
        "provider_temperature": config.provider_temperature,
        "max_hooks_per_candidate": config.max_hooks_per_candidate,
        "max_hook_characters": config.max_hook_characters,
        "provenance_max_keys": config.provenance_max_keys,
        "provenance_max_key_length": config.provenance_max_key_length,
        "provenance_max_value_length": config.provenance_max_value_length,
    }


def novelty_corpus_digest(keys: Sequence[str]) -> str:
    """Stable digest of the bounded historical comparison snapshot."""

    joined = "\n".join(sorted(str(key) for key in keys))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
