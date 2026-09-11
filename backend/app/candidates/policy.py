"""Versioned Stage 3 policy constants and bounded configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace

POLICY_VERSION = "stage3-v1"
SCORING_VERSION = "stage3-scoring-v1"
PROPOSAL_POLICY_VERSION = "stage3-proposals-v1"
CLASSIFIER_VERSION = "stage3-content-v1"
HOOK_POLICY_VERSION = "stage3-hooks-v1"
NOVELTY_POLICY_VERSION = "stage3-novelty-v1"
SEMANTIC_SCHEMA_VERSION = "stage3-semantic-v1"
INPUT_FINGERPRINT_VERSION = "2"
OUTPUT_FINGERPRINT_VERSION = "2"


@dataclass(frozen=True)
class Stage3Config:
    """Hard-bounded Stage 3 limits.

    All values are safety caps, not output targets. Semantics are documented in
    ``docs/STAGE_3_OPERATIONS.md``.
    """

    # Proposal generation.
    min_window_seconds: float = 15.0
    preferred_window_min_seconds: float = 35.0
    preferred_window_max_seconds: float = 75.0
    max_window_seconds: float = 120.0
    max_context_seconds: float = 8.0
    max_proposals_per_hour: int = 24
    max_proposals_per_source: int = 240
    max_retained_candidates: int = 60
    max_excerpt_characters: int = 4_000

    # Retention policy.
    retention_threshold: float = 0.45
    conflict_retention_threshold: float = 0.40

    # Transcript uncertainty.
    low_transcript_confidence_threshold: float = 0.55
    low_boundary_confidence_threshold: float = 0.45
    low_word_probability_threshold: float = 0.72
    uncertainty_severity_threshold: float = 0.35

    # Novelty / duplicate control.
    same_source_duplicate_threshold: float = 0.80
    cross_source_duplicate_threshold: float = 0.88
    novelty_corpus_limit: int = 500
    max_secondary_content_types: int = 3

    # Semantic provider bounds.
    max_provider_candidates: int = 32
    provider_candidates_per_request: int = 8
    max_provider_calls_per_source: int = 4
    provider_max_input_characters: int = 6_000
    provider_max_output_tokens: int = 1_024
    provider_context_characters: int = 400
    provider_temperature: float = 0.0

    # Hooks.
    max_hooks_per_candidate: int = 3
    max_hook_characters: int = 240

    # Provenance metadata bounds.
    provenance_max_keys: int = 12
    provenance_max_key_length: int = 64
    provenance_max_value_length: int = 2_048

    def with_overrides(self, **overrides: object) -> "Stage3Config":
        return replace(self, **overrides)  # type: ignore[arg-type]


DEFAULT_CONFIG = Stage3Config()
