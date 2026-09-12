"""Versioned Stage 3.5 policy constants and bounded configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace

REFINEMENT_POLICY_VERSION = "stage3.5-v1"
REFINEMENT_SCHEMA_VERSION = "stage3.5-schema-v1"
REFINEMENT_VALIDATION_VERSION = "stage3.5-validation-v1"
CONSENSUS_POLICY_VERSION = "stage3.5-consensus-v1"
BOUNDARY_POLICY_VERSION = "stage3.5-boundary-v1"
ENTITY_POLICY_VERSION = "stage3.5-entity-v1"
ROUTING_POLICY_VERSION = "stage3.5-routing-v1"
FINGERPRINT_VERSION = "1"
ADMISSION_POLICY_VERSION = "gemini-admission-v1"
HOSTED_TRANSCRIPTION_SCHEMA_VERSION = "gemini-transcription-v1"
ADJUDICATION_SCHEMA_VERSION = "gemini-adjudication-v1"


@dataclass(frozen=True)
class Stage35Config:
    """Hard-bounded Stage 3.5 limits.

    Every value is a safety cap or a quality gate, never an output target. The
    context defaults are documented in ``docs/STAGE_3_5_OPERATIONS.md``.
    """

    # Context extraction.
    candidate_pre_context_seconds: float = 5.0
    candidate_post_context_seconds: float = 5.0
    final_pre_context_seconds: float = 8.0
    final_post_context_seconds: float = 8.0
    max_refinement_window_seconds: float = 150.0
    boundary_search_radius_seconds: float = 5.0

    # Targeted local Whisper.
    candidate_beam_size: int = 5
    final_beam_size: int = 8
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    vad_filter: bool = False
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    # Acceptance gates.
    candidate_min_word_probability: float = 0.60
    candidate_min_transcript_confidence: float = 0.55
    final_min_transcript_confidence: float = 0.80
    max_edit_ratio: float = 0.60
    min_phonetic_similarity: float = 0.45
    max_inserted_words: int = 8
    repetitions_max_ratio: float = 0.40

    # Batching.
    batch_default_limit: int = 5
    batch_max_limit: int = 10

    # Bounded evidence / ambiguity.
    max_evidence_records: int = 24
    max_unresolved_spans: int = 12
    max_entity_evidence: int = 24
    max_candidate_readings: int = 4
    max_transcript_characters: int = 20_000
    max_context_characters: int = 400

    # Provider routing.
    hosted_min_uncertainty: float = 0.35
    adjudication_min_confidence: float = 0.60
    hosted_max_output_tokens: int = 2_048
    adjudication_max_output_tokens: int = 1_024

    def with_overrides(self, **overrides: object) -> "Stage35Config":
        return replace(self, **overrides)  # type: ignore[arg-type]


DEFAULT_CONFIG = Stage35Config()

VALID_PRIORITIES = ("CANDIDATE", "FINAL_CLIP")


@dataclass(frozen=True)
class AdmissionPolicy:
    """Static, configurable Gemini admission limits.

    The policy is static identity (part of runtime fingerprints); transient
    counters and cooldowns are deliberately not.
    """

    window_seconds: float = 60.0
    total_calls: int = 30
    critical_reserve: int = 8
    high_reserve: int = 6
    low_enabled: bool = False
    provider_cooldown_seconds: float = 60.0
    max_retry_after_seconds: float = 3_600.0

    def __post_init__(self) -> None:
        if self.total_calls <= 0:
            raise ValueError("admission total_calls must be positive")
        if self.critical_reserve < 0 or self.high_reserve < 0:
            raise ValueError("admission reserves must be non-negative")
        if self.critical_reserve + self.high_reserve > self.total_calls:
            raise ValueError("admission reserves cannot exceed total capacity")

    def as_dict(self) -> dict[str, object]:
        return {
            "window_seconds": self.window_seconds,
            "total_calls": self.total_calls,
            "critical_reserve": self.critical_reserve,
            "high_reserve": self.high_reserve,
            "low_enabled": self.low_enabled,
            "provider_cooldown_seconds": self.provider_cooldown_seconds,
            "max_retry_after_seconds": self.max_retry_after_seconds,
        }
