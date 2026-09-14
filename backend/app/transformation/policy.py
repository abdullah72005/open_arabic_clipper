"""Versioned Stage 4.0 policy constants, bounded config, and suitability tables.

Deterministic logic owns prerequisites, transcript/context sufficiency,
transformation necessity, source-moment structure, content-to-strategy
suitability, presentation-only zero credit, hard gates, validation, final
eligibility, fallback, and ranking. Optional providers may only assess
directions that survived those deterministic gates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.core.enums import ContentType, SourceMomentStructure, TransformationStrategyType

POLICY_VERSION = "stage4.0-v1"
SCHEMA_VERSION = "stage4.0-schema-v1"
VALIDATION_VERSION = "stage4.0-validation-v1"
ELIGIBILITY_POLICY_VERSION = "stage4.0-eligibility-v1"
STRATEGY_POLICY_VERSION = "stage4.0-strategy-v1"
PLATFORM_RISK_POLICY_VERSION = "stage4.0-platform-risk-v1"
INPUT_FINGERPRINT_VERSION = "1"
OUTPUT_FINGERPRINT_VERSION = "1"

# Official facts verified 2026-09-14 (see docs/STAGE_4_0_OPERATIONS.md).
GEMINI_ROUTINE_MODEL = "gemini-3.5-flash-lite"
GEMINI_STRONG_MODEL = "gemini-3.8-flash"
GEMINI_API_VERSION = "v1"
ADMISSION_PRIORITY = "HIGH"
PLATFORM_POLICY_CHECKED_DATE = "2026-09-14"

# Official references (durable principles only, never hidden-classifier guesses).
PLATFORM_POLICY_REFERENCES: tuple[str, ...] = (
    "YouTube channel monetization policies (reused content; inauthentic/repetitive content)",
    "YouTube spam, deceptive practices, and scams policies",
    "Meta: Rewarding Original Creators on Facebook",
    "Meta: Combating unoriginal content / Original Content Guidelines",
)


@dataclass(frozen=True)
class Stage40Config:
    """Hard-bounded Stage 4.0 limits. Every value is a safety cap or gate."""

    # Output bounds.
    max_recommended_strategies: int = 3
    max_rejected_strategies: int = 3
    max_provider_candidates: int = 5

    # Context bounds.
    max_context_segments: int = 6
    max_context_characters: int = 1_200
    min_context_characters: int = 24
    min_transcript_characters: int = 16
    min_transcript_confidence: float = 0.55
    min_moment_duration_seconds: float = 3.0
    short_moment_seconds: float = 20.0
    high_moment_density_floor: float = 0.5

    # Hard gates.
    min_added_value_density: float = 0.35
    max_source_moment_damage: float = 0.65
    max_generic_filler: float = 0.55
    max_redundant_commentary: float = 0.60
    min_originality_potential: float = 0.40
    third_party_min_originality_potential: float = 0.60
    max_template_staleness: float = 0.72
    short_moment_setup_damage_risk: float = 0.70

    # Provider bounds.
    provider_max_input_characters: int = 8_000
    provider_max_output_tokens: int = 2_048
    provider_temperature: float = 0.0
    provider_strong_thinking_level: str = "low"
    max_provider_calls_per_analysis: int = 1

    def with_overrides(self, **overrides: object) -> "Stage40Config":
        return replace(self, **overrides)  # type: ignore[arg-type]


DEFAULT_CONFIG = Stage40Config()

# Presentation-only changes are never substantive; they receive zero credit and
# can never be the basis of a recommended strategy.
PRESENTATION_ONLY_CHANGES: frozenset[str] = frozenset(
    {
        "captions",
        "crop",
        "reframe",
        "zoom",
        "punch_in",
        "border",
        "emoji",
        "gameplay",
        "broll",
        "b_roll",
        "background_loop",
        "music",
        "speed_change",
        "simple_cut",
        "filter",
        "watermark",
    }
)

# Substantive-value kinds that never qualify on their own without a specific
# value focus (guards against "commentary"/"summary" passing on name alone).
NAME_ONLY_STRATEGIES: frozenset[TransformationStrategyType] = frozenset(
    {
        TransformationStrategyType.SUMMARY,
        TransformationStrategyType.COMMENTARY,
        TransformationStrategyType.REACTION_FRAMING,
    }
)

# Deterministic content-suitability mapping. Order is preference order; the
# first entry is the least intrusive natural direction for that content type.
CONTENT_SUITABILITY: dict[ContentType, tuple[TransformationStrategyType, ...]] = {
    ContentType.INTERVIEW_INSIGHT: (
        TransformationStrategyType.SOURCE_AS_EVIDENCE,
        TransformationStrategyType.ANALYSIS,
        TransformationStrategyType.COUNTERPOINT,
        TransformationStrategyType.CONTEXT_HOOK,
    ),
    ContentType.CONTROVERSIAL_OPINION: (
        TransformationStrategyType.COUNTERPOINT,
        TransformationStrategyType.SOURCE_AS_EVIDENCE,
        TransformationStrategyType.ANALYSIS,
        TransformationStrategyType.CONTEXT_HOOK,
    ),
    ContentType.ANALYSIS: (
        TransformationStrategyType.ANALYSIS,
        TransformationStrategyType.SOURCE_AS_EVIDENCE,
        TransformationStrategyType.EXPLANATORY,
    ),
    ContentType.EDUCATIONAL: (
        TransformationStrategyType.EXPLANATORY,
        TransformationStrategyType.COMPARISON,
        TransformationStrategyType.HOOK_PLUS_TAKEAWAY,
    ),
    ContentType.TUTORIAL: (
        TransformationStrategyType.EXPLANATORY,
        TransformationStrategyType.HOOK_PLUS_TAKEAWAY,
        TransformationStrategyType.COMPARISON,
    ),
    # Funny is preservation-first: only concise context hook, genuine reaction
    # framing, or substantive source-led minimal; otherwise no strategy.
    ContentType.FUNNY: (
        TransformationStrategyType.SOURCE_LED_MINIMAL,
        TransformationStrategyType.CONTEXT_HOOK,
        TransformationStrategyType.REACTION_FRAMING,
    ),
    ContentType.STORY: (
        TransformationStrategyType.CONTEXT_HOOK,
        TransformationStrategyType.HOOK_PLUS_TAKEAWAY,
        TransformationStrategyType.SOURCE_LED_MINIMAL,
    ),
    ContentType.EMOTIONAL: (
        TransformationStrategyType.CONTEXT_HOOK,
        TransformationStrategyType.HOOK_PLUS_TAKEAWAY,
        TransformationStrategyType.SOURCE_LED_MINIMAL,
    ),
    ContentType.DEBATE: (
        TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION,
        TransformationStrategyType.COUNTERPOINT,
        TransformationStrategyType.DEBATE_CONTEXT,
    ),
    ContentType.NEWS_CURRENT_EVENT: (
        TransformationStrategyType.NEWS_CONTEXT,
        TransformationStrategyType.SOURCE_AS_EVIDENCE,
        TransformationStrategyType.EXPLANATORY,
    ),
    ContentType.SURPRISING_FACT: (
        TransformationStrategyType.CONTEXT_HOOK,
        TransformationStrategyType.HOOK_PLUS_TAKEAWAY,
        TransformationStrategyType.EXPLANATORY,
    ),
    ContentType.REACTION_WORTHY: (
        TransformationStrategyType.REACTION_FRAMING,
        TransformationStrategyType.ANALYSIS,
        TransformationStrategyType.CONTEXT_HOOK,
    ),
    ContentType.MOTIVATIONAL: (
        TransformationStrategyType.HOOK_PLUS_TAKEAWAY,
        TransformationStrategyType.SOURCE_LED_MINIMAL,
        TransformationStrategyType.CONTEXT_HOOK,
    ),
    ContentType.OTHER: (
        TransformationStrategyType.SOURCE_AS_EVIDENCE,
        TransformationStrategyType.CONTEXT_HOOK,
        TransformationStrategyType.ANALYSIS,
        TransformationStrategyType.SOURCE_LED_MINIMAL,
        TransformationStrategyType.EXPLANATORY,
    ),
}

# Default minimal sufficient intensity per content type when not otherwise
# justified. Intentionally biased toward the least intrusive sufficient option.
DEFAULT_INTENSITY: dict[ContentType, str] = {
    ContentType.EDUCATIONAL: "MODERATE",
    ContentType.TUTORIAL: "MODERATE",
    ContentType.NEWS_CURRENT_EVENT: "MODERATE",
    ContentType.DEBATE: "MODERATE",
    ContentType.ANALYSIS: "MODERATE",
    ContentType.CONTROVERSIAL_OPINION: "MODERATE",
    ContentType.INTERVIEW_INSIGHT: "MINIMAL",
    ContentType.FUNNY: "MINIMAL",
    ContentType.STORY: "MINIMAL",
    ContentType.EMOTIONAL: "MINIMAL",
    ContentType.REACTION_WORTHY: "MODERATE",
    ContentType.MOTIVATIONAL: "MINIMAL",
    ContentType.SURPRISING_FACT: "MINIMAL",
    ContentType.OTHER: "MINIMAL",
}

# Strategy identity -> substantive value it can legitimately provide. Used by
# hard validation to reject directions whose only value is cosmetic.
STRATEGY_VALUE_KINDS: dict[TransformationStrategyType, tuple[str, ...]] = {
    TransformationStrategyType.CONTEXT_HOOK: (
        "MISSING_CONTEXT",
        "EXPLANATION",
    ),
    TransformationStrategyType.HOOK_PLUS_TAKEAWAY: (
        "USEFUL_TAKEAWAY",
        "SYNTHESIS",
        "MISSING_CONTEXT",
    ),
    TransformationStrategyType.EXPLANATORY: ("EXPLANATION", "SYNTHESIS"),
    TransformationStrategyType.COMMENTARY: ("AUTHORED_THESIS", "COUNTERPOINT", "SYNTHESIS"),
    TransformationStrategyType.ANALYSIS: ("AUTHORED_THESIS", "EXPLANATION", "SYNTHESIS"),
    TransformationStrategyType.SUMMARY: ("SYNTHESIS", "USEFUL_TAKEAWAY"),
    TransformationStrategyType.COMPARISON: ("COMPARISON", "SYNTHESIS"),
    TransformationStrategyType.COUNTERPOINT: ("COUNTERPOINT",),
    TransformationStrategyType.REACTION_FRAMING: ("AUTHORED_THESIS",),
    TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY: (
        "EXPLANATION",
        "USEFUL_TAKEAWAY",
    ),
    TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION: (
        "MISSING_CONTEXT",
        "AUTHORED_THESIS",
        "SYNTHESIS",
    ),
    TransformationStrategyType.DEBATE_CONTEXT: ("MISSING_CONTEXT", "COUNTERPOINT"),
    TransformationStrategyType.NEWS_CONTEXT: ("MISSING_CONTEXT", "EXPLANATION"),
    TransformationStrategyType.SOURCE_AS_EVIDENCE: ("SOURCE_AS_EVIDENCE", "AUTHORED_THESIS"),
    TransformationStrategyType.SOURCE_LED_MINIMAL: ("USEFUL_TAKEAWAY", "INFERENCE"),
}

# Unsupported/fake-hook and slop markers (deterministic, small, testable).
FAKE_HOOK_MARKERS: tuple[str, ...] = (
    "you won't believe",
    "you wont believe",
    "لن تصدق",
    "لن تتوقع",
    "لن تصدق ما حدث",
    "shocking truth",
    "the truth they hide",
    "must watch",
    "لا تفوت",
)
DISTORTION_MARKERS: tuple[str, ...] = (
    "everyone is furious",
    "destroyed",
    "utterly humiliated",
    "exposed as a fraud",
    "the whole world is watching",
)
PARAPHRASE_MARKERS: tuple[str, ...] = (
    "basically says",
    "in other words",
    "what he means is",
    "what she means is",
)
SCRIPT_SHAPE_MARKERS: tuple[str, ...] = (
    "0:00",
    "00:00",
    "shot 1",
    "scene 1",
    "voiceover:",
    "narration:",
    "tts:",
    "render at",
    "fade in",
    "cut to",
)


def stage40_policy_payload() -> dict[str, object]:
    return {
        "policy_version": POLICY_VERSION,
        "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
        "strategy_policy_version": STRATEGY_POLICY_VERSION,
        "platform_risk_policy_version": PLATFORM_RISK_POLICY_VERSION,
        "platform_policy_checked_date": PLATFORM_POLICY_CHECKED_DATE,
    }


def stage40_config_payload(config: Stage40Config) -> dict[str, object]:
    """Every output-affecting Stage 4.0 limit/threshold participates in invalidation."""

    return {
        "max_recommended_strategies": config.max_recommended_strategies,
        "max_rejected_strategies": config.max_rejected_strategies,
        "max_provider_candidates": config.max_provider_candidates,
        "max_context_segments": config.max_context_segments,
        "max_context_characters": config.max_context_characters,
        "min_context_characters": config.min_context_characters,
        "min_transcript_characters": config.min_transcript_characters,
        "min_transcript_confidence": config.min_transcript_confidence,
        "min_moment_duration_seconds": config.min_moment_duration_seconds,
        "short_moment_seconds": config.short_moment_seconds,
        "high_moment_density_floor": config.high_moment_density_floor,
        "min_added_value_density": config.min_added_value_density,
        "max_source_moment_damage": config.max_source_moment_damage,
        "max_generic_filler": config.max_generic_filler,
        "max_redundant_commentary": config.max_redundant_commentary,
        "min_originality_potential": config.min_originality_potential,
        "third_party_min_originality_potential": config.third_party_min_originality_potential,
        "max_template_staleness": config.max_template_staleness,
        "short_moment_setup_damage_risk": config.short_moment_setup_damage_risk,
        "provider_max_input_characters": config.provider_max_input_characters,
        "provider_max_output_tokens": config.provider_max_output_tokens,
        "provider_temperature": config.provider_temperature,
        "provider_strong_thinking_level": config.provider_strong_thinking_level,
        "max_provider_calls_per_analysis": config.max_provider_calls_per_analysis,
    }


def is_structure_complex(structure: SourceMomentStructure) -> bool:
    """Pure deterministic router input for the stronger provider tier."""

    return structure in {
        SourceMomentStructure.CLAIM,
        SourceMomentStructure.DEBATE,
        SourceMomentStructure.NEWS,
    }
