"""Versioned Stage 4.1 planning policy, bounds, routing tables, and anti-slop.

Every output-affecting limit is versioned and fingersprinted. Deterministic
logic owns readiness, bounds, routing, source-span resolution, hero placement,
durations, substantive-value validation, paraphrase rejection, narration/TTS
separation, verification enforcement, material distinction, persistence
eligibility, and cache/fingerprint composition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.core.enums import (
    NarrationPurpose,
    SourceMomentStructure,
    SubstantiveValueKind,
    TransformationIntensity,
    TransformationStrategyType,
)

# v5 closure hardening: every parsed provider result, including an explicit
# no_valid_plan payload, is boundary-validated before persistence, checkpoint,
# reason, cache reuse, or handoff exposure.
POLICY_VERSION = "stage4.1-v5"
SCHEMA_VERSION = "stage4.1-schema-v5"
VALIDATION_VERSION = "stage4.1-validation-v5"
INPUT_FINGERPRINT_VERSION = "1"
OUTPUT_FINGERPRINT_VERSION = "1"
PLAN_FINGERPRINT_VERSION = "1"
PROVIDER_INPUT_FINGERPRINT_VERSION = "1"

# Official Gemini model documentation was checked 2026-09-14. See
# docs/STAGE_4_1_OPERATIONS.md for the exact links.
GEMINI_ROUTINE_MODEL = "gemini-3.5-flash-lite"
GEMINI_STRONG_MODEL = "gemini-3.8-flash"
GEMINI_API_VERSION = "v1"
ADMISSION_PRIORITY = "HIGH"
GEMINI_DOCS_CHECKED_DATE = "2026-09-14"

# Neutral production planning context defaults. There is intentionally no
# deployment-wide dialect or target-market default and no channel schema.
DEFAULT_TARGET_MARKET = "UNSPECIFIED"
DEFAULT_OUTPUT_LANGUAGE_POLICY = "SOURCE_LANGUAGE"
DEFAULT_REGISTER_INTENT = "SOURCE_COMPATIBLE"
CONTEXT_POLICY_VERSION = "stage4.1-target-context-v1"

# Semantically conservative deterministic fallback strategies: an obvious,
# grounded Stage 4.0 intent that needs no invention.
DETERMINISTIC_FALLBACK_STRATEGIES: frozenset[TransformationStrategyType] = frozenset(
    {
        TransformationStrategyType.SOURCE_LED_MINIMAL,
        TransformationStrategyType.SOURCE_AS_EVIDENCE,
    }
)

# Genuinely complex strategy work that prefers the strong hosted tier.
STRONG_ROUTE_STRATEGIES: frozenset[TransformationStrategyType] = frozenset(
    {
        TransformationStrategyType.ANALYSIS,
        TransformationStrategyType.COUNTERPOINT,
        TransformationStrategyType.COMPARISON,
        TransformationStrategyType.NEWS_CONTEXT,
        TransformationStrategyType.DEBATE_CONTEXT,
        TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION,
    }
)

# Strategies whose semantic intent is already concrete enough for a safe local
# structuring pass; used only in local_only mode.
LOCAL_STRUCTURABLE_STRATEGIES: frozenset[TransformationStrategyType] = frozenset(
    {
        *DETERMINISTIC_FALLBACK_STRATEGIES,
        TransformationStrategyType.CONTEXT_HOOK,
        TransformationStrategyType.HOOK_PLUS_TAKEAWAY,
        TransformationStrategyType.EXPLANATORY,
        TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY,
    }
)

# Original-value kinds that legitimately add a dimension the raw excerpt did
# not supply. Used by the containment guard.
_ADDITIVE_VALUE_KINDS: frozenset[SubstantiveValueKind] = frozenset(
    {
        SubstantiveValueKind.MISSING_CONTEXT,
        SubstantiveValueKind.INFERENCE,
        SubstantiveValueKind.EXPLANATION,
        SubstantiveValueKind.COMPARISON,
        SubstantiveValueKind.COUNTERPOINT,
        SubstantiveValueKind.SYNTHESIS,
        SubstantiveValueKind.AUTHORED_THESIS,
        SubstantiveValueKind.USEFUL_TAKEAWAY,
        SubstantiveValueKind.SOURCE_AS_EVIDENCE,
        SubstantiveValueKind.VERIFICATION_CORRECTION,
    }
)

# Strategy identity -> acceptable substantive value kinds for original blocks.
STRATEGY_VALUE_KINDS: dict[TransformationStrategyType, tuple[SubstantiveValueKind, ...]] = {
    TransformationStrategyType.CONTEXT_HOOK: (
        SubstantiveValueKind.MISSING_CONTEXT,
        SubstantiveValueKind.EXPLANATION,
    ),
    TransformationStrategyType.HOOK_PLUS_TAKEAWAY: (
        SubstantiveValueKind.USEFUL_TAKEAWAY,
        SubstantiveValueKind.SYNTHESIS,
        SubstantiveValueKind.MISSING_CONTEXT,
    ),
    TransformationStrategyType.EXPLANATORY: (
        SubstantiveValueKind.EXPLANATION,
        SubstantiveValueKind.SYNTHESIS,
    ),
    TransformationStrategyType.COMMENTARY: (
        SubstantiveValueKind.AUTHORED_THESIS,
        SubstantiveValueKind.COUNTERPOINT,
        SubstantiveValueKind.SYNTHESIS,
    ),
    TransformationStrategyType.ANALYSIS: (
        SubstantiveValueKind.AUTHORED_THESIS,
        SubstantiveValueKind.EXPLANATION,
        SubstantiveValueKind.SYNTHESIS,
    ),
    TransformationStrategyType.SUMMARY: (
        SubstantiveValueKind.SYNTHESIS,
        SubstantiveValueKind.USEFUL_TAKEAWAY,
    ),
    TransformationStrategyType.COMPARISON: (
        SubstantiveValueKind.COMPARISON,
        SubstantiveValueKind.SYNTHESIS,
    ),
    TransformationStrategyType.COUNTERPOINT: (SubstantiveValueKind.COUNTERPOINT,),
    TransformationStrategyType.REACTION_FRAMING: (SubstantiveValueKind.AUTHORED_THESIS,),
    TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY: (
        SubstantiveValueKind.EXPLANATION,
        SubstantiveValueKind.USEFUL_TAKEAWAY,
    ),
    TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION: (
        SubstantiveValueKind.MISSING_CONTEXT,
        SubstantiveValueKind.AUTHORED_THESIS,
        SubstantiveValueKind.SYNTHESIS,
    ),
    TransformationStrategyType.DEBATE_CONTEXT: (
        SubstantiveValueKind.MISSING_CONTEXT,
        SubstantiveValueKind.COUNTERPOINT,
    ),
    TransformationStrategyType.NEWS_CONTEXT: (
        SubstantiveValueKind.MISSING_CONTEXT,
        SubstantiveValueKind.EXPLANATION,
    ),
    TransformationStrategyType.SOURCE_AS_EVIDENCE: (
        SubstantiveValueKind.SOURCE_AS_EVIDENCE,
        SubstantiveValueKind.AUTHORED_THESIS,
    ),
    TransformationStrategyType.SOURCE_LED_MINIMAL: (
        SubstantiveValueKind.USEFUL_TAKEAWAY,
        SubstantiveValueKind.INFERENCE,
        SubstantiveValueKind.EXPLANATION,
    ),
}

# Presentation-only, fake-hook, distortion, paraphrase, and script markers.
PRESENTATION_ONLY_TERMS: tuple[str, ...] = (
    "caption",
    "captions",
    "subtitle",
    "crop",
    "reframe",
    "zoom",
    "punch in",
    "border",
    "emoji",
    "gameplay",
    "b-roll",
    "broll",
    "background loop",
    "music",
    "speed change",
    "filter",
    "watermark",
    "ترجمة",
    "تكبير",
    "إطار",
    "موسيقى",
)
FAKE_HOOK_MARKERS: tuple[str, ...] = (
    "you won't believe",
    "you wont believe",
    "لن تصدق",
    "لن تتوقع",
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
    "الجميع غاضب",
    "فضح",
)
PARAPHRASE_SCAFFOLD_MARKERS: tuple[str, ...] = (
    "basically says",
    "in other words",
    "what he means is",
    "what she means is",
    "he is saying",
    "she is saying",
    "they are saying",
    "يعني",
    "أي بمعنى",
    "ما يقوله هو",
    "ما تقصده",
)
GENERIC_VALUE_MARKERS: tuple[str, ...] = (
    "interesting",
    "insightful",
    "important to note",
    "adds value",
    "very useful",
    "مثير للاهتمام",
    "مهم جدا",
    "قيمة مضافة",
)
# Arabic/English words that only assert a verification already happened.
VERIFIED_CLAIM_MARKERS: tuple[str, ...] = (
    "verified fact",
    "it is confirmed",
    "fact checked",
    "تم التحقق",
    "مؤكد أن",
)

# Hard Stage 4.1 boundary markers. These are intentionally explicit, bounded
# phrases (not bare words) so legitimate semantic use such as "model" in
# "explain the model" is never blocked. Applied to every provider-controlled
# free-text field that can persist into a plan.
TTS_SELECTION_MARKERS: tuple[str, ...] = (
    "tts",
    "text-to-speech",
    "text to speech",
    "speech synthesis",
    "speech-synthesis",
    "synthesize speech",
    "synthesise speech",
    "voiceover",
    "voice over",
    "voice-over",
    "voice id",
    "voice_id",
    "voice model",
    "voice provider",
    "narrator voice",
    "speaker voice",
    "voice clone",
    "voice cloning",
    "elevenlabs",
    "amazon polly",
    "azure speech",
    "google tts",
    "gemini tts",
    "male voice",
    "female voice",
)
# TTS provider identities. A provider token co-occurring with a voice/narration
# context is provider-plus-voice selection, even when no explicit TTS marker is
# present (for example "Use Gemini voice Charon").
TTS_PROVIDER_TOKENS: tuple[str, ...] = (
    "gemini",
    "google",
    "openai",
    "anthropic",
    "elevenlabs",
    "polly",
    "azure",
    "amazon",
    "microsoft",
    "qwen",
    "ollama",
    "coqui",
    "speechify",
    "murf",
    "descript",
    "play.ht",
    "wellsaid",
)
# Voice/narration context cues bounded to selection contexts.
VOICE_SELECTION_CONTEXT_MARKERS: tuple[str, ...] = (
    "voice",
    "narrat",
    "tts",
    "text-to-speech",
    "text to speech",
    "synthes",
    "speech",
    "speak",
    "speaker",
    "vocal",
)
RENDERING_INSTRUCTION_MARKERS: tuple[str, ...] = (
    "ffmpeg",
    "timeline",
    "shot list",
    "storyboard",
    "keyframe",
    "key frame",
    "frame rate",
    "render at",
    "render the clip",
    "render the video",
    "render this clip",
    "cut to",
    "fade in",
    "fade out",
    "green screen",
    "b-roll",
    "broll",
    "voiceover:",
)
# Presentation-only transformation claims that masquerade as originality.
COSMETIC_EVASION_MARKERS: tuple[str, ...] = (
    "mirror the video",
    "mirroring the video",
    "mirror image",
    "horizontal flip",
    "flip the video",
    "pitch shift",
    "pitch-shift",
    "change the pitch",
    "speed up the",
    "slow down the video",
    "sped up",
    "remove watermark",
    "watermark removal",
    "obfuscate",
    "strip metadata",
)
# Platform-detection evasion tactics.
PLATFORM_EVASION_MARKERS: tuple[str, ...] = (
    "evade detection",
    "evading detection",
    "bypass detection",
    "avoid detection",
    "platform detection",
    "content id evasion",
    "avoid copyright detection",
    "copyright evasion",
    "fingerprint evasion",
    "dodge the algorithm",
    "trick the algorithm",
    "avoid the algorithm",
    "monetization loophole",
)
# Presentation/repost-only "value" that can never establish originality.
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


@dataclass(frozen=True)
class Stage41Config:
    """Hard-bounded Stage 4.1 planning limits. Every value is a safety cap."""

    max_plans: int = 3
    max_blocks_per_plan: int = 8
    max_authored_before_hero_seconds: float = 3.0
    strict_authored_before_hero_seconds: float = 1.5
    # True elapsed block duration before the hero (includes any preceding
    # source SUPPORT excerpts). The authored-material cap above remains an
    # additional protection.
    max_elapsed_before_hero_seconds: float = 3.0
    strict_elapsed_before_hero_seconds: float = 1.5
    short_moment_seconds: float = 20.0
    high_moment_density_floor: float = 0.5
    min_source_excerpt_seconds: float = 0.4
    max_plan_duration_seconds: float = 180.0
    min_substantive_intent_characters: int = 12
    containment_reject_ratio: float = 0.80
    # Raw hosted generate_content ceiling for one plan-set run.
    max_hosted_raw_calls: int = 2

    # Bounded provider input/output.
    provider_max_input_characters: int = 12_000
    provider_max_words: int = 400
    provider_max_output_tokens: int = 4_096
    provider_temperature: float = 0.0
    provider_strong_thinking_level: str = "low"
    max_provider_calls_per_plan_set: int = 2

    # Local inference bounds.
    local_max_output_tokens: int = 2_048

    def with_overrides(self, **overrides: object) -> "Stage41Config":
        return replace(self, **overrides)  # type: ignore[arg-type]


DEFAULT_CONFIG = Stage41Config()


def is_strict_hero_window(
    structure: SourceMomentStructure, duration: float, density: float
) -> bool:
    """Short, dense, joke, or payoff-first moments get the stricter hero cap."""

    if structure in {SourceMomentStructure.JOKE, SourceMomentStructure.PAYOFF}:
        return True
    if duration <= 8.0:
        return True
    return density >= 0.75


def is_complex_strategy(
    strategy_type: TransformationStrategyType,
    intensity: TransformationIntensity,
    requires_verification: bool,
) -> bool:
    """Pure deterministic strong-tier route input."""

    return (
        strategy_type in STRONG_ROUTE_STRATEGIES
        or intensity is TransformationIntensity.STRONG
        or requires_verification
    )


def strategy_value_kinds(
    strategy_type: TransformationStrategyType,
) -> tuple[SubstantiveValueKind, ...]:
    return STRATEGY_VALUE_KINDS.get(strategy_type, (SubstantiveValueKind.SYNTHESIS,))


def additive_value_kinds() -> frozenset[SubstantiveValueKind]:
    return _ADDITIVE_VALUE_KINDS


def narration_purposes() -> tuple[NarrationPurpose, ...]:
    return tuple(NarrationPurpose)


def stage41_policy_payload() -> dict[str, object]:
    return {
        "policy_version": POLICY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "validation_version": VALIDATION_VERSION,
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "gemini_docs_checked_date": GEMINI_DOCS_CHECKED_DATE,
        "strict_hero_window": True,
    }


def stage41_config_payload(config: Stage41Config) -> dict[str, object]:
    """Every output-affecting Stage 4.1 limit participates in invalidation."""

    return {
        "max_plans": config.max_plans,
        "max_blocks_per_plan": config.max_blocks_per_plan,
        "max_authored_before_hero_seconds": config.max_authored_before_hero_seconds,
        "strict_authored_before_hero_seconds": config.strict_authored_before_hero_seconds,
        "max_elapsed_before_hero_seconds": config.max_elapsed_before_hero_seconds,
        "strict_elapsed_before_hero_seconds": config.strict_elapsed_before_hero_seconds,
        "short_moment_seconds": config.short_moment_seconds,
        "high_moment_density_floor": config.high_moment_density_floor,
        "min_source_excerpt_seconds": config.min_source_excerpt_seconds,
        "max_plan_duration_seconds": config.max_plan_duration_seconds,
        "min_substantive_intent_characters": config.min_substantive_intent_characters,
        "containment_reject_ratio": config.containment_reject_ratio,
        "max_hosted_raw_calls": config.max_hosted_raw_calls,
        "provider_max_input_characters": config.provider_max_input_characters,
        "provider_max_words": config.provider_max_words,
        "provider_max_output_tokens": config.provider_max_output_tokens,
        "provider_temperature": config.provider_temperature,
        "provider_strong_thinking_level": config.provider_strong_thinking_level,
        "max_provider_calls_per_plan_set": config.max_provider_calls_per_plan_set,
        "local_max_output_tokens": config.local_max_output_tokens,
    }
