"""Versioned Stage 4.2 governor policy, bounds, platform-policy profile.

Every output-affecting limit is versioned and in the input fingerprint.
Deterministic logic owns integrity revalidation, evidence derivation, hard gates,
dimension assessment, status precedence, platform-risk interpretation, and
cache/fingerprint composition. A provider may only supply bounded observable
semantic findings; it can never assign a final governor status or platform
classification.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.transformation.planning.policy import (  # noqa: F401  (re-exported markers)
    COSMETIC_EVASION_MARKERS,
    DISTORTION_MARKERS,
    FAKE_HOOK_MARKERS,
    GENERIC_VALUE_MARKERS,
    PARAPHRASE_SCAFFOLD_MARKERS,
    PLATFORM_EVASION_MARKERS,
    PRESENTATION_ONLY_CHANGES,
    PRESENTATION_ONLY_TERMS,
    RENDERING_INSTRUCTION_MARKERS,
    TTS_PROVIDER_TOKENS,
    TTS_SELECTION_MARKERS,
    VERIFIED_CLAIM_MARKERS,
    VOICE_SELECTION_CONTEXT_MARKERS,
)

# v4: a direct-quote/reference-framed claim is grounded only when every added
# material token is harmless attribution/interpretive framing; a valid quote may
# never mask an added factual predicate, changed predicate, event, date, or
# number. v3: factual claim support is conservative — lexical overlap never
# establishes entailment; only supported quotes, near-exact restatements, or
# clearly-tied non-factual interpretations are deterministically grounded.
GOVERNOR_POLICY_VERSION = "stage4.2-v4"
GOVERNOR_SCHEMA_VERSION = "stage4.2-schema-v1"
GOVERNOR_VALIDATION_VERSION = "stage4.2-validation-v4"
INPUT_FINGERPRINT_VERSION = "1"
OUTPUT_FINGERPRINT_VERSION = "1"
PLAN_GOVERNANCE_FINGERPRINT_VERSION = "1"
PROVIDER_INPUT_FINGERPRINT_VERSION = "1"

# Official platform documentation was checked 2026-09-17. See
# docs/STAGE_4_2_OPERATIONS.md for the exact links.
PLATFORM_POLICY_PROFILE_VERSION = "stage4.2-platform-policy-2026-09-17-v1"
PLATFORM_POLICY_CHECKED_AT = "2026-09-17"

# Official Gemini model documentation was checked 2026-09-17.
GEMINI_ROUTINE_MODEL = "gemini-3.5-flash-lite"
GEMINI_STRONG_MODEL = "gemini-3.8-flash"
GEMINI_API_VERSION = "v1"
GEMINI_DOCS_CHECKED_DATE = "2026-09-17"
ADMISSION_PRIORITY = "HIGH"

ACCOUNT_LEVEL_REPETITION = "DEFERRED_TO_STAGE_7"

# Closed set of provider finding codes that may be persisted as bounded provider
# evidence. Any other value is discarded at the provider boundary so a provider
# can never smuggle arbitrary durable reason-like text into persistence.
ACCEPTED_PROVIDER_FINDING_CODES: frozenset[str] = frozenset(
    {
        "CONTEXT_DISTORTION",
        "FALSE_ATTRIBUTION",
        "SARCASM_LITERALIZED",
        "SPECULATION_AS_FACT",
        "UNRELATED_SOURCE_EVIDENCE",
        "FAKE_HOOK",
        "UNSUPPORTED_CLAIM",
        "REDUNDANT_PARAPHRASE",
        "GENERIC_FILLER",
        "SOURCE_MOMENT_INTERRUPTED",
        "NARRATION_UNNECESSARY",
        "NARRATION_POSITION_DAMAGING",
        "TEMPLATE_SHAPED",
        "COHERENCE_MIXED",
    }
)

PLATFORM_LIMITATIONS: tuple[str, ...] = (
    "Decision support only; not a legal, copyright, monetization, recommendation, "
    "or enforcement guarantee.",
    "Proprietary platform classifiers are not modeled; only observable plan "
    "characteristics are evaluated.",
)

OFFICIAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("YouTube channel monetization policies", "https://support.google.com/youtube/answer/1311392"),
    ("YouTube spam policy", "https://support.google.com/youtube/answer/2801973"),
    (
        "Meta: Rewarding Original Creators on Facebook",
        "https://about.fb.com/news/2021/07/rewarding-original-creators-on-facebook/",
    ),
    (
        "Meta: Cracking Down on Spammy Content on Facebook",
        "https://about.fb.com/news/2023/08/cracking-down-on-spammy-content-on-facebook/",
    ),
    (
        "Facebook original-content business guidance",
        "https://www.facebook.com/business/help/1136636083752902",
    ),
)

# Forbidden provider output: platform-safety guarantees and final governor
# statuses. A provider may never claim a platform outcome or assign a status.
PLATFORM_GUARANTEE_MARKERS: tuple[str, ...] = (
    "safe for youtube",
    "safe for facebook",
    "safe on youtube",
    "safe on facebook",
    "guaranteed monetizable",
    "monetization guaranteed",
    "guaranteed monetization",
    "algorithm safe",
    "safe for the algorithm",
    "will not be flagged",
    "won't be flagged",
    "will not get flagged",
    "bypasses detection",
    "bypass detection",
    "detection proof",
    "platform safe",
    "platform-approved",
    "monetization safe",
    "يوتيوب آمن",
    "آمن على يوتيوب",
    "آمن على فيسبوك",
    "مضمون الربح",
)
FORBIDDEN_STATUS_MARKERS: tuple[str, ...] = (
    "approved_for_selection",
    "approved_for_stage4_3",
    "approved for stage 4.3",
    "approved with caution",
    "blocked_pending_verification",
    "revision_required",
    "rejected_by_governor",
    "governance_deferred",
    "eligible_for_stage4_3",
)

# Deterministic pacing damage markers.
SEVERE_PREAMBLE_SECONDS = 8.0

# Presentation-only value kinds can never establish substantive originality.
ADDITIVE_VALUE_KINDS: frozenset[str] = frozenset(
    {
        "MISSING_CONTEXT",
        "INFERENCE",
        "EXPLANATION",
        "COMPARISON",
        "COUNTERPOINT",
        "SYNTHESIS",
        "AUTHORED_THESIS",
        "USEFUL_TAKEAWAY",
        "SOURCE_AS_EVIDENCE",
        "VERIFICATION_CORRECTION",
    }
)

# Strategies that intrinsically frame the source as evidence: a high source
# ratio is expected and must not be treated as low transformation.
SOURCE_AS_EVIDENCE_STRATEGIES: frozenset[str] = frozenset(
    {"SOURCE_AS_EVIDENCE", "SOURCE_LED_MINIMAL"}
)


@dataclass(frozen=True)
class Stage42Config:
    """Hard-bounded Stage 4.2 governor limits. Every value is a safety cap."""

    max_plans: int = 3

    # Deterministic retention/pacing bounds.
    max_authored_before_hero_seconds: float = 3.0
    strict_authored_before_hero_seconds: float = 1.5
    max_elapsed_before_hero_seconds: float = 3.0
    strict_elapsed_before_hero_seconds: float = 1.5
    short_moment_seconds: float = 20.0
    high_moment_density_floor: float = 0.5
    min_source_excerpt_seconds: float = 0.4
    max_plan_duration_seconds: float = 180.0
    min_substantive_intent_characters: int = 12
    containment_reject_ratio: float = 0.80
    excessive_preamble_seconds: float = 8.0
    excessive_narration_seconds: float = 12.0
    narration_share_concern: float = 0.5
    source_dominance_high_ratio: float = 0.85
    source_dominance_low_ratio: float = 0.20
    over_fragmented_source_count: int = 4
    max_switching_blocks: int = 8

    # Raw hosted generate_content ceiling for one governance run.
    max_hosted_raw_calls: int = 2

    # Bounded provider input/output.
    provider_max_input_characters: int = 12_000
    provider_max_words: int = 400
    provider_max_output_tokens: int = 3_072
    provider_temperature: float = 0.0
    provider_strong_thinking_level: str = "low"
    max_provider_calls_per_governance: int = 2

    # Local inference bounds.
    local_max_output_tokens: int = 2_048

    def with_overrides(self, **overrides: object) -> "Stage42Config":
        return replace(self, **overrides)  # type: ignore[arg-type]


DEFAULT_CONFIG = Stage42Config()

__all__ = [
    "ACCEPTED_PROVIDER_FINDING_CODES",
    "COSMETIC_EVASION_MARKERS",
    "DISTORTION_MARKERS",
    "FAKE_HOOK_MARKERS",
    "GENERIC_VALUE_MARKERS",
    "PARAPHRASE_SCAFFOLD_MARKERS",
    "PLATFORM_EVASION_MARKERS",
    "PRESENTATION_ONLY_CHANGES",
    "PRESENTATION_ONLY_TERMS",
    "RENDERING_INSTRUCTION_MARKERS",
    "TTS_PROVIDER_TOKENS",
    "TTS_SELECTION_MARKERS",
    "VERIFIED_CLAIM_MARKERS",
    "VOICE_SELECTION_CONTEXT_MARKERS",
    "DEFAULT_CONFIG",
    "Stage42Config",
    "governance_config_payload",
    "governance_policy_payload",
    "is_strict_hero_window",
    "platform_policy_payload",
]


def is_strict_hero_window(
    structure: str, duration: float, density: float, config: Stage42Config
) -> bool:
    """Short, dense, joke, or payoff-first moments get the stricter hero cap.

    The ``short_moment_seconds`` and ``high_moment_density_floor`` thresholds are
    read from ``Stage42Config`` (so they are output-affecting and part of the
    input fingerprint) rather than being hard-coded.
    """

    if structure in {"JOKE", "PAYOFF"}:
        return True
    if duration <= max(0.0, float(config.short_moment_seconds)):
        return True
    return density >= max(0.0, float(config.high_moment_density_floor))


def governance_policy_payload() -> dict[str, object]:
    return {
        "policy_version": GOVERNOR_POLICY_VERSION,
        "schema_version": GOVERNOR_SCHEMA_VERSION,
        "validation_version": GOVERNOR_VALIDATION_VERSION,
        "platform_policy_profile_version": PLATFORM_POLICY_PROFILE_VERSION,
        "platform_policy_checked_at": PLATFORM_POLICY_CHECKED_AT,
        "gemini_docs_checked_date": GEMINI_DOCS_CHECKED_DATE,
    }


def governance_config_payload(config: Stage42Config) -> dict[str, object]:
    return {
        "max_plans": config.max_plans,
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
        "excessive_preamble_seconds": config.excessive_preamble_seconds,
        "excessive_narration_seconds": config.excessive_narration_seconds,
        "narration_share_concern": config.narration_share_concern,
        "source_dominance_high_ratio": config.source_dominance_high_ratio,
        "source_dominance_low_ratio": config.source_dominance_low_ratio,
        "over_fragmented_source_count": config.over_fragmented_source_count,
        "max_switching_blocks": config.max_switching_blocks,
        "max_hosted_raw_calls": config.max_hosted_raw_calls,
        "provider_max_input_characters": config.provider_max_input_characters,
        "provider_max_words": config.provider_max_words,
        "provider_max_output_tokens": config.provider_max_output_tokens,
        "provider_temperature": config.provider_temperature,
        "provider_strong_thinking_level": config.provider_strong_thinking_level,
        "max_provider_calls_per_governance": config.max_provider_calls_per_governance,
        "local_max_output_tokens": config.local_max_output_tokens,
    }


def platform_policy_payload() -> dict[str, object]:
    """Immutable code-defined platform-policy profile (never scraped at runtime)."""

    return {
        "policy_profile_version": PLATFORM_POLICY_PROFILE_VERSION,
        "policy_checked_at": PLATFORM_POLICY_CHECKED_AT,
        "sources": [{"title": title, "url": url} for title, url in OFFICIAL_SOURCES],
        "durable_concepts": {
            "youtube": [
                "borrowed material requires significant original contribution or "
                "meaningful difference",
                "critical review, explanation, commentary and substantive editing can add value",
                "minimal changes remain reused-content risk even with permission",
                "reused-content review is separate from copyright",
                "generic, repetitive, template-like or mass-produced material carries "
                "inauthentic-content risk",
                "automated high-volume minimal variation, scraped reposting, deceptive "
                "presentation and technical detection evasion are spam/integrity risks",
                "some channel-wide review factors cannot be evaluated per plan",
            ],
            "facebook": [
                "creator-produced material is original",
                "third-party material may qualify when it presents genuinely new "
                "information, analysis or substantial storyline improvement",
                "facial reaction, stitching or narrating what is already visible without "
                "meaningful addition remains unoriginal",
                "borders, captions and speed changes are minor edits and receive no "
                "originality credit",
                "spam-network behavior, unrelated metadata, coordinated engagement and "
                "account-level flooding are outside Stage 4.2",
            ],
        },
        "account_level_repetition": ACCOUNT_LEVEL_REPETITION,
        "limitations": list(PLATFORM_LIMITATIONS),
    }
