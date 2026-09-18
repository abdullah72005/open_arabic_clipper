"""Versioned Stage 5.0 render-contract policy and closed constants.

Every output-affecting constant lives here so preflight is auditable and
fingerprintable. The module is pure: no network, no provider, no model loading,
no audio decoding, no rendering. Protected semantic operator vocabularies are
deliberately re-declared here (mirroring the frozen Stage 4.2 governed
vocabulary) so Stage 5.0 can fail closed on meaning changes without touching
Stage 4.2 behavior.

Policy version ``stage5.0-v3`` and compatibility policy version
``stage5.0-compatibility-v3`` invalidate every pre-fix render contract by
changing the input fingerprint. They cover the corrective patch's behavior
changes (strict recovered-code-switch admission, typographic-apostrophe
normalization, and the now-complete fingerprint inputs) plus the sealing
patch's reachable boundary/timing classification: when wording is unchanged the
deterministic timing bands and complete-thought/window-clipping checks own the
verdict. The schema and fingerprint versions are unchanged because no persisted
JSON shape changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

RENDER_CONTRACT_POLICY_VERSION = "stage5.0-v3"
RENDER_CONTRACT_SCHEMA_VERSION = "stage5.0-schema-v1"
RENDER_CONTRACT_FINGERPRINT_VERSION = "1"
COMPATIBILITY_POLICY_VERSION = "stage5.0-compatibility-v3"
RENDER_PROFILE_VERSION = "stage5.0-render-profile-v1"

# ---------------------------------------------------------------------------
# Persisted statuses / outcomes
# ---------------------------------------------------------------------------

READY_FOR_RENDER_PLANNING = "READY_FOR_RENDER_PLANNING"
MATERIALIZATION_REQUIRED = "MATERIALIZATION_REQUIRED"
FINAL_CLIP_REFINEMENT_REQUIRED = "FINAL_CLIP_REFINEMENT_REQUIRED"
UPSTREAM_REVALIDATION_REQUIRED = "UPSTREAM_REVALIDATION_REQUIRED"
SOURCE_MEDIA_UNAVAILABLE = "SOURCE_MEDIA_UNAVAILABLE"
INVALID_SOURCE_BINDING = "INVALID_SOURCE_BINDING"
BLOCKED = "BLOCKED"

EXECUTABLE_STATUSES = frozenset({READY_FOR_RENDER_PLANNING, MATERIALIZATION_REQUIRED})
NON_EXECUTABLE_STATUSES = frozenset(
    {
        FINAL_CLIP_REFINEMENT_REQUIRED,
        UPSTREAM_REVALIDATION_REQUIRED,
        SOURCE_MEDIA_UNAVAILABLE,
        INVALID_SOURCE_BINDING,
        BLOCKED,
    }
)

EXACT_MATCH = "EXACT_MATCH"
COMPATIBLE_NON_MATERIAL_CHANGE = "COMPATIBLE_NON_MATERIAL_CHANGE"
MATERIAL_SEMANTIC_CHANGE = "MATERIAL_SEMANTIC_CHANGE"
MATERIAL_TIMING_CHANGE = "MATERIAL_TIMING_CHANGE"
SOURCE_SPAN_NO_LONGER_VALID = "SOURCE_SPAN_NO_LONGER_VALID"
UNRESOLVED_COMPATIBILITY = "UNRESOLVED_COMPATIBILITY"

# Most severe first. Used to pick the overall per-block outcome.
OUTCOME_PRECEDENCE: tuple[str, ...] = (
    SOURCE_SPAN_NO_LONGER_VALID,
    UNRESOLVED_COMPATIBILITY,
    MATERIAL_SEMANTIC_CHANGE,
    MATERIAL_TIMING_CHANGE,
    COMPATIBLE_NON_MATERIAL_CHANGE,
    EXACT_MATCH,
)

COMPATIBLE_OUTCOMES = frozenset({EXACT_MATCH, COMPATIBLE_NON_MATERIAL_CHANGE})

# ---------------------------------------------------------------------------
# Bounded tolerances
# ---------------------------------------------------------------------------

PLANNING_BOUNDS_TOLERANCE_SECONDS = 0.75
MEDIA_BOUNDS_TOLERANCE_SECONDS = 0.75
PAYOFF_COVERAGE_TOLERANCE_SECONDS = 0.75
ALIGNMENT_TIME_WINDOW_SECONDS = 8.0
MAX_ALIGNMENT_WORDS = 600
MIN_ALIGNMENT_TOKEN_COVERAGE = 0.5
COMPATIBLE_EDIT_RATIO = 0.20
UNRESOLVED_EDIT_RATIO = 0.50
NON_MATERIAL_DRIFT_SECONDS = 1.5
MATERIAL_DRIFT_SECONDS = 3.0

AUTHORED_SLOT_MATERIALIZATION_REQUIRED = True

# ---------------------------------------------------------------------------
# Protected semantic operators (English + Arabic)
# ---------------------------------------------------------------------------

NEGATION_OPERATORS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "cannot",
        "without",
        "لا",
        "لن",
        "لم",
        "ليس",
        "ليست",
        "ليسوا",
        "ما",
        "مش",
    }
)

EXCLUSIVITY_OPERATORS: frozenset[str] = frozenset(
    {
        "only",
        "solely",
        "exclusively",
        "except",
        "unless",
        "just",
        "فقط",
        "وحده",
        "وحدها",
        "حصرا",
    }
)

MODALITY_OPERATORS: frozenset[str] = frozenset(
    {
        "will",
        "would",
        "may",
        "might",
        "must",
        "should",
        "can",
        "could",
        "shall",
        "ought",
        "قد",
        "يمكن",
        "ممكن",
        "يجب",
        "لازم",
        "سوف",
        "ربما",
    }
)

PROTECTED_SEMANTIC_OPERATORS: frozenset[str] = (
    NEGATION_OPERATORS | EXCLUSIVITY_OPERATORS | MODALITY_OPERATORS
)

# Conservative contraction expansion applied before tokenizing, so ``can't``
# and ``cannot`` compare equal and an operator change is never masked by
# spelling.
CONTRACTION_EXPANSIONS: Mapping[str, str] = {
    "can't": "cannot",
    "cant": "cannot",
    "cannot": "cannot",
    "won't": "will not",
    "wont": "will not",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mustn't": "must not",
    "never": "never",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
}

# Bounded conservative filler tokens: removal/addition alone is non-material.
FILLER_TOKENS: frozenset[str] = frozenset(
    {
        "uh",
        "um",
        "erm",
        "ah",
        "eh",
        "hmm",
        "mm",
        "like",
        "basically",
        "actually",
        "literally",
        "يعني",
        "اه",
        "امم",
        "طب",
        "بس",
    }
)

# ---------------------------------------------------------------------------
# Render profile registry (code-defined, versioned)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderProfile:
    """One code-defined output profile. No codec flags, no platform deltas."""

    key: str
    semantic_version: str
    aspect_ratio: str
    width: int
    height: int
    frame_rate_policy: str
    fallback_frame_rate: float
    safe_zone_profile: str

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "semantic_version": self.semantic_version,
            "aspect_ratio": self.aspect_ratio,
            "width": self.width,
            "height": self.height,
            "frame_rate_policy": self.frame_rate_policy,
            "fallback_frame_rate": self.fallback_frame_rate,
            "safe_zone_profile": self.safe_zone_profile,
        }


SHORTS_1080X1920 = RenderProfile(
    key="SHORTS_1080X1920",
    semantic_version=RENDER_PROFILE_VERSION,
    aspect_ratio="9:16",
    width=1080,
    height=1920,
    frame_rate_policy="SOURCE_COMPATIBLE",
    fallback_frame_rate=30.0,
    safe_zone_profile="SHORTS_VERTICAL_SAFE_ZONE_V1",
)

RENDER_PROFILES: Mapping[str, RenderProfile] = {SHORTS_1080X1920.key: SHORTS_1080X1920}

DEFAULT_RENDER_PROFILE_KEY = SHORTS_1080X1920.key

# Frame-rate ceiling above which a source rate is not trusted for direct reuse.
FRAME_RATE_COMPATIBLE_MAX = 60.0

# Explicitly rejected Stage 5.0 discussion states (never persisted).
COMPATIBILITY_CHECK_REQUIRED_FLAG = "compatibility_recheck_required"

# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------

NO_SELECTED_PLAN = "NO_SELECTED_PLAN"
SELECTION_NOT_SELECTED = "SELECTION_NOT_SELECTED"
SELECTED_PLAN_NOT_FOUND = "SELECTED_PLAN_NOT_FOUND"
PLAN_NOT_CURRENT = "PLAN_NOT_CURRENT"
STALE_SELECTION_INPUT = "STALE_SELECTION_INPUT"
UPSTREAM_CHAIN_STALE = "UPSTREAM_CHAIN_STALE"
UNRESOLVED_REQUIRED_VERIFICATION = "UNRESOLVED_REQUIRED_VERIFICATION"
NO_USABLE_FINAL_CLIP_REFINEMENT = "NO_USABLE_FINAL_CLIP_REFINEMENT"

SOURCE_MEDIA_NOT_INGESTED = "SOURCE_MEDIA_NOT_INGESTED"
SOURCE_MEDIA_UNMANAGED_PATH = "SOURCE_MEDIA_UNMANAGED_PATH"
SOURCE_MEDIA_MISSING = "SOURCE_MEDIA_MISSING"
SOURCE_MEDIA_ZERO_BYTES = "SOURCE_MEDIA_ZERO_BYTES"
SOURCE_MEDIA_CORRUPT = "SOURCE_MEDIA_CORRUPT"
SOURCE_MEDIA_NO_VIDEO_STREAM = "SOURCE_MEDIA_NO_VIDEO_STREAM"
SOURCE_AUDIO_STREAM_MISSING = "SOURCE_AUDIO_STREAM_MISSING"
SOURCE_MEDIA_INVALID_DURATION = "SOURCE_MEDIA_INVALID_DURATION"
SOURCE_MEDIA_CHANGED_DURING_PREFLIGHT = "SOURCE_MEDIA_CHANGED_DURING_PREFLIGHT"

EXCERPT_OUT_OF_BOUNDS = "EXCERPT_OUT_OF_BOUNDS"
ALIGNMENT_AMBIGUOUS = "ALIGNMENT_AMBIGUOUS"
SEMANTIC_OPERATOR_CHANGED = "SEMANTIC_OPERATOR_CHANGED"
NUMBER_OR_DATE_CHANGED = "NUMBER_OR_DATE_CHANGED"
ENTITY_CHANGED = "ENTITY_CHANGED"
GROUNDING_QUOTE_LOST = "GROUNDING_QUOTE_LOST"
PAYOFF_NOT_COVERED = "PAYOFF_NOT_COVERED"
EXCERPT_CUTS_THOUGHT = "EXCERPT_CUTS_THOUGHT"
EXCERPT_CLIPPED_BY_WINDOW = "EXCERPT_CLIPPED_BY_WINDOW"
FINAL_CLIP_MEANING_CRITICAL_UNRESOLVED = "FINAL_CLIP_MEANING_CRITICAL_UNRESOLVED"
WORD_EVIDENCE_INSUFFICIENT = "WORD_EVIDENCE_INSUFFICIENT"

SPAN_OUT_OF_MEDIA_BOUNDS = "SPAN_OUT_OF_MEDIA_BOUNDS"
SPAN_REVERSED_OR_NEGATIVE = "SPAN_REVERSED_OR_NEGATIVE"

MINOR_WORDING_CHANGE = "MINOR_WORDING_CHANGE"
TIMING_DRIFT_BOUNDED = "TIMING_DRIFT_BOUNDED"
BOUNDARY_ADJUSTED = "BOUNDARY_ADJUSTED"
RECOVERED_CODE_SWITCH_TOKEN = "RECOVERED_CODE_SWITCH_TOKEN"
EXACT_IDENTITY_MATCH = "EXACT_IDENTITY_MATCH"
SOURCE_ONLY_PLAN = "SOURCE_ONLY_PLAN"
NARRATION_MATERIALIZATION_REQUIRED = "NARRATION_MATERIALIZATION_REQUIRED"
AUTHORED_TEXT_MATERIALIZATION_REQUIRED = "AUTHORED_TEXT_MATERIALIZATION_REQUIRED"

# ---------------------------------------------------------------------------
# Settings projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage50Config:
    """Bounded Stage 5.0 configuration derived from runtime Settings."""

    profile_key: str = DEFAULT_RENDER_PROFILE_KEY
    max_frame_rate: float = FRAME_RATE_COMPATIBLE_MAX
    probe_reuse_enabled: bool = True


def stage50_config_payload(config: Stage50Config) -> dict[str, object]:
    """Deterministic fingerprint payload for all output-affecting policy.

    Deliberately excludes TTS provider/model/voice, caption font/animation,
    face-tracking output, crop path, B-roll, final codec tuning, and publishing
    metadata: none of those are Stage 5.0 inputs.
    """

    return {
        "policy_version": RENDER_CONTRACT_POLICY_VERSION,
        "schema_version": RENDER_CONTRACT_SCHEMA_VERSION,
        "fingerprint_version": RENDER_CONTRACT_FINGERPRINT_VERSION,
        "compatibility_policy_version": COMPATIBILITY_POLICY_VERSION,
        "render_profile_version": RENDER_PROFILE_VERSION,
        "config": {
            "profile_key": config.profile_key,
            "max_frame_rate": config.max_frame_rate,
            "probe_reuse_enabled": config.probe_reuse_enabled,
        },
        "tolerances": {
            "planning_bounds_tolerance_seconds": PLANNING_BOUNDS_TOLERANCE_SECONDS,
            "media_bounds_tolerance_seconds": MEDIA_BOUNDS_TOLERANCE_SECONDS,
            "payoff_coverage_tolerance_seconds": PAYOFF_COVERAGE_TOLERANCE_SECONDS,
            "alignment_time_window_seconds": ALIGNMENT_TIME_WINDOW_SECONDS,
            "max_alignment_words": MAX_ALIGNMENT_WORDS,
            "min_alignment_token_coverage": MIN_ALIGNMENT_TOKEN_COVERAGE,
            "compatible_edit_ratio": COMPATIBLE_EDIT_RATIO,
            "unresolved_edit_ratio": UNRESOLVED_EDIT_RATIO,
            "non_material_drift_seconds": NON_MATERIAL_DRIFT_SECONDS,
            "material_drift_seconds": MATERIAL_DRIFT_SECONDS,
        },
        "protected_semantic_operators": {
            "negation": sorted(NEGATION_OPERATORS),
            "exclusivity": sorted(EXCLUSIVITY_OPERATORS),
            "modality": sorted(MODALITY_OPERATORS),
        },
        "contraction_expansions": dict(sorted(CONTRACTION_EXPANSIONS.items())),
        "filler_tokens": sorted(FILLER_TOKENS),
        "render_profiles": {
            key: profile.as_dict() for key, profile in sorted(RENDER_PROFILES.items())
        },
    }


def render_profile_for(key: str | None) -> RenderProfile:
    """Return the requested profile or the deterministic default."""

    if key is not None and key in RENDER_PROFILES:
        return RENDER_PROFILES[key]
    return RENDER_PROFILES[DEFAULT_RENDER_PROFILE_KEY]


def outcome_rank(outcome: str | None) -> int:
    """Index into :data:`OUTCOME_PRECEDENCE`; unknown values fail closed."""

    if outcome is None:
        return len(OUTCOME_PRECEDENCE)
    try:
        return OUTCOME_PRECEDENCE.index(outcome)
    except ValueError:
        return len(OUTCOME_PRECEDENCE)


def most_blocking_outcome(outcomes: list[str]) -> str | None:
    """Return the most severe outcome among ``outcomes`` (or None if empty)."""

    if not outcomes:
        return None
    return min(outcomes, key=outcome_rank)


__all__ = [
    "ALIGNMENT_TIME_WINDOW_SECONDS",
    "AUTHORED_SLOT_MATERIALIZATION_REQUIRED",
    "BLOCKED",
    "COMPATIBILITY_POLICY_VERSION",
    "COMPATIBLE_EDIT_RATIO",
    "COMPATIBLE_OUTCOMES",
    "CONTRACTION_EXPANSIONS",
    "DEFAULT_RENDER_PROFILE_KEY",
    "EXACT_MATCH",
    "EXCLUSIVITY_OPERATORS",
    "EXECUTABLE_STATUSES",
    "FILLER_TOKENS",
    "FINAL_CLIP_REFINEMENT_REQUIRED",
    "FRAME_RATE_COMPATIBLE_MAX",
    "INVALID_SOURCE_BINDING",
    "MATERIALIZATION_REQUIRED",
    "MATERIAL_DRIFT_SECONDS",
    "MAX_ALIGNMENT_WORDS",
    "MEDIA_BOUNDS_TOLERANCE_SECONDS",
    "MIN_ALIGNMENT_TOKEN_COVERAGE",
    "MODALITY_OPERATORS",
    "NEGATION_OPERATORS",
    "NON_EXECUTABLE_STATUSES",
    "NON_MATERIAL_DRIFT_SECONDS",
    "OUTCOME_PRECEDENCE",
    "PLANNING_BOUNDS_TOLERANCE_SECONDS",
    "PROTECTED_SEMANTIC_OPERATORS",
    "READY_FOR_RENDER_PLANNING",
    "RENDER_CONTRACT_FINGERPRINT_VERSION",
    "RENDER_CONTRACT_POLICY_VERSION",
    "RENDER_CONTRACT_SCHEMA_VERSION",
    "RENDER_PROFILES",
    "RENDER_PROFILE_VERSION",
    "SHORTS_1080X1920",
    "SOURCE_MEDIA_UNAVAILABLE",
    "Stage50Config",
    "UNRESOLVED_EDIT_RATIO",
    "UPSTREAM_REVALIDATION_REQUIRED",
    "most_blocking_outcome",
    "outcome_rank",
    "render_profile_for",
    "stage50_config_payload",
]
