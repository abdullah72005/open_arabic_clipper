"""Deterministic Stage 4.0 eligibility gates, source-moment structure, and risk.

Pure functions only: no network, no model loading, no audio decoding. These
functions own prerequisites, transcript/context sufficiency, transformation
necessity, obvious low-value cases, source-moment structure, hard blockers, and
the platform-risk decision-support snapshot.
"""

from __future__ import annotations

from app.core.enums import (
    ContentType,
    MediaOriginType,
    OriginalityRisk,
    PlatformRiskKind,
    PlatformRiskLevel,
    RightsRisk,
    RightsStatus,
    SourceMomentStructure,
)
from app.transformation.policy import (
    PLATFORM_POLICY_CHECKED_DATE,
    PLATFORM_POLICY_REFERENCES,
    PLATFORM_RISK_POLICY_VERSION,
    Stage40Config,
)
from app.transformation.types import SourceMoment, TransformationInputs, clamp

# ---- Bounded reason codes -------------------------------------------------

REASON_TRANSCRIPT_SHORT = "TRANSCRIPT_TOO_SHORT"
REASON_TRANSCRIPT_LOW_CONFIDENCE = "TRANSCRIPT_LOW_CONFIDENCE"
REASON_MEANING_CRITICAL_AMBIGUITY = "MEANING_CRITICAL_AMBIGUITY"
REASON_CONTEXT_MISSING = "REQUIRED_CONTEXT_UNAVAILABLE"
REASON_PROVENANCE_CONFLICT = "UNRESOLVED_PROVENANCE_CONFLICT"
REASON_THIRD_PARTY_TRANSFORMATION_REQUIRED = "THIRD_PARTY_TRANSFORMATION_REQUIRED"
REASON_NO_SUBSTANTIVE_STRATEGY = "NO_SUBSTANTIVE_STRATEGY"
REASON_ELEVATED_PROVENANCE_RISK = "ELEVATED_PROVENANCE_RISK"
REASON_EXTERNAL_VERIFICATION_REQUIRED = "EXTERNAL_VERIFICATION_REQUIRED"
REASON_HIGH_DAMAGE_RISK = "HIGH_SOURCE_MOMENT_DAMAGE_RISK"
REASON_SHORT_MOMENT_SETUP_DAMAGE = "SHORT_MOMENT_SETUP_DAMAGE"
REASON_PRESENTATION_ONLY_ONLY = "PRESENTATION_ONLY_ONLY"
REASON_EXTERNAL_FACT_UNMARKED = "UNVERIFIED_EXTERNAL_FACT_DEPENDENCY"

_THIRD_PARTY_RIGHTS = {
    RightsStatus.THIRD_PARTY_UNKNOWN,
    RightsStatus.THIRD_PARTY_REUSE,
}
_BROADCAST_ORIGINS = {
    MediaOriginType.MOVIE_TV,
    MediaOriginType.NEWS_CLIP,
    MediaOriginType.SPORTS_BROADCAST,
}
_CONTINUATION_TOKENS = frozenset(
    {
        "و",
        "لكن",
        "بس",
        "ثم",
        "لأن",
        "عشان",
        "علشان",
        "and",
        "but",
        "so",
        "because",
        "then",
        "also",
        "however",
    }
)

_STRUCTURE_BY_CONTENT: dict[ContentType, SourceMomentStructure] = {
    ContentType.EDUCATIONAL: SourceMomentStructure.EXPLANATION,
    ContentType.TUTORIAL: SourceMomentStructure.EXPLANATION,
    ContentType.ANALYSIS: SourceMomentStructure.EXPLANATION,
    ContentType.INTERVIEW_INSIGHT: SourceMomentStructure.CLAIM,
    ContentType.CONTROVERSIAL_OPINION: SourceMomentStructure.CLAIM,
    ContentType.SURPRISING_FACT: SourceMomentStructure.CLAIM,
    ContentType.DEBATE: SourceMomentStructure.DEBATE,
    ContentType.NEWS_CURRENT_EVENT: SourceMomentStructure.NEWS,
    ContentType.STORY: SourceMomentStructure.STORY,
    ContentType.EMOTIONAL: SourceMomentStructure.STORY,
    ContentType.FUNNY: SourceMomentStructure.JOKE,
    ContentType.REACTION_WORTHY: SourceMomentStructure.PAYOFF,
    ContentType.MOTIVATIONAL: SourceMomentStructure.PAYOFF,
    ContentType.OTHER: SourceMomentStructure.UNKNOWN,
}


def is_third_party(inputs: TransformationInputs) -> bool:
    if inputs.rights_status in {item.value for item in _THIRD_PARTY_RIGHTS}:
        return True
    try:
        origin = MediaOriginType(inputs.media_origin)
    except ValueError:
        return False
    return origin in _BROADCAST_ORIGINS


def _first_token(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.split()[0].strip("،,.!?؟:").casefold()


def transcript_blocker(inputs: TransformationInputs, config: Stage40Config) -> str | None:
    text = inputs.transcript.strip()
    if len(text) < config.min_transcript_characters:
        return REASON_TRANSCRIPT_SHORT
    if inputs.transcript_confidence < config.min_transcript_confidence:
        return REASON_TRANSCRIPT_LOW_CONFIDENCE
    for span in inputs.unresolved_spans:
        if not isinstance(span, dict):
            continue
        critical = bool(span.get("meaning_critical"))
        resolved = str(span.get("resolution_state", "")).upper() == "RESOLVED"
        if critical and not resolved:
            return REASON_MEANING_CRITICAL_AMBIGUITY
    return None


def context_blocker(inputs: TransformationInputs, config: Stage40Config) -> str | None:
    """Detect a moment that cannot be understood without unavailable context.

    A short fragment that starts mid-thought (a leading continuation token) and
    has no bounded nearby context cannot be transformed safely.
    """

    context = " ".join(item for item in inputs.context_segments if item).strip()
    if context:
        return None
    token = _first_token(inputs.transcript)
    if token in _CONTINUATION_TOKENS:
        return REASON_CONTEXT_MISSING
    if len(inputs.transcript.strip()) < config.min_context_characters:
        return REASON_CONTEXT_MISSING
    return None


def policy_provenance_blocker(inputs: TransformationInputs) -> str | None:
    """Reserve the unresolved outcome for explicit stored conflicting evidence.

    Unknown or plainly third-party provenance never triggers this by itself.
    """

    snapshot = inputs.provenance_snapshot or {}
    if snapshot.get("provenance_conflict") or snapshot.get("rights_conflict"):
        return REASON_PROVENANCE_CONFLICT
    conflicts = snapshot.get("conflicts")
    if isinstance(conflicts, list) and conflicts:
        return REASON_PROVENANCE_CONFLICT
    if snapshot.get("rights_status_conflict") or snapshot.get("originality_conflict"):
        return REASON_PROVENANCE_CONFLICT
    return None


def transformation_necessity(inputs: TransformationInputs) -> float:
    """How much substantive transformation the source requires, in [0, 1]."""

    necessity = 0.20
    if is_third_party(inputs):
        necessity = 0.70
    elif inputs.rights_status == RightsStatus.UNKNOWN.value:
        necessity = 0.45
    elif inputs.rights_status in {
        RightsStatus.OWNED.value,
        RightsStatus.LICENSED.value,
        RightsStatus.PERMISSION.value,
        RightsStatus.PUBLIC_DOMAIN.value,
        RightsStatus.OTHER_ALLOWED.value,
    }:
        necessity = 0.15
    if inputs.originality_risk is OriginalityRisk.TRANSFORMATION_REQUIRED:
        necessity = max(necessity, 0.75)
    if inputs.rights_risk is RightsRisk.ELEVATED:
        necessity = min(1.0, necessity + 0.10)
    return clamp(necessity)


def required_originality(inputs: TransformationInputs, config: Stage40Config) -> float:
    if is_third_party(inputs):
        return config.third_party_min_originality_potential
    return config.min_originality_potential


def derive_source_moment(inputs: TransformationInputs, config: Stage40Config) -> SourceMoment:
    """Classify the moment's rhetorical shape from bounded deterministic cues."""

    structure = _STRUCTURE_BY_CONTENT.get(inputs.content_type, SourceMomentStructure.UNKNOWN)
    evidence: list[str] = [f"content_type={inputs.content_type.value}"]
    hook_index: int | None = None
    payoff_index: int | None = None
    for hook in inputs.hooks:
        if not isinstance(hook, dict):
            continue
        hook_type = str(hook.get("type", "")).upper()
        indexes = hook.get("source_segment_indexes")
        first_index = None
        if isinstance(indexes, list) and indexes and isinstance(indexes[0], int):
            first_index = int(indexes[0])
        if hook_type == "PAYOFF_FIRST":
            structure = SourceMomentStructure.PAYOFF
            payoff_index = first_index
        elif hook_type == "QUESTION" and structure in {
            SourceMomentStructure.UNKNOWN,
            SourceMomentStructure.EXPLANATION,
        }:
            structure = SourceMomentStructure.QUESTION_ANSWER
        elif hook_type == "CONTRADICTION" and structure == SourceMomentStructure.UNKNOWN:
            structure = SourceMomentStructure.CLAIM
        hook_index = hook_index if first_index is None else (hook_index or first_index)
        evidence.append(f"hook={hook_type or 'UNKNOWN'}")
    if inputs.duration <= config.short_moment_seconds:
        evidence.append("short_moment")
    if inputs.moment_density_score >= config.high_moment_density_floor:
        evidence.append("high_moment_density")
    return SourceMoment(
        structure=structure,
        hook_index=hook_index,
        payoff_index=payoff_index,
        duration=inputs.duration,
        moment_density=inputs.moment_density_score,
        evidence=tuple(evidence),
    )


def _risk(
    kind: PlatformRiskKind,
    level: PlatformRiskLevel,
    reasons: list[str],
) -> dict[str, object]:
    return {
        "kind": kind.value,
        "level": level.value,
        "reasons": reasons,
        "limitations": [
            "Decision support only; not a legal, copyright, or monetization guarantee."
        ],
    }


def platform_risk_snapshot(
    inputs: TransformationInputs,
    *,
    source_dominance_risk: float,
    template_staleness_risk: float,
    presentation_only: bool,
) -> dict[str, object]:
    """Bounded platform-risk decision-support snapshot. Never a guarantee."""

    third_party = is_third_party(inputs)
    transformation_required = (
        inputs.originality_risk is OriginalityRisk.TRANSFORMATION_REQUIRED or third_party
    )
    dimensions = [
        _risk(
            PlatformRiskKind.YOUTUBE_REUSED_CONTENT,
            (
                PlatformRiskLevel.HIGH
                if third_party and presentation_only
                else PlatformRiskLevel.MODERATE
                if third_party
                else PlatformRiskLevel.LOW
            ),
            [
                "Third-party/unknown source material."
                if third_party
                else "Material is operator-owned or licensed."
            ],
        ),
        _risk(
            PlatformRiskKind.YOUTUBE_INAUTHENTIC_REPETITIVE,
            (
                PlatformRiskLevel.HIGH
                if template_staleness_risk >= 0.7
                else PlatformRiskLevel.MODERATE
                if template_staleness_risk >= 0.4
                else PlatformRiskLevel.LOW
            ),
            [f"Template staleness risk {template_staleness_risk:.2f}."],
        ),
        _risk(
            PlatformRiskKind.FACEBOOK_UNORIGINAL_CONTENT,
            (
                PlatformRiskLevel.HIGH
                if third_party and presentation_only
                else PlatformRiskLevel.MODERATE
                if third_party
                else PlatformRiskLevel.LOW
            ),
            [
                "Minor edits/borders/captions do not qualify as meaningful enhancement."
                if presentation_only
                else "Meaningful transformation expected."
            ],
        ),
        _risk(
            PlatformRiskKind.SPAM_TEMPLATE_HEAVY,
            (PlatformRiskLevel.HIGH if template_staleness_risk >= 0.7 else PlatformRiskLevel.LOW),
            ["Deterministic template-staleness estimate only."],
        ),
        _risk(
            PlatformRiskKind.SOURCE_DOMINANCE,
            (
                PlatformRiskLevel.HIGH
                if source_dominance_risk >= 0.7
                else PlatformRiskLevel.MODERATE
                if source_dominance_risk >= 0.4
                else PlatformRiskLevel.LOW
            ),
            [f"Source-dominance risk {source_dominance_risk:.2f}."],
        ),
    ]
    return {
        "policy_version": PLATFORM_RISK_POLICY_VERSION,
        "checked_date": PLATFORM_POLICY_CHECKED_DATE,
        "transformation_required": transformation_required,
        "presentation_only": presentation_only,
        "dimensions": dimensions,
        "references": list(PLATFORM_POLICY_REFERENCES),
        "disclaimer": (
            "Decision support only. Not a monetization guarantee, copyright opinion, "
            "or legal conclusion."
        ),
    }
