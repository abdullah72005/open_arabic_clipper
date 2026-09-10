"""Deterministic Arabic dialect profiling and protected-token preservation.

This module is deliberately pure: no network access, no LLM call, no model
loading, no audio decoding, no mutation of transcript text, and no dependency
on Qwen or Gemini. It is used by Stage 2.5 normalization to derive one
conservative source-level dialect decision and to attach exact code-switch
evidence, and by Stage 2.7 validation to preserve protected tokens.

Dialect profile semantics
-------------------------
``ArabicDialectProfile`` describes the speech actually present in the source.
It is not a target audience, localization choice, translation target, or
requested output style. A ``None`` profile means Arabic dialect profiling is
not applicable because there is no Arabic evidence; ``UNKNOWN_ARABIC`` means
Arabic is present but the regional/formal profile is uncertain, mixed,
unsupported, or insufficiently evidenced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from app.transcription.arabic import normalize_for_comparison

DIALECT_POLICY_VERSION = "dialect-policy-v1"
PRESERVATION_POLICY_VERSION = "preservation-policy-v1"
MAX_SAMPLE_SEGMENTS = 48


class ArabicDialectProfile(str, Enum):
    """Conservative source-level Arabic dialect profiles."""

    EGYPTIAN = "EGYPTIAN"
    SAUDI = "SAUDI"
    GULF = "GULF"
    LEVANTINE = "LEVANTINE"
    MSA = "MSA"
    UNKNOWN_ARABIC = "UNKNOWN_ARABIC"


class DialectSelectionMethod(str, Enum):
    """How the effective dialect profile was selected."""

    OPERATOR_OVERRIDE = "operator_override"
    DETECTED = "detected"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class CodeSwitchEvidence:
    """Exact Latin-bearing evidence already present in raw text."""

    suspected: bool
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class DialectDetectionResult:
    """One deterministic source-level dialect decision."""

    profile: ArabicDialectProfile | None
    confidence: float
    selection: DialectSelectionMethod
    reason_code: str
    evidence: dict[str, object]


# Marker weights: 2 for high-specificity region markers, 1 for supporting
# markers that are shared or less distinctive. Markers are stored in the same
# normalized comparison form produced by ``normalize_for_comparison``.
_COMPETING_PROFILES = (
    ArabicDialectProfile.EGYPTIAN,
    ArabicDialectProfile.SAUDI,
    ArabicDialectProfile.GULF,
    ArabicDialectProfile.LEVANTINE,
    ArabicDialectProfile.MSA,
)

_COLLOQUIAL_PROFILES = (
    ArabicDialectProfile.EGYPTIAN,
    ArabicDialectProfile.SAUDI,
    ArabicDialectProfile.GULF,
    ArabicDialectProfile.LEVANTINE,
)

_PROFILE_MARKERS: dict[ArabicDialectProfile, dict[str, int]] = {
    ArabicDialectProfile.EGYPTIAN: {
        "عايز": 2,
        "دلوقتي": 2,
        "ايه": 2,
        "ازيك": 2,
        "اوي": 2,
        "امبارح": 2,
        "كده": 1,
        "مش": 1,
    },
    ArabicDialectProfile.SAUDI: {
        "وش": 2,
        "الحين": 2,
        "ابغي": 2,
        "رايك": 1,
    },
    ArabicDialectProfile.GULF: {
        "شلون": 2,
        "شلونك": 2,
        "وايد": 2,
        "اشكثر": 2,
        "زين": 1,
    },
    ArabicDialectProfile.LEVANTINE: {
        "شو": 2,
        "هلأ": 2,
        "منيح": 2,
        "نبلش": 2,
        "بدي": 2,
        "رايك": 1,
    },
    ArabicDialectProfile.MSA: {
        "سوف": 2,
        "الان": 2,
        "من الضروري": 2,
        "من اجل": 2,
        "وليست": 1,
        "وليس": 1,
        "ليس": 1,
        "التي": 1,
        "الذين": 1,
    },
}

_KNOWN_PROFILE_MIN_DISTINCT = 2
_KNOWN_PROFILE_MIN_SCORE = 4
_KNOWN_PROFILE_MIN_LEAD = 2
_MSA_MIN_SCORE = 6
_MSA_MAX_COMPETING_COLLOQUIAL = 4

_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def _has_arabic_script(text: str) -> bool:
    return _ARABIC_SCRIPT.search(text) is not None


def _raw_segment_text(segment: Mapping[str, object]) -> str:
    value = segment.get("raw_text")
    if value is None:
        value = segment.get("text")
    return str(value or "")


def sample_segment_indexes(
    segments: Sequence[Mapping[str, object]], max_samples: int = MAX_SAMPLE_SEGMENTS
) -> tuple[int, ...]:
    """Deterministic representative sample of at most ``max_samples`` segments.

    Only non-empty segments are eligible. When the source has more non-empty
    segments than the sample budget, indexes are spaced evenly across the
    source and always include the first and last eligible segment.
    """

    if max_samples <= 0:
        return ()
    non_empty = [
        index for index, segment in enumerate(segments) if _raw_segment_text(segment).strip()
    ]
    if not non_empty:
        return ()
    if len(non_empty) <= max_samples:
        return tuple(non_empty)
    count = len(non_empty)
    step = (count - 1) / (max_samples - 1) if max_samples > 1 else 0.0
    selected = (non_empty[round(position * step)] for position in range(max_samples))
    return tuple(dict.fromkeys(selected))


def extract_protected_tokens(text: str) -> tuple[str, ...]:
    """Exact ordered protected tokens from raw text.

    Preserves Latin words and names, abbreviations, ordinary dotted,
    underscored, hyphenated, slash, ``+`` and ``#`` technical forms where
    practical, and Western and Arabic-Indic numbers including simple compound
    numeric/date forms. Spelling, order, casing, and digits are exact.
    """

    return tuple(_PROTECTED_TOKEN.findall(text))


_PROTECTED_TOKEN = re.compile(
    r"[A-Za-z]+[A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+"
    r"|[A-Za-z]+[A-Za-z0-9]*"
    r"|\d+(?:[.,:/-]\d+)*"
    r"|[٠-٩]+(?:[.,:/-][٠-٩]+)*"
)


def _has_latin_letter(token: str) -> bool:
    return any(character.isascii() and character.isalpha() for character in token)


def code_switch_evidence(text: str) -> CodeSwitchEvidence:
    """Extract Latin-bearing protected evidence from raw text.

    Numbers alone are protected evidence but do not imply language switching;
    only Latin-letter-bearing tokens set the suspicion signal. The caller gates
    the Arabic source context separately so an English-only source is never
    labeled Arabic-English code switching.
    """

    tokens = extract_protected_tokens(text)
    latin = tuple(token for token in tokens if _has_latin_letter(token))
    return CodeSwitchEvidence(suspected=bool(latin), tokens=latin)


class DialectDetector:
    """Conservative deterministic source-level dialect selection."""

    def __init__(
        self,
        max_samples: int = MAX_SAMPLE_SEGMENTS,
        policy_version: str = DIALECT_POLICY_VERSION,
    ) -> None:
        self._max_samples = max_samples
        self._policy_version = policy_version

    def detect(
        self,
        segments: Sequence[Mapping[str, object]],
        *,
        language: str | None = None,
        override: ArabicDialectProfile | None = None,
    ) -> DialectDetectionResult:
        """Return one deterministic dialect decision for the source transcript.

        Operator override wins with confidence 1.0. Otherwise the decision is
        derived from immutable raw segment text only, never provider
        reconstructed text.
        """

        sampled_indexes = sample_segment_indexes(segments, self._max_samples)
        sampled = [segments[index] for index in sampled_indexes]

        if override is not None:
            profile = ArabicDialectProfile(override.value)
            return DialectDetectionResult(
                profile=profile,
                confidence=1.0,
                selection=DialectSelectionMethod.OPERATOR_OVERRIDE,
                reason_code="operator_override",
                evidence=self._evidence(
                    selection=DialectSelectionMethod.OPERATOR_OVERRIDE,
                    reason_code="operator_override",
                    sampled_indexes=sampled_indexes,
                    override=profile,
                ),
            )

        if not _arabic_applicable(sampled, language):
            return DialectDetectionResult(
                profile=None,
                confidence=0.0,
                selection=DialectSelectionMethod.NOT_APPLICABLE,
                reason_code="no_arabic_evidence",
                evidence=self._evidence(
                    selection=DialectSelectionMethod.NOT_APPLICABLE,
                    reason_code="no_arabic_evidence",
                    sampled_indexes=sampled_indexes,
                ),
            )

        normalized = normalize_for_comparison(" ".join(_raw_segment_text(item) for item in sampled))
        scores, distinct = _score_markers(normalized)
        winner, runner_up = _top_two(scores)
        winner_score = scores[winner]
        runner_up_score = scores[runner_up]
        lead = winner_score - runner_up_score

        if _select_known_profile(winner, winner_score, runner_up_score, distinct, scores):
            confidence = min(0.99, 0.70 + 0.03 * winner_score + 0.02 * lead)
            reason_code = f"detected_{winner.value.lower()}"
            return DialectDetectionResult(
                profile=winner,
                confidence=confidence,
                selection=DialectSelectionMethod.DETECTED,
                reason_code=reason_code,
                evidence=self._evidence(
                    selection=DialectSelectionMethod.DETECTED,
                    reason_code=reason_code,
                    sampled_indexes=sampled_indexes,
                    marker_scores=scores,
                    distinct_marker_counts=distinct,
                    winner_score=winner_score,
                    runner_up_score=runner_up_score,
                ),
            )

        reason_code = (
            "insufficient_arabic_markers" if winner_score == 0 else "ambiguous_mixed_evidence"
        )
        return DialectDetectionResult(
            profile=ArabicDialectProfile.UNKNOWN_ARABIC,
            confidence=0.0,
            selection=DialectSelectionMethod.UNKNOWN,
            reason_code=reason_code,
            evidence=self._evidence(
                selection=DialectSelectionMethod.UNKNOWN,
                reason_code=reason_code,
                sampled_indexes=sampled_indexes,
                marker_scores=scores,
                distinct_marker_counts=distinct,
                winner_score=winner_score,
                runner_up_score=runner_up_score,
            ),
        )

    def _evidence(
        self,
        *,
        selection: DialectSelectionMethod,
        reason_code: str,
        sampled_indexes: tuple[int, ...],
        override: ArabicDialectProfile | None = None,
        marker_scores: Mapping[ArabicDialectProfile, int] | None = None,
        distinct_marker_counts: Mapping[ArabicDialectProfile, int] | None = None,
        winner_score: int | None = None,
        runner_up_score: int | None = None,
    ) -> dict[str, object]:
        evidence: dict[str, object] = {
            "dialect_policy_version": self._policy_version,
            "selection_method": selection.value,
            "reason_code": reason_code,
            "sampled_segment_count": len(sampled_indexes),
            "sampled_segment_indexes": list(sampled_indexes),
        }
        if override is not None:
            evidence["override"] = override.value
        if marker_scores is not None:
            evidence["marker_scores"] = {
                profile.value: score for profile, score in marker_scores.items()
            }
        if distinct_marker_counts is not None:
            evidence["distinct_marker_counts"] = {
                profile.value: count for profile, count in distinct_marker_counts.items()
            }
        if winner_score is not None:
            evidence["winner_score"] = winner_score
        if runner_up_score is not None:
            evidence["runner_up_score"] = runner_up_score
        return evidence


def _arabic_applicable(segments: Sequence[Mapping[str, object]], language: str | None) -> bool:
    """Arabic applicability from the reported language and/or Arabic script."""

    if language is not None and str(language).strip().lower().startswith("ar"):
        return True
    return any(_has_arabic_script(_raw_segment_text(item)) for item in segments)


def _score_markers(
    normalized_text: str,
) -> tuple[dict[ArabicDialectProfile, int], dict[ArabicDialectProfile, int]]:
    """Per-profile weighted marker scores and distinct-marker counts."""

    scores: dict[ArabicDialectProfile, int] = {profile: 0 for profile in _COMPETING_PROFILES}
    distinct: dict[ArabicDialectProfile, set[str]] = {
        profile: set() for profile in _COMPETING_PROFILES
    }
    padded = f" {normalized_text} "
    for profile, markers in _PROFILE_MARKERS.items():
        for marker, weight in markers.items():
            if _contains_marker(padded, marker):
                scores[profile] += weight
                distinct[profile].add(marker)
    return scores, {profile: len(items) for profile, items in distinct.items()}


def _contains_marker(padded_text: str, marker: str) -> bool:
    """Boundary-safe marker containment on a space-padded normalized copy."""

    return f" {marker} " in padded_text


def _top_two(
    scores: Mapping[ArabicDialectProfile, int],
) -> tuple[ArabicDialectProfile, ArabicDialectProfile]:
    ranked = sorted(scores, key=lambda profile: (-scores[profile], profile.value))
    return ranked[0], ranked[1]


def _select_known_profile(
    winner: ArabicDialectProfile,
    winner_score: int,
    runner_up_score: int,
    distinct: Mapping[ArabicDialectProfile, int],
    scores: Mapping[ArabicDialectProfile, int],
) -> bool:
    """Whether the marker evidence confidently selects a known profile.

    A known regional profile requires at least two distinct markers, a weighted
    score of at least 4, a lead of at least 2 over the runner-up, and a score of
    at least 1.5 times max(runner_up, 1). MSA additionally requires strong
    formal evidence and no meaningful competing colloquial score.
    """

    if distinct[winner] < _KNOWN_PROFILE_MIN_DISTINCT:
        return False
    if winner_score < _KNOWN_PROFILE_MIN_SCORE:
        return False
    lead = winner_score - runner_up_score
    if lead < _KNOWN_PROFILE_MIN_LEAD:
        return False
    if winner_score < 1.5 * max(runner_up_score, 1):
        return False
    if winner is ArabicDialectProfile.MSA:
        if winner_score < _MSA_MIN_SCORE:
            return False
        if any(
            scores[profile] >= _MSA_MAX_COMPETING_COLLOQUIAL for profile in _COLLOQUIAL_PROFILES
        ):
            return False
    return True
