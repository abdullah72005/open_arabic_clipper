"""Server-side confidence policy for safe contextual reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from app.transcription.reconstruction.types import ConfidenceLevel

CONFIDENCE_POLICY_VERSION = "one-pass-provider-confidence-v1"
_HIGH_SCORE = 0.82
_HIGH_PHONETIC_SIMILARITY = 0.72
_MEDIUM_SCORE = 0.74


@dataclass(frozen=True)
class ReconstructionDecision:
    level: ConfidenceLevel
    applied: bool
    provider_confidence: float
    reason: str | None = None
    score: float | None = None


def decide_candidate(
    *,
    provider_confidence: float,
    phonetic_similarity: float,
    raw_acoustic_confidence: float | None,
    edit_ratio: float,
    token_delta: int,
) -> ReconstructionDecision:
    """Apply the one scalar the model actually emits, never fabricated dimensions."""

    acoustic = raw_acoustic_confidence or 0.0
    score = provider_confidence - 0.20 * acoustic * edit_ratio
    high_checks = {
        "score": score >= _HIGH_SCORE,
        "phonetic_similarity": phonetic_similarity >= _HIGH_PHONETIC_SIMILARITY,
    }
    if all(high_checks.values()):
        return ReconstructionDecision(
            ConfidenceLevel.HIGH, True, provider_confidence, "one_pass_high", score
        )
    medium_checks = {
        "score": score >= _MEDIUM_SCORE,
        "provider_confidence": provider_confidence >= 0.75,
        "edit_ratio": edit_ratio <= 0.20,
        "token_delta": token_delta <= 1,
        "phonetic_similarity": phonetic_similarity >= 0.85,
    }
    if all(medium_checks.values()):
        return ReconstructionDecision(
            ConfidenceLevel.MEDIUM, False, provider_confidence, "one_pass_medium", score
        )
    failed = [name for name, passed in high_checks.items() if not passed]
    return ReconstructionDecision(
        ConfidenceLevel.LOW,
        False,
        provider_confidence,
        ",".join(failed) or "low_confidence",
        score,
    )


def is_near_acceptance(
    decision: ReconstructionDecision,
    *,
    phonetic_similarity: float,
    margin: float,
) -> bool:
    """True when an unaccepted candidate is narrowly below the HIGH acceptance gate.

    This justifies a bounded local-to-Gemini escalation but never makes an
    unsafe local candidate acceptable.
    """

    if decision.applied or decision.score is None:
        return False
    return (
        decision.score >= _HIGH_SCORE - margin
        and phonetic_similarity >= _HIGH_PHONETIC_SIMILARITY - margin
    )
