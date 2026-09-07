"""Server-side confidence policy for safe contextual reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from app.transcription.reconstruction.types import ConfidenceLevel, ResolutionScores


@dataclass(frozen=True)
class ReconstructionDecision:
    level: ConfidenceLevel
    applied: bool
    score: float
    reason: str | None = None


def decide_candidate(
    *,
    phonetic_similarity: float,
    resolution: ResolutionScores,
    raw_acoustic_confidence: float | None,
    edit_ratio: float,
    margin: float,
    token_delta: int,
) -> ReconstructionDecision:
    """Apply only candidates satisfying all independently checked HIGH bounds."""

    acoustic = raw_acoustic_confidence or 0.0
    score = (
        0.35 * phonetic_similarity
        + 0.25 * resolution.semantic_coherence
        + 0.15 * resolution.discourse_continuity
        + 0.10 * resolution.egyptian_naturalness
        + 0.10 * resolution.entity_consistency
        + 0.05 * resolution.selection_confidence
        - 0.20 * acoustic * edit_ratio
    )
    high_checks = {
        "score": score >= 0.82,
        "margin": margin >= 0.12,
        "phonetic_similarity": phonetic_similarity >= 0.72,
        "semantic_coherence": resolution.semantic_coherence >= 0.80,
    }
    if all(high_checks.values()):
        return ReconstructionDecision(ConfidenceLevel.HIGH, True, score, None)
    medium_checks = {
        "score": score >= 0.74,
        "margin": margin >= 0.08,
        "edit_ratio": edit_ratio <= 0.20,
        "token_delta": token_delta <= 1,
        "phonetic_similarity": phonetic_similarity >= 0.85,
        "semantic_coherence": resolution.semantic_coherence >= 0.75,
    }
    if all(medium_checks.values()):
        return ReconstructionDecision(ConfidenceLevel.MEDIUM, False, score, None)
    failed = [name for name, passed in high_checks.items() if not passed]
    return ReconstructionDecision(
        ConfidenceLevel.LOW, False, score, ",".join(failed) or "low_confidence"
    )
