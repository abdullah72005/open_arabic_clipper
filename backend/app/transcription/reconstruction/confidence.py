"""Server-side confidence policy for safe contextual reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from app.transcription.reconstruction.types import ConfidenceLevel, ResolutionScores


@dataclass(frozen=True)
class ReconstructionDecision:
    level: ConfidenceLevel
    applied: bool
    score: float


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
    if (
        score >= 0.86
        and margin >= 0.12
        and phonetic_similarity >= 0.72
        and resolution.semantic_coherence >= 0.80
    ):
        return ReconstructionDecision(ConfidenceLevel.HIGH, True, score)
    if (
        score >= 0.74
        and margin >= 0.08
        and edit_ratio <= 0.20
        and token_delta <= 1
        and phonetic_similarity >= 0.85
        and resolution.semantic_coherence >= 0.75
    ):
        return ReconstructionDecision(ConfidenceLevel.MEDIUM, False, score)
    return ReconstructionDecision(ConfidenceLevel.LOW, False, score)
