from app.transcription.reconstruction.confidence import decide_candidate
from app.transcription.reconstruction.types import ConfidenceLevel, ResolutionScores


def test_medium_candidate_is_review_only() -> None:
    """A plausible small edit cannot replace final text without HIGH confidence."""

    decision = decide_candidate(
        phonetic_similarity=0.85,
        resolution=ResolutionScores(0.75, 0.75, 0.75, 1.0, 0.8),
        raw_acoustic_confidence=0.0,
        edit_ratio=0.1,
        margin=0.08,
        token_delta=1,
    )

    assert decision.level is ConfidenceLevel.MEDIUM
    assert decision.applied is False


def test_high_candidate_is_only_automatic_reconstruction() -> None:
    """Only candidates meeting every HIGH boundary may replace Stage 2.5 output."""

    decision = decide_candidate(
        phonetic_similarity=0.9,
        resolution=ResolutionScores(0.9, 0.9, 0.9, 1.0, 0.9),
        raw_acoustic_confidence=0.0,
        edit_ratio=0.1,
        margin=0.12,
        token_delta=1,
    )

    assert decision.level is ConfidenceLevel.HIGH
    assert decision.applied is True
