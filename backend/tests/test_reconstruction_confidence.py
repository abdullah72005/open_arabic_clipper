import pytest

from app.transcription.reconstruction.confidence import (
    CONFIDENCE_POLICY_VERSION,
    decide_candidate,
)
from app.transcription.reconstruction.types import ConfidenceLevel


def test_high_candidate_is_only_automatic_reconstruction() -> None:
    """A candidate meeting every HIGH boundary may replace Stage 2.5 output."""

    decision = decide_candidate(
        provider_confidence=0.82,
        phonetic_similarity=0.9,
        raw_acoustic_confidence=0.0,
        edit_ratio=0.1,
        token_delta=1,
    )

    assert decision.level is ConfidenceLevel.HIGH
    assert decision.applied is True
    assert decision.provider_confidence == 0.82


def test_immediately_below_high_boundary_is_not_applied() -> None:
    """The effective 0.82 apply threshold is preserved without tuning."""

    decision = decide_candidate(
        provider_confidence=0.81,
        phonetic_similarity=0.9,
        raw_acoustic_confidence=0.0,
        edit_ratio=0.1,
        token_delta=1,
    )

    assert decision.level is ConfidenceLevel.MEDIUM
    assert decision.applied is False
    assert decision.provider_confidence == 0.81


def test_medium_candidate_is_review_only() -> None:
    """A plausible small edit cannot replace final text without HIGH confidence."""

    decision = decide_candidate(
        provider_confidence=0.76,
        phonetic_similarity=0.85,
        raw_acoustic_confidence=0.0,
        edit_ratio=0.1,
        token_delta=1,
    )

    assert decision.level is ConfidenceLevel.MEDIUM
    assert decision.applied is False


def test_acoustic_penalty_lowers_score_without_changing_evidence() -> None:
    """Low raw acoustic confidence reduces the one-scalar score deterministically."""

    decision = decide_candidate(
        provider_confidence=0.9,
        phonetic_similarity=0.9,
        raw_acoustic_confidence=1.0,
        edit_ratio=0.5,
        token_delta=1,
    )

    assert decision.applied is False
    assert decision.provider_confidence == 0.9


def test_confidence_policy_version_is_fingerprintable() -> None:
    """The provisional policy carries a stable version for runtime fingerprints."""

    assert CONFIDENCE_POLICY_VERSION == "one-pass-provider-confidence-v1"


@pytest.mark.parametrize(
    ("provider_confidence", "expected_level"),
    [
        (0.82, ConfidenceLevel.HIGH),
        (0.81, ConfidenceLevel.MEDIUM),
        (0.75, ConfidenceLevel.MEDIUM),
        (0.74, ConfidenceLevel.LOW),
    ],
)
def test_high_and_medium_boundaries(
    provider_confidence: float, expected_level: ConfidenceLevel
) -> None:
    decision = decide_candidate(
        provider_confidence=provider_confidence,
        phonetic_similarity=0.9,
        raw_acoustic_confidence=0.0,
        edit_ratio=0.1,
        token_delta=1,
    )

    assert decision.level is expected_level
