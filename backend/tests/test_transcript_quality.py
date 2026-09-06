from __future__ import annotations

import pytest

from app.models import Transcript
from app.services.source_quality import assess_transcript_quality


def transcript_with(
    *,
    probabilities: list[float],
    statuses: list[str] | None = None,
    provider_availability: str = "AVAILABLE",
) -> Transcript:
    statuses = statuses or ["UNCHANGED_HIGH_CONFIDENCE"]
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": " كلام",
            "corrected_text": "كلام",
            "reconstruction_status": statuses[min(index, len(statuses) - 1)],
            "reconstruction_confidence": 0.0,
            "routing_priority": "RECONSTRUCT",
            "words": [
                {"word": f" w{word}", "probability": probability}
                for word, probability in enumerate(probabilities)
            ],
        }
        for index in range(len(statuses))
    ]
    return Transcript(
        whisper_model="large-v3-turbo",
        transcription_options={},
        input_fingerprint="a" * 64,
        raw_text="كلام",
        normalized_text="كلام",
        corrected_text="كلام",
        final_text="كلام",
        language="ar",
        detected_language_probability=0.99,
        duration=3.0,
        segments=segments,
        word_segments=[],
        reconstruction_metadata={"provider_availability": provider_availability},
    )


def test_low_confidence_word_ratio_uses_configured_boundary() -> None:
    evidence = assess_transcript_quality(
        transcript_with(probabilities=[0.71, 0.72, 0.90])
    )

    assert evidence.low_confidence_word_ratio == pytest.approx(1 / 3)


def test_applied_segment_blends_acoustic_and_reconstruction_confidence() -> None:
    transcript = transcript_with(probabilities=[0.8], statuses=["APPLIED"])
    transcript.segments[0]["reconstruction_confidence"] = 0.6

    evidence = assess_transcript_quality(transcript)

    assert evidence.score == pytest.approx(0.35 * 0.8 + 0.65 * 0.6)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("LOW_CONFIDENCE_UNRESOLVED", 0.45),
        ("PROVIDER_UNAVAILABLE", 0.40),
        ("FAILED", 0.25),
        ("MANUAL_OVERRIDE", 0.95),
    ],
)
def test_reconstruction_status_applies_quality_cap_or_review_score(
    status: str, expected: float
) -> None:
    transcript = transcript_with(probabilities=[0.99], statuses=[status])

    evidence = assess_transcript_quality(transcript)

    assert evidence.score == pytest.approx(expected)


def test_provider_unavailable_segment_is_uncapped_when_not_routed() -> None:
    transcript = transcript_with(
        probabilities=[0.91],
        statuses=["PROVIDER_UNAVAILABLE"],
        provider_availability="UNAVAILABLE",
    )
    transcript.segments[0]["routing_priority"] = "KEEP"

    evidence = assess_transcript_quality(transcript)

    assert evidence.score == pytest.approx(0.91)


def test_not_required_segments_are_excluded_from_weighted_average() -> None:
    transcript = transcript_with(
        probabilities=[0.9], statuses=["NOT_REQUIRED", "FAILED"]
    )

    evidence = assess_transcript_quality(transcript)

    assert evidence.score == pytest.approx(0.25)
    assert evidence.unresolved_segment_ratio == pytest.approx(1.0)


def test_missing_word_confidence_is_zero_without_false_low_confidence() -> None:
    transcript = transcript_with(probabilities=[])

    evidence = assess_transcript_quality(transcript)

    assert evidence.score == 0.0
    assert evidence.low_confidence_word_ratio == 0.0


def test_manual_review_is_required_for_unresolved_or_unavailable_arabic() -> None:
    unresolved = assess_transcript_quality(
        transcript_with(probabilities=[0.9], statuses=["LOW_CONFIDENCE_UNRESOLVED"])
    )
    unavailable = assess_transcript_quality(
        transcript_with(probabilities=[0.9], provider_availability="UNAVAILABLE")
    )

    assert unresolved.manual_review_required is True
    assert unavailable.manual_review_required is True


def test_applied_stage25_correction_blends_only_when_reconstruction_not_run() -> None:
    transcript = transcript_with(probabilities=[0.8], statuses=["UNCHANGED_HIGH_CONFIDENCE"])
    transcript.segments[0].update(
        reconstruction_status=None,
        correction_applied=True,
        correction_confidence=0.6,
    )

    evidence = assess_transcript_quality(transcript)

    assert evidence.score == pytest.approx(0.5 * 0.8 + 0.5 * 0.6)
