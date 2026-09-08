import math

import pytest

from app.transcription.reconstruction.windows import (
    WindowConfig,
    acoustic_evidence,
    build_reconstruction_window,
)


def test_window_expands_both_sides_without_changing_target_identity() -> None:
    """A target sees bounded neighboring context while retaining its stable index."""

    segments = [
        {"start": float(index * 2), "end": float(index * 2 + 2), "text": f"s{index}"}
        for index in range(9)
    ]

    window = build_reconstruction_window(segments, 4, WindowConfig())

    assert window.target_segment_index == 4
    assert [item.segment_index for item in window.segments] == [2, 3, 4, 5, 6]
    assert window.segments[-1].end - window.segments[0].start == 10.0


def test_window_keeps_an_oversized_target_without_adding_context() -> None:
    """A long source segment remains one stable output slot instead of being split."""

    window = build_reconstruction_window(
        [
            {"start": 0.0, "end": 2.0, "text": "before"},
            {"start": 2.0, "end": 20.0, "text": "long"},
            {"start": 20.0, "end": 22.0, "text": "after"},
        ],
        1,
        WindowConfig(),
    )

    assert [item.segment_index for item in window.segments] == [1]
    assert window.target_segment_index == 1


def test_acoustic_score_uses_documented_weights() -> None:
    """Changing any public Whisper confidence signal changes reconstruction evidence."""

    evidence = acoustic_evidence(
        {
            "avg_logprob": -0.2,
            "no_speech_prob": 0.1,
            "words": [{"probability": 0.8}, {"probability": 0.6}],
        }
    )

    expected = 0.50 * 0.7 + 0.35 * math.exp(-0.2) + 0.15 * 0.9
    assert evidence.confidence == pytest.approx(expected)


def test_acoustic_score_is_unknown_without_public_confidence_fields() -> None:
    """Missing model fields must not be mistaken for low-confidence speech."""

    assert acoustic_evidence({"text": "أهلا"}).confidence is None
