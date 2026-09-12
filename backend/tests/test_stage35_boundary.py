"""Deterministic tests for Stage 3.5 boundary refinement."""

import pytest

from app.refinement.boundary import BoundarySignals, refine_boundaries
from app.refinement.types import WordTimestamp


def _signals(**overrides: object) -> BoundarySignals:
    base: dict[str, object] = {
        "coarse_start": 10.0,
        "coarse_end": 20.0,
        "context_start": 0.0,
        "context_end": 30.0,
        "word_timestamps": (),
        "source_word_timestamps": (),
        "silence_intervals": (),
        "radius_seconds": 5.0,
        "min_duration_seconds": 0.5,
    }
    base.update(overrides)
    return BoundarySignals(**base)  # type: ignore[arg-type]


def test_refined_bounds_stay_within_context_with_source_time_words() -> None:
    words = (
        WordTimestamp("a", 9.8, 10.4),
        WordTimestamp("b", 19.6, 20.4),
    )
    result = refine_boundaries(
        _signals(
            context_start=8.0,
            context_end=22.0,
            silence_intervals=((8.5, 9.4), (20.6, 21.4)),
            word_timestamps=words,
        )
    )

    assert result.start >= 8.0
    assert result.end <= 22.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.evidence["word_count"] == 2


def test_coarse_edges_tighten_to_nearby_silence_gap() -> None:
    result = refine_boundaries(
        _signals(
            silence_intervals=((8.0, 9.6), (20.4, 22.0)),
            word_timestamps=(
                WordTimestamp("w", 9.7, 10.2),
                WordTimestamp("x", 19.8, 20.3),
            ),
        )
    )

    assert result.start == pytest.approx(9.6)
    assert result.end == pytest.approx(20.4)
    assert "silence_boundary" in result.reasons
    assert result.confidence > 0.5


def test_weak_evidence_retains_coarse_edge() -> None:
    result = refine_boundaries(_signals())

    assert result.start == pytest.approx(10.0)
    assert result.end == pytest.approx(20.0)
    assert "coarse_edge_retained" in result.reasons
    assert result.confidence <= 0.5


def test_never_uses_full_context_bounds_by_default() -> None:
    result = refine_boundaries(
        _signals(context_start=0.0, context_end=100.0, coarse_start=40.0, coarse_end=60.0)
    )

    assert result.start > 0.0
    assert result.end < 100.0
    assert result.start == pytest.approx(40.0)
    assert result.end == pytest.approx(60.0)


def test_impossible_input_raises() -> None:
    with pytest.raises(ValueError):
        refine_boundaries(_signals(context_start=10.0, context_end=10.0))

    with pytest.raises(ValueError):
        refine_boundaries(_signals(coarse_start=float("nan")))


def test_duration_floor_respected() -> None:
    result = refine_boundaries(
        _signals(coarse_start=10.0, coarse_end=10.2, min_duration_seconds=1.0)
    )

    assert result.end - result.start >= 1.0
    assert "min_duration_coarse_retained" in result.reasons
