"""Validated ordered occurrences and source/output mapping for Stage 5.2.

The current source-validation order follows the ordered contract blocks; a
render never sorts occurrences by source timestamp and never merges separate
occurrences merely because their intervals touch. Output time is unit-speed:

    output_time = occurrence.output_start + source_time - occurrence.source_start

Boundaries are accumulated cumulatively (never accumulated independently per
scene), so multi-span execution cannot drift.
"""

from __future__ import annotations

from fractions import Fraction

from app.render.execution.policy import UNSUPPORTED_OUTPUT_GEOMETRY
from app.render.execution.types import (
    RenderSpec,
    TimelineManifest,
    TimelineOccurrence,
)

_BOUNDS_EPSILON = 1e-6


class TimelineError(ValueError):
    """A source/output timeline could not be built deterministically."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _round_frames(seconds: float, frame_rate: Fraction) -> int:
    return int(round(seconds * float(frame_rate)))


def _round_samples(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def validate_occurrence_order(occurrences: tuple[TimelineOccurrence, ...]) -> None:
    """Require contiguous, non-overlapping output intervals in contract order."""

    if not occurrences:
        raise TimelineError("NO_RENDERABLE_OCCURRENCES")
    cursor = 0.0
    for occurrence in occurrences:
        if occurrence.source_end <= occurrence.source_start:
            raise TimelineError("OCCURRENCE_BOUNDS_INVALID")
        if abs(occurrence.output_start - cursor) > 1e-4:
            raise TimelineError("OCCURRENCE_BOUNDS_INVALID")
        if occurrence.output_end <= occurrence.output_start:
            raise TimelineError("OCCURRENCE_BOUNDS_INVALID")
        cursor = occurrence.output_end


def map_source_to_output(occurrences: tuple[TimelineOccurrence, ...], source_time: float) -> float:
    """Map a source-local timestamp to the output timeline (unit speed)."""

    for occurrence in occurrences:
        if occurrence.source_start <= source_time < occurrence.source_end:
            return occurrence.output_start + source_time - occurrence.source_start
    if occurrences and abs(source_time - occurrences[-1].source_end) < _BOUNDS_EPSILON:
        return occurrences[-1].output_end
    raise TimelineError("OCCURRENCE_BOUNDS_INVALID")


def build_timeline(spec: RenderSpec) -> TimelineManifest:
    """Build the quantized source/output timeline for a validated spec."""

    if spec.output_width <= 0 or spec.output_height <= 0:
        raise TimelineError(UNSUPPORTED_OUTPUT_GEOMETRY)
    validate_occurrence_order(spec.occurrences)
    frame_rate = spec.output_frame_rate
    if frame_rate <= 0:
        raise TimelineError(UNSUPPORTED_OUTPUT_GEOMETRY)

    output_duration = sum(occurrence.output_duration for occurrence in spec.occurrences)
    frame_count = _round_frames(output_duration, frame_rate)
    sample_count = _round_samples(output_duration, 48000)
    return TimelineManifest(
        occurrences=spec.occurrences,
        frame_rate=frame_rate,
        sample_rate=48000,
        source_duration=spec.source_duration,
        output_duration=output_duration,
        output_frame_count=max(1, frame_count),
        output_sample_count=max(1, sample_count),
    )


def frame_boundaries(durations: list[float], frame_rate: Fraction) -> list[int]:
    """Cumulative output-frame boundaries for a sequence of segment durations."""

    boundaries = [0]
    total = 0.0
    for duration in durations:
        total += duration
        boundaries.append(_round_frames(total, frame_rate))
    return boundaries


def sample_boundaries(durations: list[float], sample_rate: int = 48000) -> list[int]:
    """Cumulative output-sample boundaries for a sequence of segment durations."""

    boundaries = [0]
    total = 0.0
    for duration in durations:
        total += duration
        boundaries.append(_round_samples(total, sample_rate))
    return boundaries


__all__ = [
    "TimelineError",
    "build_timeline",
    "frame_boundaries",
    "map_source_to_output",
    "sample_boundaries",
    "validate_occurrence_order",
]
