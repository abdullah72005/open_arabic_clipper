"""Immutable values shared by contextual reconstruction services."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcousticEvidence:
    """Public faster-whisper confidence indicators for one raw segment."""

    confidence: float | None
    average_word_probability: float | None
    average_log_probability: float | None
    no_speech_probability: float | None


@dataclass(frozen=True)
class WindowSegment:
    """One immutable transcript segment included in a reconstruction window."""

    segment_index: int
    start: float
    end: float
    raw_text: str
    corrected_text: str
    acoustic: AcousticEvidence


@dataclass(frozen=True)
class ReconstructionWindow:
    """Bounded local transcript context for exactly one target segment."""

    target_segment_index: int
    segments: tuple[WindowSegment, ...]
