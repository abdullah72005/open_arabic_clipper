"""Build bounded reconstruction windows from immutable transcript evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.transcription.reconstruction.types import (
    AcousticEvidence,
    ReconstructionWindow,
    WindowSegment,
)


@dataclass(frozen=True)
class WindowConfig:
    """Fixed bounds limiting provider context and request payload size."""

    target_seconds: float = 8.0
    target_segments: int = 5
    max_seconds: float = 15.0
    max_segments: int = 8
    provider_batch_windows: int = 16
    provider_batch_characters: int = 48_000


def acoustic_evidence(segment: Mapping[str, object]) -> AcousticEvidence:
    """Derive confidence from public Whisper fields without inventing missing data."""

    word_probabilities = [
        float(word["probability"])
        for word in _words(segment)
        if isinstance(word.get("probability"), int | float)
    ]
    word_score = sum(word_probabilities) / len(word_probabilities) if word_probabilities else None
    avg_logprob = _number(segment.get("avg_logprob"))
    no_speech_prob = _number(segment.get("no_speech_prob"))
    weighted: list[tuple[float, float]] = []
    if word_score is not None:
        weighted.append((0.50, word_score))
    if avg_logprob is not None:
        weighted.append((0.35, min(1.0, max(0.0, math.exp(avg_logprob)))))
    if no_speech_prob is not None:
        weighted.append((0.15, min(1.0, max(0.0, 1.0 - no_speech_prob))))
    confidence = (
        sum(weight * value for weight, value in weighted) / sum(weight for weight, _ in weighted)
        if weighted
        else None
    )
    return AcousticEvidence(confidence, word_score, avg_logprob, no_speech_prob)


def build_reconstruction_window(
    segments: Sequence[Mapping[str, object]],
    target_index: int,
    config: WindowConfig = WindowConfig(),
) -> ReconstructionWindow:
    """Return bounded ordered context while keeping the target's original identity."""

    if not 0 <= target_index < len(segments):
        raise IndexError("target_index is outside the segment list")
    selected = [target_index]
    if _duration(segments, selected) > config.max_seconds:
        return ReconstructionWindow(
            target_index, tuple(_window_segment(segments, target_index) for _ in selected)
        )

    left = target_index - 1
    right = target_index + 1
    choose_left = True
    while len(selected) < config.max_segments:
        candidates = (left, right) if choose_left else (right, left)
        candidate = next((index for index in candidates if 0 <= index < len(segments)), None)
        if candidate is None:
            break
        proposed = sorted((*selected, candidate))
        if _duration(segments, proposed) > config.max_seconds:
            alternate = right if candidate == left else left
            if not 0 <= alternate < len(segments):
                break
            proposed = sorted((*selected, alternate))
            if _duration(segments, proposed) > config.max_seconds:
                break
            candidate = alternate
        selected = proposed
        if candidate < target_index:
            left = candidate - 1
        else:
            right = candidate + 1
        choose_left = not choose_left
        if (
            len(selected) >= config.target_segments
            and _duration(segments, selected) >= config.target_seconds
        ):
            break
    return ReconstructionWindow(
        target_index, tuple(_window_segment(segments, index) for index in selected)
    )


def _window_segment(segments: Sequence[Mapping[str, object]], index: int) -> WindowSegment:
    segment = segments[index]
    raw_text = str(segment.get("raw_text", segment.get("text", "")))
    return WindowSegment(
        segment_index=index,
        start=float(segment.get("start", 0.0)),
        end=float(segment.get("end", 0.0)),
        raw_text=raw_text,
        corrected_text=str(segment.get("corrected_text", raw_text)),
        acoustic=acoustic_evidence(segment),
    )


def _duration(segments: Sequence[Mapping[str, object]], indexes: Sequence[int]) -> float:
    return float(segments[indexes[-1]].get("end", 0.0)) - float(
        segments[indexes[0]].get("start", 0.0)
    )


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _words(segment: Mapping[str, object]) -> list[Mapping[str, object]]:
    words = segment.get("words")
    return [word for word in words if isinstance(word, Mapping)] if isinstance(words, list) else []
