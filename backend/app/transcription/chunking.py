"""Timestamp-aware transcript chunks for later semantic analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ChunkConfig:
    """Bounded chunk size controls for long media."""

    target_seconds: float = 90.0
    context_segments: int = 1


@dataclass(frozen=True)
class TranscriptChunk:
    """One coherent analysis window and its neighboring context."""

    start_time: float
    end_time: float
    text: str
    segment_indexes: list[int]
    preceding_context: str
    following_context: str


def build_chunks(
    segments: Sequence[Mapping[str, object]], config: ChunkConfig = ChunkConfig()
) -> list[TranscriptChunk]:
    """Group complete segments near target duration, preserving neighbor text."""

    if config.target_seconds <= 0:
        raise ValueError("target_seconds must be positive")
    if config.context_segments < 0:
        raise ValueError("context_segments must be non-negative")
    chunks: list[TranscriptChunk] = []
    current_indexes: list[int] = []
    for index, segment in enumerate(segments):
        if current_indexes:
            current_start = _time(segments[current_indexes[0]], "start")
            prospective_end = _time(segment, "end")
            if prospective_end - current_start > config.target_seconds:
                chunks.append(_make_chunk(segments, current_indexes, config.context_segments))
                current_indexes = []
        current_indexes.append(index)
    if current_indexes:
        chunks.append(_make_chunk(segments, current_indexes, config.context_segments))
    return chunks


def _make_chunk(
    segments: Sequence[Mapping[str, object]], indexes: list[int], context_count: int
) -> TranscriptChunk:
    first, last = indexes[0], indexes[-1]
    before = range(max(0, first - context_count), first)
    after = range(last + 1, min(len(segments), last + 1 + context_count))
    return TranscriptChunk(
        start_time=_time(segments[first], "start"),
        end_time=_time(segments[last], "end"),
        text=" ".join(_text(segments[index]) for index in indexes).strip(),
        segment_indexes=indexes.copy(),
        preceding_context=" ".join(_text(segments[index]) for index in before).strip(),
        following_context=" ".join(_text(segments[index]) for index in after).strip(),
    )


def _time(segment: Mapping[str, object], field: str) -> float:
    return float(segment[field])


def _text(segment: Mapping[str, object]) -> str:
    return str(
        segment.get("final_text") or segment.get("normalized_text") or segment.get("text", "")
    )
