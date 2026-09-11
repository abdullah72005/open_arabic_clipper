"""Pure, deterministic, streaming-friendly Stage 3 coarse proposal generation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from app.candidates.cues import CONTENT_CUES, CONTRAST_CUES, FILLER_CUES
from app.candidates.policy import DEFAULT_CONFIG, Stage3Config
from app.candidates.text import (
    analysis_segment_text,
    analysis_texts,
    is_question,
    is_sentence_end,
    segment_end,
    segment_start,
    tokenize,
)
from app.candidates.types import Proposal

_PAUSE_SECONDS = 1.2
_ENERGY_CHANGE = 0.18
_MERGE_OVERLAP = 0.5
_MERGE_SIMILARITY = 0.6


def generate_proposals(
    segments: Sequence[Mapping[str, object]],
    *,
    duration: float,
    config: Stage3Config = DEFAULT_CONFIG,
    silence_intervals: Sequence[Mapping[str, object]] = (),
    features: Sequence[Mapping[str, object]] = (),
) -> list[Proposal]:
    """Generate bounded coarse proposals in one near-linear pass.

    Boundaries are formed from segment/word timestamps, sentence punctuation,
    pauses, speaker changes, question/answer and contrast/topic transitions, and
    RMS energy changes. Fixed fallback windows are used only when meaningful
    boundaries are unavailable.
    """

    count = len(segments)
    if count == 0:
        return []
    effective_duration = max(duration, segment_end(segments[-1]), 0.0)
    texts = analysis_texts(segments)
    starts = [_clamp(segment_start(segment), 0.0, effective_duration) for segment in segments]
    ends = [_clamp(segment_end(segment), 0.0, effective_duration) for segment in segments]

    ranges: list[tuple[int, int, str]] = []
    start = 0
    for index in range(count):
        index_duration = ends[index] - starts[start]
        if index == count - 1:
            ranges.append((start, index, "end_of_source"))
            break
        if index_duration >= config.max_window_seconds:
            ranges.append((start, index, "max_window"))
            start = index + 1
            continue
        reason = _boundary_reason(segments, texts, starts, ends, index, features, silence_intervals)
        if reason is not None and index_duration >= config.min_window_seconds:
            if index_duration >= config.preferred_window_min_seconds:
                ranges.append((start, index, reason))
                start = index + 1
    if start < count and (not ranges or ranges[-1][1] < start):
        ranges.append((start, count - 1, "tail"))

    if not ranges:
        ranges = _fallback_ranges(starts, ends, config)
    if not ranges:
        return []

    proposals = [
        _build_proposal(start, end, reason, segments, starts, ends, effective_duration)
        for start, end, reason in ranges
    ]
    proposals = [proposal for proposal in proposals if proposal.end_time > proposal.start_time]
    proposals = _merge_similar(proposals, config)
    proposals = _prune(proposals, segments, effective_duration, config)
    return proposals


def _boundary_reason(
    segments: Sequence[Mapping[str, object]],
    texts: Sequence[str],
    starts: Sequence[float],
    ends: Sequence[float],
    index: int,
    features: Sequence[Mapping[str, object]],
    silence_intervals: Sequence[Mapping[str, object]],
) -> str | None:
    current = segments[index]
    following = segments[index + 1]
    if _speaker(current) is not None and _speaker(following) is not None:
        if _speaker(current) != _speaker(following):
            return "speaker_change"
    gap = starts[index + 1] - ends[index]
    if gap >= _PAUSE_SECONDS:
        return "pause"
    if _in_silence(ends[index], silence_intervals):
        return "silence"
    if _energy_change(ends[index], features):
        return "energy_change"
    if is_sentence_end(texts[index]):
        if is_question(texts[index]) and not is_question(texts[index + 1]):
            return "question_answer"
        return "sentence_end"
    if _starts_with_cue(texts[index + 1], CONTRAST_CUES):
        return "contrast"
    return None


def _fallback_ranges(
    starts: Sequence[float], ends: Sequence[float], config: Stage3Config
) -> list[tuple[int, int, str]]:
    """Only used when no meaningful segment/pause boundary exists."""

    if not starts:
        return []
    target = config.preferred_window_max_seconds
    ranges: list[tuple[int, int, str]] = []
    start = 0
    for index in range(len(starts)):
        duration = ends[index] - starts[start]
        if duration >= target or index == len(starts) - 1:
            ranges.append((start, index, "fallback_window"))
            start = index + 1
    return [item for item in ranges if item[0] <= item[1]]


def _build_proposal(
    start: int,
    end: int,
    reason: str,
    segments: Sequence[Mapping[str, object]],
    starts: Sequence[float],
    ends: Sequence[float],
    duration: float,
) -> Proposal:
    text = " ".join(
        analysis_segment_text(segments[index]).strip()
        for index in range(start, end + 1)
        if analysis_segment_text(segments[index]).strip()
    ).strip()
    start_time = _clamp(starts[start], 0.0, duration)
    end_time = _clamp(max(ends[end], start_time), 0.0, duration)
    return Proposal(
        start_segment_index=start,
        end_segment_index=end,
        start_time=start_time,
        end_time=end_time,
        text=text,
        boundary_reason=reason,
        segment_indexes=tuple(range(start, end + 1)),
    )


def _merge_similar(proposals: list[Proposal], config: Stage3Config) -> list[Proposal]:
    merged: list[Proposal] = []
    for proposal in proposals:
        target = None
        for existing in merged:
            if (
                _overlap_ratio(existing, proposal) >= _MERGE_OVERLAP
                and _text_similarity(existing.text, proposal.text) >= _MERGE_SIMILARITY
            ):
                target = existing
                break
        if target is None:
            merged.append(proposal)
            continue
        index = merged.index(target)
        merged[index] = _union(target, proposal)
    return merged


def _union(first: Proposal, second: Proposal) -> Proposal:
    start = min(first.start_segment_index, second.start_segment_index)
    end = max(first.end_segment_index, second.end_segment_index)
    text = first.text if len(first.text) >= len(second.text) else second.text
    return Proposal(
        start_segment_index=start,
        end_segment_index=end,
        start_time=min(first.start_time, second.start_time),
        end_time=max(first.end_time, second.end_time),
        text=text,
        boundary_reason="merged",
        segment_indexes=tuple(range(start, end + 1)),
    )


def _overlap_ratio(first: Proposal, second: Proposal) -> float:
    overlap = max(
        0.0, min(first.end_time, second.end_time) - max(first.start_time, second.start_time)
    )
    shorter = min(first.duration, second.duration)
    return overlap / shorter if shorter > 0 else 0.0


def _text_similarity(first: str, second: str) -> float:
    left = set(tokenize(first))
    right = set(tokenize(second))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _prune(
    proposals: list[Proposal],
    segments: Sequence[Mapping[str, object]],
    duration: float,
    config: Stage3Config,
) -> list[Proposal]:
    hours = max(1.0, duration / 3600.0)
    cap = min(
        config.max_proposals_per_source,
        int(math.ceil(hours * config.max_proposals_per_hour)),
    )
    cap = max(cap, 1)
    if len(proposals) <= cap:
        return proposals
    scored = sorted(
        proposals,
        key=lambda proposal: _pre_score(proposal, segments),
        reverse=True,
    )
    kept = scored[:cap]
    return sorted(kept, key=lambda proposal: proposal.start_segment_index)


def _pre_score(proposal: Proposal, segments: Sequence[Mapping[str, object]]) -> float:
    tokens = tokenize(proposal.text)
    cue_hits = sum(sum(1 for cue in cues if cue in proposal.text) for cues in CONTENT_CUES.values())
    filler = sum(1 for cue in FILLER_CUES if cue in proposal.text)
    suitability = 1.0 - min(1.0, abs(proposal.duration - 55.0) / 55.0)
    density = min(1.0, len(tokens) / max(1.0, proposal.duration) * 8.0)
    return cue_hits * 0.5 + len(tokens) * 0.01 + suitability + density - filler * 0.2


def _speaker(segment: Mapping[str, object]) -> str | None:
    value = segment.get("speaker")
    return str(value) if isinstance(value, str) and value else None


def _starts_with_cue(text: str, cues: Sequence[str]) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(cue) for cue in cues)


def _in_silence(moment: float, silence_intervals: Sequence[Mapping[str, object]]) -> bool:
    for interval in silence_intervals:
        start = _number(interval.get("start"))
        end = _number(interval.get("end"))
        if start - 0.05 <= moment <= end + 0.05:
            return True
    return False


def _energy_change(moment: float, features: Sequence[Mapping[str, object]]) -> bool:
    before = None
    after = None
    for feature in features:
        start = _number(feature.get("start"))
        end = _number(feature.get("end"))
        rms = _number(feature.get("rms"))
        if end <= moment:
            before = rms
        elif start >= moment and after is None:
            after = rms
    if before is None or after is None:
        return False
    return abs(after - before) >= _ENERGY_CHANGE


def _number(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        high = low
    return max(low, min(high, value))
