"""Conservative deterministic candidate boundary refinement.

The refiner searches a bounded radius around the coarse edge for the nearest
natural boundary, preferring silence, then an inter-word gap, then a sentence
sized gap, then a complete-word edge. It never falls back to the full context
window: with no usable evidence it retains the coarse edge and reports low
confidence. All inputs are validated; impossible geometry raises ``ValueError``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.refinement.types import BoundaryResult, WordTimestamp

BOUNDARY_POLICY_VERSION = "stage3.5-boundary-v1"
_SENTENCE_GAP_SECONDS = 0.35
_COARSE_RETAINED_CONFIDENCE = 0.2
_FALLBACK_CONFIDENCE = 0.15
_PRIORITY_SILENCE = 0
_PRIORITY_GAP = 1
_PRIORITY_SENTENCE_GAP = 2
_PRIORITY_WORD = 3
_PRIORITY_CONFIDENCE = {
    _PRIORITY_SILENCE: 0.9,
    _PRIORITY_GAP: 0.8,
    _PRIORITY_SENTENCE_GAP: 0.7,
    _PRIORITY_WORD: 0.55,
}
_PRIORITY_LABEL = {
    _PRIORITY_SILENCE: "silence_boundary",
    _PRIORITY_GAP: "gap_boundary",
    _PRIORITY_SENTENCE_GAP: "sentence_gap_boundary",
    _PRIORITY_WORD: "word_boundary",
}


@dataclass(frozen=True)
class BoundarySignals:
    """Bounded deterministic inputs for one boundary-refinement pass."""

    coarse_start: float
    coarse_end: float
    context_start: float
    context_end: float
    word_timestamps: tuple[WordTimestamp, ...]
    source_word_timestamps: tuple[WordTimestamp, ...] = ()
    silence_intervals: tuple[tuple[float, float], ...] = ()
    radius_seconds: float = 5.0
    min_duration_seconds: float = 0.5


@dataclass(frozen=True)
class _EdgeDecision:
    position: float
    confidence: float
    reasons: tuple[str, ...]
    evidence: dict[str, object]
    found: bool


def _validate(signals: BoundarySignals) -> None:
    values = (
        signals.coarse_start,
        signals.coarse_end,
        signals.context_start,
        signals.context_end,
        signals.radius_seconds,
        signals.min_duration_seconds,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("boundary inputs must be finite")
    if signals.context_start < 0.0:
        raise ValueError("context_start must be non-negative")
    if signals.context_end <= signals.context_start:
        raise ValueError("context_end must exceed context_start")


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _safe_words(signals: BoundarySignals) -> tuple[WordTimestamp, ...]:
    raw = signals.word_timestamps or signals.source_word_timestamps
    words = [
        word
        for word in raw
        if math.isfinite(word.start) and math.isfinite(word.end) and word.end >= word.start
    ]
    return tuple(sorted(words, key=lambda word: (word.start, word.end)))


def _safe_silences(signals: BoundarySignals) -> tuple[tuple[float, float], ...]:
    intervals = [
        (start, end)
        for start, end in signals.silence_intervals
        if math.isfinite(start) and math.isfinite(end) and end >= start
    ]
    return tuple(sorted(intervals))


def _refine_start(
    *,
    coarse: float,
    context_start: float,
    context_end: float,
    words: tuple[WordTimestamp, ...],
    silences: tuple[tuple[float, float], ...],
    radius: float,
) -> _EdgeDecision:
    low = max(context_start, coarse - radius)
    high = min(context_end, coarse + radius)
    candidates: list[tuple[float, int, float, str]] = []

    for start, end in silences:
        if end < low or start > high:
            continue
        position = _clamp(end, context_start, context_end)
        candidates.append((abs(position - coarse), _PRIORITY_SILENCE, position, "silence"))

    for index in range(len(words) - 1):
        gap = words[index + 1].start - words[index].end
        if gap <= 0.0:
            continue
        position = words[index + 1].start
        within = low <= position <= high
        spans_coarse = words[index].end <= coarse <= words[index + 1].start
        if not within and not spans_coarse:
            continue
        priority = _PRIORITY_SENTENCE_GAP if gap >= _SENTENCE_GAP_SECONDS else _PRIORITY_GAP
        candidates.append((abs(position - coarse), priority, position, f"word_{index + 1}"))

    for index, word in enumerate(words):
        if low <= word.start <= high:
            candidates.append(
                (abs(word.start - coarse), _PRIORITY_WORD, word.start, f"word_{index}")
            )

    if not candidates:
        return _EdgeDecision(
            position=coarse,
            confidence=_FALLBACK_CONFIDENCE,
            reasons=("coarse_edge_retained",),
            evidence={"reason": "coarse_edge_retained", "alternatives": []},
            found=False,
        )

    candidates.sort(key=lambda item: (item[1], item[0], item[2]))
    distance, priority, position, label = candidates[0]
    confidence = max(0.2, _PRIORITY_CONFIDENCE[priority] - 0.05 * distance)
    alternatives = [
        {"position": round(item[2], 6), "priority": item[1], "label": item[3]}
        for item in candidates[1:6]
    ]
    return _EdgeDecision(
        position=position,
        confidence=confidence,
        reasons=(_PRIORITY_LABEL[priority],),
        evidence={
            "reason": _PRIORITY_LABEL[priority],
            "chosen": {
                "position": round(position, 6),
                "priority": priority,
                "label": label,
                "distance": round(distance, 6),
            },
            "alternatives": alternatives,
        },
        found=True,
    )


def _refine_end(
    *,
    coarse: float,
    context_start: float,
    context_end: float,
    words: tuple[WordTimestamp, ...],
    silences: tuple[tuple[float, float], ...],
    radius: float,
) -> _EdgeDecision:
    low = max(context_start, coarse - radius)
    high = min(context_end, coarse + radius)
    candidates: list[tuple[float, int, float, str]] = []

    for start, end in silences:
        if end < low or start > high:
            continue
        position = _clamp(start, context_start, context_end)
        candidates.append((abs(position - coarse), _PRIORITY_SILENCE, position, "silence"))

    for index in range(len(words) - 1):
        gap = words[index + 1].start - words[index].end
        if gap <= 0.0:
            continue
        position = words[index].end
        within = low <= position <= high
        spans_coarse = words[index].end <= coarse <= words[index + 1].start
        if not within and not spans_coarse:
            continue
        priority = _PRIORITY_SENTENCE_GAP if gap >= _SENTENCE_GAP_SECONDS else _PRIORITY_GAP
        candidates.append((abs(position - coarse), priority, position, f"word_{index}"))

    for index, word in enumerate(words):
        if low <= word.end <= high:
            candidates.append((abs(word.end - coarse), _PRIORITY_WORD, word.end, f"word_{index}"))

    if not candidates:
        return _EdgeDecision(
            position=coarse,
            confidence=_FALLBACK_CONFIDENCE,
            reasons=("coarse_edge_retained",),
            evidence={"reason": "coarse_edge_retained", "alternatives": []},
            found=False,
        )

    candidates.sort(key=lambda item: (item[1], item[0], item[2]))
    distance, priority, position, label = candidates[0]
    confidence = max(0.2, _PRIORITY_CONFIDENCE[priority] - 0.05 * distance)
    alternatives = [
        {"position": round(item[2], 6), "priority": item[1], "label": item[3]}
        for item in candidates[1:6]
    ]
    return _EdgeDecision(
        position=position,
        confidence=confidence,
        reasons=(_PRIORITY_LABEL[priority],),
        evidence={
            "reason": _PRIORITY_LABEL[priority],
            "chosen": {
                "position": round(position, 6),
                "priority": priority,
                "label": label,
                "distance": round(distance, 6),
            },
            "alternatives": alternatives,
        },
        found=True,
    )


def _expand_to_minimum(
    start: float,
    end: float,
    *,
    context_start: float,
    context_end: float,
    min_duration: float,
) -> tuple[float, float]:
    if end - start >= min_duration:
        return start, end
    missing = min_duration - (end - start)
    start = max(context_start, start - missing / 2.0)
    end = min(context_end, end + missing / 2.0)
    if end - start < min_duration:
        if start <= context_start + 1e-9:
            end = min(context_end, start + min_duration)
        else:
            start = max(context_start, end - min_duration)
    return start, end


def refine_boundaries(signals: BoundarySignals) -> BoundaryResult:
    """Refine coarse candidate boundaries within their context window."""

    _validate(signals)
    context_start = max(0.0, signals.context_start)
    context_end = signals.context_end
    radius = max(0.0, signals.radius_seconds)
    min_duration = max(0.0, signals.min_duration_seconds)

    coarse_start = _clamp(signals.coarse_start, context_start, context_end)
    coarse_end = _clamp(signals.coarse_end, context_start, context_end)
    if coarse_start > coarse_end:
        raise ValueError("coarse_start must not exceed coarse_end")

    words = _safe_words(signals)
    silences = _safe_silences(signals)

    start_decision = _refine_start(
        coarse=coarse_start,
        context_start=context_start,
        context_end=context_end,
        words=words,
        silences=silences,
        radius=radius,
    )
    end_decision = _refine_end(
        coarse=coarse_end,
        context_start=context_start,
        context_end=context_end,
        words=words,
        silences=silences,
        radius=radius,
    )

    refined_start = _clamp(start_decision.position, context_start, context_end)
    refined_end = _clamp(end_decision.position, context_start, context_end)
    reasons: list[str] = []
    confidence = min(start_decision.confidence, end_decision.confidence)

    if refined_start >= refined_end:
        refined_start, refined_end = coarse_start, coarse_end
        reasons.append("coarse_edge_retained")
        confidence = _COARSE_RETAINED_CONFIDENCE

    if refined_end - refined_start < min_duration:
        refined_start, refined_end = coarse_start, coarse_end
        if refined_end - refined_start < min_duration:
            refined_start, refined_end = _expand_to_minimum(
                refined_start,
                refined_end,
                context_start=context_start,
                context_end=context_end,
                min_duration=min_duration,
            )
        reasons.append("min_duration_coarse_retained")
        confidence = min(confidence, _COARSE_RETAINED_CONFIDENCE)

    if refined_start >= refined_end:
        refined_start, refined_end = _expand_to_minimum(
            coarse_start,
            coarse_end,
            context_start=context_start,
            context_end=context_end,
            min_duration=min(min_duration, context_end - context_start),
        )
        reasons.append("coarse_edge_retained")
        confidence = min(confidence, _COARSE_RETAINED_CONFIDENCE)

    for decision in (start_decision, end_decision):
        for reason in decision.reasons:
            if reason not in reasons:
                reasons.append(reason)

    evidence: dict[str, object] = {
        "policy_version": BOUNDARY_POLICY_VERSION,
        "coarse_start": round(coarse_start, 6),
        "coarse_end": round(coarse_end, 6),
        "context_start": round(context_start, 6),
        "context_end": round(context_end, 6),
        "radius_seconds": radius,
        "min_duration_seconds": min_duration,
        "start": start_decision.evidence,
        "end": end_decision.evidence,
        "word_count": len(words),
        "silence_count": len(silences),
    }

    return BoundaryResult(
        start=refined_start,
        end=refined_end,
        confidence=min(1.0, max(0.0, confidence)),
        reasons=tuple(reasons),
        evidence=evidence,
    )
