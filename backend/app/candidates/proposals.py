"""Pure, deterministic, streaming-friendly Stage 3 coarse proposal generation.

Proposals are built over a flat sequence of bounded "atoms". A normal transcript
segment is one atom; a segment longer than ``max_window_seconds`` is split at
deterministic word/timestamp boundaries (or, when only text is available, at a
bounded proportional-character fallback) so every proposal is inside configured
bounds. This is still coarse proposal discovery, never exact boundary refinement;
Stage 3.5 owns final boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.candidates.cues import CONTENT_CUES, CONTRAST_CUES, FILLER_CUES
from app.candidates.policy import DEFAULT_CONFIG, Stage3Config
from app.candidates.text import (
    analysis_segment_text,
    contains_cue,
    is_question,
    is_sentence_end,
    matching_text,
    normalized_cue,
    segment_end,
    segment_start,
    tokenize,
)
from app.candidates.types import Proposal

_PAUSE_SECONDS = 1.2
_ENERGY_CHANGE = 0.18
_MERGE_OVERLAP = 0.5
_MERGE_SIMILARITY = 0.6


@dataclass(frozen=True)
class _Atom:
    segment_index: int
    start: float
    end: float
    text: str
    speaker: str | None


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
    RMS energy changes.
    """

    if not segments:
        return []
    effective_duration = duration if duration > 0 else max(segment_end(segments[-1]), 0.0)
    atoms = _build_atoms(segments, config, effective_duration)
    atoms = _enforce_atom_bounds(atoms, config.max_window_seconds)
    if not atoms:
        return []
    starts = [atom.start for atom in atoms]
    ends = [atom.end for atom in atoms]
    matchings = [matching_text(atom.text) for atom in atoms]

    ranges: list[tuple[int, int, str]] = []
    total = len(atoms)
    start = 0
    for index in range(total):
        if index == total - 1:
            ranges.append((start, index, "end_of_source"))
            break
        duration_to_index = ends[index] - starts[start]
        if (ends[index + 1] - starts[start]) > config.max_window_seconds:
            ranges.append((start, index, "max_window"))
            start = index + 1
            continue
        reason = _boundary_reason(
            atoms, matchings, starts, ends, index, features, silence_intervals
        )
        if (
            reason is not None
            and duration_to_index >= config.min_window_seconds
            and duration_to_index >= config.preferred_window_min_seconds
        ):
            ranges.append((start, index, reason))
            start = index + 1
    if start < total and (not ranges or ranges[-1][1] < start):
        ranges.append((start, total - 1, "tail"))

    if not ranges:
        ranges = _fallback_ranges(atoms, config)
    if not ranges:
        return []

    proposals = [
        _build_proposal(start, end, reason, atoms, effective_duration)
        for start, end, reason in ranges
    ]
    proposals = [proposal for proposal in proposals if proposal.end_time > proposal.start_time]
    proposals = _merge_similar(proposals, config)
    proposals = _prune(proposals, config)
    return proposals


def _build_atoms(
    segments: Sequence[Mapping[str, object]], config: Stage3Config, duration: float
) -> list[_Atom]:
    atoms: list[_Atom] = []
    for index, segment in enumerate(segments):
        start = _clamp(segment_start(segment), 0.0, duration)
        end = _clamp(segment_end(segment), 0.0, duration)
        text = analysis_segment_text(segment)
        speaker = _speaker(segment)
        if end - start > config.max_window_seconds:
            atoms.extend(
                _split_segment(index, start, end, text, segment, speaker, config, duration)
            )
        else:
            atoms.append(_Atom(index, start, max(start, end), text, speaker))
    return atoms


def _split_segment(
    index: int,
    start: float,
    end: float,
    text: str,
    segment: Mapping[str, object],
    speaker: str | None,
    config: Stage3Config,
    duration: float,
) -> list[_Atom]:
    words = _word_spans(segment, duration)
    if words and any(token for token, _, _ in words):
        chunks = _chunk_words(words, start, end, config)
    else:
        chunks = _chunk_text(text, start, end, config)
    atoms = [
        _Atom(index, chunk_start, chunk_end, chunk_text, speaker)
        for chunk_start, chunk_end, chunk_text in chunks
    ]
    if not atoms:
        atoms = [_Atom(index, start, end, text, speaker)]
    return atoms


def _word_spans(segment: Mapping[str, object], duration: float) -> list[tuple[str, float, float]]:
    raw = segment.get("words")
    if not isinstance(raw, list):
        return []
    spans: list[tuple[str, float, float]] = []
    for word in raw:
        if not isinstance(word, Mapping):
            continue
        word_start = _finite(word.get("start"))
        word_end = _finite(word.get("end"))
        if word_start is None or word_end is None or word_end < word_start:
            continue
        if duration > 0:
            word_start = _clamp(word_start, 0.0, duration)
            word_end = _clamp(word_end, 0.0, duration)
        if word_end <= word_start and word_start >= duration > 0:
            continue
        spans.append((str(word.get("word") or "").strip(), word_start, word_end))
    return spans


def _enforce_atom_bounds(atoms: Sequence[_Atom], limit: float) -> list[_Atom]:
    """Guarantee no atom exceeds ``limit``; split by time with proportional text.

    A single pathological word/timestamp span larger than the window cap cannot be
    retained as an atom. Its text is distributed deterministically across bounded
    time slices so the outer bound always holds.
    """

    if limit <= 0:
        return list(atoms)
    bounded: list[_Atom] = []
    for atom in atoms:
        duration = atom.end - atom.start
        if duration <= limit:
            bounded.extend([atom])
            continue
        count = max(1, math.ceil(duration / limit))
        total = len(atom.text)
        for slice_index in range(count):
            slice_start = atom.start + duration * (slice_index / count)
            slice_end = (
                atom.end
                if slice_index == count - 1
                else atom.start + duration * ((slice_index + 1) / count)
            )
            begin = round(total * (slice_index / count))
            finish = (
                total if slice_index == count - 1 else round(total * ((slice_index + 1) / count))
            )
            bounded.append(
                _Atom(
                    atom.segment_index,
                    slice_start,
                    slice_end,
                    atom.text[begin:finish],
                    atom.speaker,
                )
            )
    return bounded


def _chunk_words(
    spans: Sequence[tuple[str, float, float]], start: float, end: float, config: Stage3Config
) -> list[tuple[float, float, str]]:
    limit = config.max_window_seconds
    chunks: list[tuple[float, float, str]] = []
    current: list[tuple[str, float, float]] = []
    chunk_start = start
    for span in spans:
        _, word_start, word_end = span
        if current and (word_end - chunk_start) > limit:
            chunks.append((chunk_start, current[-1][2], " ".join(item[0] for item in current)))
            current = []
            chunk_start = word_start
        current.append(span)
    if current:
        chunks.append((chunk_start, current[-1][2], " ".join(item[0] for item in current)))
    return [
        (chunk_start, chunk_end, text) for chunk_start, chunk_end, text in chunks if text.strip()
    ]


def _chunk_text(
    text: str, start: float, end: float, config: Stage3Config
) -> list[tuple[float, float, str]]:
    """Deterministic bounded fallback preserving approximate text/time correspondence."""

    tokens = text.split()
    total = end - start
    if not tokens or total <= config.max_window_seconds:
        return [(start, end, text)]
    count = min(len(tokens), max(1, math.ceil(total / config.max_window_seconds)))
    weights = [len(token) + 1 for token in tokens]
    total_weight = sum(weights) or 1
    chunks: list[tuple[float, float, str]] = []
    previous_index = 0
    previous_time = start
    cumulative = 0
    for position in range(count):
        end_index = (
            len(tokens)
            if position == count - 1
            else max(previous_index + 1, round((position + 1) * len(tokens) / count))
        )
        end_index = min(len(tokens), max(end_index, previous_index + 1))
        group = tokens[previous_index:end_index]
        cumulative += sum(weights[previous_index:end_index])
        chunk_end = end if position == count - 1 else start + total * (cumulative / total_weight)
        chunks.append((previous_time, max(previous_time, chunk_end), " ".join(group)))
        previous_index = end_index
        previous_time = chunk_end
    return chunks


def _boundary_reason(
    atoms: Sequence[_Atom],
    matchings: Sequence[str],
    starts: Sequence[float],
    ends: Sequence[float],
    index: int,
    features: Sequence[Mapping[str, object]],
    silence_intervals: Sequence[Mapping[str, object]],
) -> str | None:
    current = atoms[index]
    following = atoms[index + 1]
    if current.speaker is not None and following.speaker is not None:
        if current.speaker != following.speaker:
            return "speaker_change"
    gap = starts[index + 1] - ends[index]
    if gap >= _PAUSE_SECONDS:
        return "pause"
    if _in_silence(ends[index], silence_intervals):
        return "silence"
    if _energy_change(ends[index], features):
        return "energy_change"
    if is_sentence_end(current.text):
        if is_question(current.text) and not is_question(following.text):
            return "question_answer"
        return "sentence_end"
    if _starts_with_cue(matchings[index + 1]):
        return "contrast"
    return None


def _fallback_ranges(atoms: Sequence[_Atom], config: Stage3Config) -> list[tuple[int, int, str]]:
    if not atoms:
        return []
    target = config.preferred_window_max_seconds
    ranges: list[tuple[int, int, str]] = []
    start = 0
    total = len(atoms)
    for index in range(total):
        if index == total - 1:
            ranges.append((start, index, "fallback_window"))
            break
        if (atoms[index + 1].end - atoms[start].start) > config.max_window_seconds:
            ranges.append((start, index, "fallback_window"))
            start = index + 1
            continue
        if (atoms[index].end - atoms[start].start) >= target:
            ranges.append((start, index, "fallback_window"))
            start = index + 1
    return [item for item in ranges if item[0] <= item[1]]


def _build_proposal(
    start: int,
    end: int,
    reason: str,
    atoms: Sequence[_Atom],
    duration: float,
) -> Proposal:
    selected = atoms[start : end + 1]
    text = " ".join(atom.text.strip() for atom in selected if atom.text.strip()).strip()
    segment_indexes = tuple(dict.fromkeys(atom.segment_index for atom in selected))
    start_time = _clamp(atoms[start].start, 0.0, duration)
    end_time = _clamp(max(atoms[end].end, start_time), 0.0, duration)
    return Proposal(
        start_segment_index=atoms[start].segment_index,
        end_segment_index=atoms[end].segment_index,
        start_time=start_time,
        end_time=end_time,
        text=text,
        boundary_reason=reason,
        segment_indexes=segment_indexes,
        span_start=start,
        span_end=end,
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
    start = min(first.span_start, second.span_start)
    end = max(first.span_end, second.span_end)
    text = first.text if len(first.text) >= len(second.text) else second.text
    segment_indexes = tuple(dict.fromkeys((*first.segment_indexes, *second.segment_indexes)))
    return Proposal(
        start_segment_index=min(first.start_segment_index, second.start_segment_index),
        end_segment_index=max(first.end_segment_index, second.end_segment_index),
        start_time=min(first.start_time, second.start_time),
        end_time=max(first.end_time, second.end_time),
        text=text,
        boundary_reason="merged",
        segment_indexes=segment_indexes,
        span_start=start,
        span_end=end,
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


def _prune(proposals: list[Proposal], config: Stage3Config) -> list[Proposal]:
    """Apply the loose raw safety cap; the tight shortlist caps run after analysis.

    This bound exists only to protect CPU/memory during discovery. It never decides
    the final shortlist: the per-hour/per-source shortlist caps are applied later,
    after full deterministic scoring, using the real ``clip_score`` ranking.
    """

    hours = max(1.0, max((proposal.end_time for proposal in proposals), default=0.0) / 3600.0)
    cap = min(
        config.max_raw_proposals_per_source,
        int(math.ceil(hours * config.max_raw_proposals_per_hour)),
    )
    cap = max(cap, 1)
    if len(proposals) <= cap:
        return proposals
    scored = sorted(proposals, key=_pre_score, reverse=True)
    kept = scored[:cap]
    return sorted(kept, key=lambda proposal: proposal.start_time)


def _pre_score(proposal: Proposal) -> float:
    tokens = tokenize(proposal.text)
    matching = matching_text(proposal.text)
    cue_hits = sum(
        sum(1 for cue in cues if contains_cue(matching, cue)) for cues in CONTENT_CUES.values()
    )
    filler = sum(1 for cue in FILLER_CUES if contains_cue(matching, cue))
    suitability = 1.0 - min(1.0, abs(proposal.duration - 55.0) / 55.0)
    density = min(1.0, len(tokens) / max(1.0, proposal.duration) * 8.0)
    return cue_hits * 0.5 + len(tokens) * 0.01 + suitability + density - filler * 0.2


def _speaker(segment: Mapping[str, object]) -> str | None:
    value = segment.get("speaker")
    return str(value) if isinstance(value, str) and value else None


def _starts_with_cue(matching: str) -> bool:
    stripped = matching.lstrip()
    return any(stripped.startswith(normalized_cue(cue)) for cue in CONTRAST_CUES)


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


def _finite(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _number(value: object) -> float:
    number = _finite(value)
    return number if number is not None else 0.0


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        high = low
    return max(low, min(high, value))
