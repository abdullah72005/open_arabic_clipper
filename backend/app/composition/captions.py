"""Deterministic FINAL_CLIP caption construction for Stage 5.1.

This module is pure and provider-free: no rendering, no provider call, no model
loading, no audio decoding, no network. It turns the selected bound source spans
plus the FINAL_CLIP word timestamps into a deterministic :class:`CaptionPlan`.

Guarantees:

- caption events are built only from words that fall inside a selected bound
  span; a span with no usable word evidence emits ``CAPTION_EVIDENCE_MISSING``
  and never fabricates text;
- canonical event text stays in logical Unicode order and is the exact source
  word tokens joined by single spaces (no reordering, no replacement, no
  direction controls);
- segmentation splits on Latin/Arabic punctuation and inter-word pauses and
  respects max duration, min duration, max words, and script-aware estimated
  width with greedy wrapping into at most ``max_lines`` lines;
- placement starts in the ``LOWER`` band and switches the *whole scene* to
  ``UPPER`` when the scene's lower band is persistently intersected by protected
  face boxes, with scene-level hysteresis so a scene only reverts when it is
  completely collision-free. Captions are never dropped for a face.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from app.composition.policy import (
    CAPTION_LAYOUT_POLICY_VERSION,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    CaptionPlacementZone,
    CaptionStyle,
    PlanReasonCode,
    SafeZoneProfile,
    Stage51Config,
    safe_zone_for,
)
from app.composition.types import CaptionEvent

SpanInput: TypeAlias = tuple[int, float, float, int | None, int | None, str | None]

CAPTION_EVIDENCE_MISSING = PlanReasonCode.CAPTION_EVIDENCE_MISSING.value
CAPTION_COLLISION_UNRESOLVED = PlanReasonCode.CAPTION_COLLISION_UNRESOLVED.value
BIDI_CONTROL_NEUTRALIZED = PlanReasonCode.BIDI_CONTROL_NEUTRALIZED.value

#: Area fraction of the caption band a face may cover before the scene switches.
COLLISION_AREA_THRESHOLD = 0.15

_PUNCTUATION_BREAK_CHARS = frozenset(".!?,;:…،؛؟۔")
_SPAN_TIME_TOLERANCE = 0.5
_FLOAT_EPSILON = 1e-6
_WIDTH_SAFETY_FACTOR = 1.12
_LINE_HEIGHT_FACTOR = 1.25
_MIN_BAND_HEIGHT_PX = 1.0

_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_BIDI_CONTROL_CODEPOINTS = frozenset(range(0x202A, 0x202F)) | frozenset(range(0x2066, 0x206A))


@dataclass(frozen=True)
class SceneFaceBox:
    """One anonymous protected face box with its scene time range.

    Coordinates are display-normalized (origin top-left, x right, y down) and
    match :class:`app.composition.types.FaceDetection` semantics.
    """

    scene_index: int
    start: float
    end: float
    x: float
    y: float
    w: float
    h: float

    def as_dict(self) -> dict[str, object]:
        return {
            "scene_index": self.scene_index,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "w": round(self.w, 6),
            "h": round(self.h, 6),
        }


@dataclass(frozen=True)
class MissingEvidenceMarker:
    """One bound span that had no usable FINAL_CLIP word evidence."""

    block_index: int
    start: float
    end: float
    source_role: str | None = None
    reason: str = CAPTION_EVIDENCE_MISSING

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "source_role": self.source_role,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CaptionPlan:
    """Deterministic FINAL_CLIP caption plan (never a rendered asset)."""

    policy_version: str
    events: tuple[CaptionEvent, ...]
    missing_evidence: tuple[MissingEvidenceMarker, ...] = ()
    scene_zones: Mapping[int, str] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "events": [event.as_dict() for event in self.events],
            "missing_evidence": [marker.as_dict() for marker in self.missing_evidence],
            "scene_zones": {str(key): value for key, value in sorted(self.scene_zones.items())},
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class _Word:
    index: int
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class _Span:
    block_index: int
    start: float
    end: float
    word_start_index: int | None
    word_end_index: int | None
    source_role: str | None
    is_hero: bool


def _require_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"word field {field_name!r} must be numeric")
    return float(value)


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"word field {field_name!r} must be an integer")
    return value


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"word field {field_name!r} must be a string")
    return value


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, field_name)


def _optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name)


def _coerce_span(raw: object, hero_block_index: int | None) -> _Span:
    if isinstance(raw, tuple | list):
        if len(raw) != 6:
            raise ValueError("bound span must have six fields")
        values = list(raw)
        block_index = _require_int(values[0], "block_index")
        start = _require_float(values[1], "start")
        end = _require_float(values[2], "end")
        word_start = _optional_int(values[3], "word_start_index")
        word_end = _optional_int(values[4], "word_end_index")
        source_role = _optional_str(values[5], "source_role")
    else:
        block_index = _require_int(getattr(raw, "block_index"), "block_index")
        start = _require_float(getattr(raw, "start"), "start")
        end = _require_float(getattr(raw, "end"), "end")
        word_start = _optional_int(getattr(raw, "word_start_index", None), "word_start_index")
        word_end = _optional_int(getattr(raw, "word_end_index", None), "word_end_index")
        source_role = _optional_str(getattr(raw, "source_role", None), "source_role")
    is_hero = block_index == hero_block_index or (
        source_role is not None and source_role.upper() == "HERO"
    )
    return _Span(
        block_index=block_index,
        start=start,
        end=end,
        word_start_index=word_start,
        word_end_index=word_end,
        source_role=source_role,
        is_hero=is_hero,
    )


def _coerce_scene_face_box(raw: object) -> SceneFaceBox:
    if isinstance(raw, SceneFaceBox):
        return raw
    if isinstance(raw, Mapping):
        return SceneFaceBox(
            scene_index=_require_int(raw.get("scene_index"), "scene_index"),
            start=_require_float(raw.get("start"), "start"),
            end=_require_float(raw.get("end"), "end"),
            x=_require_float(raw.get("x"), "x"),
            y=_require_float(raw.get("y"), "y"),
            w=_require_float(raw.get("w"), "w"),
            h=_require_float(raw.get("h"), "h"),
        )
    if isinstance(raw, tuple | list):
        values = list(raw)
        if len(values) == 7:
            return SceneFaceBox(
                scene_index=_require_int(values[0], "scene_index"),
                start=_require_float(values[1], "start"),
                end=_require_float(values[2], "end"),
                x=_require_float(values[3], "x"),
                y=_require_float(values[4], "y"),
                w=_require_float(values[5], "w"),
                h=_require_float(values[6], "h"),
            )
        if len(values) == 4:
            detection = values[3]
            return SceneFaceBox(
                scene_index=_require_int(values[0], "scene_index"),
                start=_require_float(values[1], "start"),
                end=_require_float(values[2], "end"),
                x=_require_float(getattr(detection, "x"), "x"),
                y=_require_float(getattr(detection, "y"), "y"),
                w=_require_float(getattr(detection, "w"), "w"),
                h=_require_float(getattr(detection, "h"), "h"),
            )
        raise ValueError("scene face box must have four or seven fields")
    raise ValueError("unsupported scene face box value")


def _coerce_word(raw: Mapping[str, object]) -> _Word:
    return _Word(
        index=_require_int(raw.get("index"), "index"),
        text=_require_str(raw.get("text"), "text"),
        start=_require_float(raw.get("start"), "start"),
        end=_require_float(raw.get("end"), "end"),
    )


def index_words(word_timestamps: Sequence[Mapping[str, object]]) -> dict[int, _Word]:
    """Index FINAL_CLIP word timestamps by their stable word index."""

    words: dict[int, _Word] = {}
    for raw in word_timestamps:
        word = _coerce_word(raw)
        words[word.index] = word
    return words


def _contains_bidi_control(text: str) -> bool:
    return any(ord(character) in _BIDI_CONTROL_CODEPOINTS for character in text)


def _is_arabic_script(codepoint: int) -> bool:
    return any(low <= codepoint <= high for low, high in _ARABIC_RANGES)


def _char_advance(character: str, style: CaptionStyle) -> float:
    codepoint = ord(character)
    font_size = float(style.font_size)
    if _is_arabic_script(codepoint):
        return 0.55 * font_size
    if character.isspace():
        return 0.28 * font_size
    if character.isalnum():
        return 0.56 * font_size
    if unicodedata.category(character).startswith("P"):
        return 0.30 * font_size
    return 0.50 * font_size


def estimate_text_width(text: str, style: CaptionStyle) -> float:
    """Script-aware estimated rendered width with a safety factor."""

    advance = 0.0
    for character in text:
        advance += _char_advance(character, style)
    advance += style.spacing * max(0, len(text) - 1)
    return float(advance * _WIDTH_SAFETY_FACTOR)


def wrap_text(text: str, style: CaptionStyle, max_width_px: float) -> tuple[tuple[str, ...], bool]:
    """Greedy wrap at spaces; returns ``(lines, overflow)``.

    ``overflow`` is True only when the wrapped result needs more than
    ``style.max_lines`` lines. An over-wide single word occupies its own line
    (never split, never hyphenated).
    """

    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if estimate_text_width(candidate, style) <= max_width_px:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return tuple(lines), len(lines) > style.max_lines


def _spans_for_max_width(style: CaptionStyle, safe_zone: SafeZoneProfile) -> float:
    usable = float(OUTPUT_WIDTH - safe_zone.left_px() - safe_zone.right_px())
    return float(min(float(style.max_line_width_fraction) * OUTPUT_WIDTH, usable))


def _word_range_inside_span(word: _Word, span: _Span) -> bool:
    return (
        word.start >= span.start - _SPAN_TIME_TOLERANCE
        and word.end <= span.end + _SPAN_TIME_TOLERANCE
    )


def _select_span_words(span: _Span, words: Mapping[int, _Word]) -> tuple[_Word, ...] | None:
    if span.word_start_index is not None and span.word_end_index is not None:
        assert span.word_start_index <= span.word_end_index, (
            "bound span word indexes must be ordered"
        )
        selected: list[_Word] = []
        for index in range(span.word_start_index, span.word_end_index + 1):
            word = words.get(index)
            assert word is not None, "bound span references a missing word index"
            assert _word_range_inside_span(word, span), "word index falls outside its bound span"
            selected.append(word)
        return tuple(selected) if selected else None

    selected = [
        word
        for word in words.values()
        if word.start >= span.start - _FLOAT_EPSILON and word.end <= span.end + _FLOAT_EPSILON
    ]
    if not selected:
        return None
    selected.sort(key=lambda word: word.index)
    return tuple(selected)


def _ends_with_break_punctuation(text: str) -> bool:
    stripped = text.rstrip("\"'”’)]}")
    return bool(stripped) and stripped[-1] in _PUNCTUATION_BREAK_CHARS


def _has_boundary(left: _Word, right: _Word, style: CaptionStyle) -> bool:
    if _ends_with_break_punctuation(left.text):
        return True
    return bool((right.start - left.end) >= float(style.pause_split_seconds))


def _segment_span_words(
    words: Sequence[_Word],
    style: CaptionStyle,
    max_width_px: float,
) -> list[tuple[_Word, ...]]:
    fragments: list[tuple[_Word, ...]] = []
    total = len(words)
    start = 0
    while start < total:
        end = start
        fragment: list[_Word] = [words[start]]
        while end + 1 < total:
            left = words[end]
            right = words[end + 1]
            if _has_boundary(left, right, style):
                break
            candidate = fragment + [right]
            text = " ".join(word.text for word in candidate)
            _, overflow = wrap_text(text, style, max_width_px)
            duration = right.end - fragment[0].start
            if (
                overflow
                or len(candidate) > style.max_words_per_event
                or duration > style.max_event_duration
            ):
                break
            fragment.append(right)
            end += 1
        fragments.append(tuple(fragment))
        start = end + 1
    return _merge_short_fragments(fragments, style, max_width_px)


def _merge_short_fragments(
    fragments: Sequence[tuple[_Word, ...]],
    style: CaptionStyle,
    max_width_px: float,
) -> list[tuple[_Word, ...]]:
    result: list[list[_Word]] = [list(fragment) for fragment in fragments]
    changed = True
    while changed and len(result) > 1:
        changed = False
        for index in range(len(result) - 1):
            current = result[index]
            following = result[index + 1]
            duration = current[-1].end - current[0].start
            if duration >= style.min_event_duration:
                continue
            combined = current + following
            text = " ".join(word.text for word in combined)
            _, overflow = wrap_text(text, style, max_width_px)
            combined_duration = combined[-1].end - combined[0].start
            if (
                overflow
                or len(combined) > style.max_words_per_event
                or combined_duration > style.max_event_duration
            ):
                continue
            result[index] = combined
            del result[index + 1]
            changed = True
            break
    return [tuple(fragment) for fragment in result]


def _scene_intervals(
    scene_face_boxes: Sequence[SceneFaceBox],
) -> dict[int, tuple[float, float]]:
    intervals: dict[int, tuple[float, float]] = {}
    for box in scene_face_boxes:
        existing = intervals.get(box.scene_index)
        if existing is None:
            intervals[box.scene_index] = (box.start, box.end)
        else:
            intervals[box.scene_index] = (
                min(existing[0], box.start),
                max(existing[1], box.end),
            )
    return intervals


def _scene_for_event(
    start: float,
    end: float,
    intervals: Mapping[int, tuple[float, float]],
) -> int | None:
    midpoint = (start + end) / 2.0
    matches = [
        scene_index
        for scene_index, (scene_start, scene_end) in intervals.items()
        if scene_start <= midpoint <= scene_end
    ]
    if not matches:
        return None
    return min(matches)


def _faces_for_scene(
    scene_index: int | None,
    scene_face_boxes: Sequence[SceneFaceBox],
) -> list[SceneFaceBox]:
    if scene_index is None:
        return []
    return [box for box in scene_face_boxes if box.scene_index == scene_index]


def _caption_band(
    zone: CaptionPlacementZone,
    line_count: int,
    style: CaptionStyle,
    safe_zone: SafeZoneProfile,
) -> tuple[float, float, float, float]:
    height = max(_MIN_BAND_HEIGHT_PX, line_count * style.font_size * _LINE_HEIGHT_FACTOR)
    left = float(safe_zone.left_px())
    right = float(OUTPUT_WIDTH - safe_zone.right_px())
    if zone is CaptionPlacementZone.LOWER:
        bottom = float(OUTPUT_HEIGHT - (safe_zone.bottom_px() + safe_zone.caption_bottom_gap_px))
        top = bottom - height
    else:
        top = float(safe_zone.top_px() + safe_zone.caption_top_gap_px)
        bottom = top + height
    return (left, top, right, bottom)


def _overlap_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    overlap_w = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap_h = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return overlap_w * overlap_h


def _collision_fraction(
    band: tuple[float, float, float, float],
    faces: Sequence[SceneFaceBox],
    event_start: float,
    event_end: float,
) -> float:
    band_area = max(0.0, band[2] - band[0]) * max(0.0, band[3] - band[1])
    if band_area <= 0.0:
        return 0.0
    best = 0.0
    for face in faces:
        if face.end < event_start or face.start > event_end:
            continue
        rect = (
            face.x * OUTPUT_WIDTH,
            face.y * OUTPUT_HEIGHT,
            (face.x + face.w) * OUTPUT_WIDTH,
            (face.y + face.h) * OUTPUT_HEIGHT,
        )
        best = max(best, _overlap_area(band, rect) / band_area)
    return best


@dataclass
class _EventDraft:
    block_index: int
    word_start_index: int
    word_end_index: int
    start: float
    end: float
    text: str
    lines: tuple[str, ...]
    source_role: str | None
    is_hero: bool
    lower_fraction: float = 0.0
    upper_fraction: float = 0.0
    scene_index: int | None = None


def _trim_overlaps(drafts: list[_EventDraft]) -> None:
    drafts.sort(key=lambda draft: (draft.start, draft.word_start_index, draft.block_index))
    for current, following in zip(drafts, drafts[1:]):
        if current.end > following.start >= current.start:
            current.end = max(current.start, following.start)


def _decide_scene_zone(
    lower_fraction: float,
    upper_fraction: float,
    previous_zone: str | None,
) -> tuple[str, str, bool]:
    both_collide = (
        lower_fraction > COLLISION_AREA_THRESHOLD and upper_fraction > COLLISION_AREA_THRESHOLD
    )
    if both_collide:
        if upper_fraction < lower_fraction:
            return (CaptionPlacementZone.UPPER.value, CAPTION_COLLISION_UNRESOLVED, True)
        return (CaptionPlacementZone.LOWER.value, CAPTION_COLLISION_UNRESOLVED, True)

    if previous_zone == CaptionPlacementZone.UPPER.value:
        if lower_fraction <= 0.0:
            return (
                CaptionPlacementZone.LOWER.value,
                "SCENE_COLLISION_FREE_HYSTERESIS_LOWER",
                False,
            )
        return (CaptionPlacementZone.UPPER.value, "SCENE_HYSTERESIS_HOLD_UPPER", False)

    if lower_fraction > COLLISION_AREA_THRESHOLD:
        return (CaptionPlacementZone.UPPER.value, "SCENE_COLLISION_SWITCH_UPPER", False)
    return (CaptionPlacementZone.LOWER.value, "DEFAULT_LOWER_ZONE", False)


def _event_id(block_index: int, word_start_index: int, word_end_index: int) -> str:
    return f"caption-{block_index}-{word_start_index}-{word_end_index}"


def _build_event_drafts(
    spans: Sequence[object],
    words: Mapping[int, _Word],
    style: CaptionStyle,
    max_width_px: float,
    safe_zone: SafeZoneProfile,
    scene_face_boxes: Sequence[SceneFaceBox],
    hero_block_index: int | None,
) -> tuple[list[_EventDraft], list[MissingEvidenceMarker], list[str]]:
    drafts: list[_EventDraft] = []
    markers: list[MissingEvidenceMarker] = []
    reasons: set[str] = set()
    intervals = _scene_intervals(scene_face_boxes)

    for raw_span in spans:
        span = _coerce_span(raw_span, hero_block_index)
        block_index = span.block_index
        source_role = span.source_role
        selected = _select_span_words(span, words)
        if selected is None:
            markers.append(
                MissingEvidenceMarker(
                    block_index=block_index,
                    start=span.start,
                    end=span.end,
                    source_role=source_role,
                )
            )
            reasons.add(CAPTION_EVIDENCE_MISSING)
            continue

        for fragment in _segment_span_words(selected, style, max_width_px):
            text = " ".join(word.text for word in fragment)
            lines, _ = wrap_text(text, style, max_width_px)
            event_start = fragment[0].start
            event_end = max(fragment[-1].end, event_start + style.min_event_duration)
            event_end = min(
                event_end,
                event_start + style.max_event_duration,
                span.end + style.tail_after_last_word,
            )
            event_end = max(event_end, event_start)
            draft = _EventDraft(
                block_index=block_index,
                word_start_index=fragment[0].index,
                word_end_index=fragment[-1].index,
                start=event_start,
                end=event_end,
                text=text,
                lines=lines,
                source_role=source_role,
                is_hero=span.is_hero,
            )
            drafts.append(draft)
            if _contains_bidi_control(text):
                reasons.add(BIDI_CONTROL_NEUTRALIZED)

    for draft in drafts:
        draft.scene_index = _scene_for_event(draft.start, draft.end, intervals)
        faces = _faces_for_scene(draft.scene_index, scene_face_boxes)
        lower_band = _caption_band(CaptionPlacementZone.LOWER, len(draft.lines), style, safe_zone)
        upper_band = _caption_band(CaptionPlacementZone.UPPER, len(draft.lines), style, safe_zone)
        draft.lower_fraction = _collision_fraction(lower_band, faces, draft.start, draft.end)
        draft.upper_fraction = _collision_fraction(upper_band, faces, draft.start, draft.end)

    return drafts, markers, sorted(reasons)


def _resolve_scene_zones(
    drafts: Sequence[_EventDraft],
    previous_scene_zones: Mapping[int, str],
) -> tuple[dict[int, str], dict[int, tuple[str, float, float, bool]]]:
    scenes: dict[int, list[_EventDraft]] = {}
    for draft in drafts:
        if draft.scene_index is None:
            continue
        scenes.setdefault(draft.scene_index, []).append(draft)

    zones: dict[int, str] = {}
    details: dict[int, tuple[str, float, float, bool]] = {}
    for scene_index in sorted(scenes):
        scene_events = scenes[scene_index]
        lower = max((draft.lower_fraction for draft in scene_events), default=0.0)
        upper = max((draft.upper_fraction for draft in scene_events), default=0.0)
        zone, reason, both = _decide_scene_zone(lower, upper, previous_scene_zones.get(scene_index))
        zones[scene_index] = zone
        details[scene_index] = (reason, lower, upper, both)
    return zones, details


def _emit_events(
    drafts: list[_EventDraft],
    previous_scene_zones: Mapping[int, str],
) -> tuple[list[CaptionEvent], dict[int, str], set[str]]:
    _trim_overlaps(drafts)
    zones, details = _resolve_scene_zones(drafts, previous_scene_zones)

    events: list[CaptionEvent] = []
    reasons: set[str] = set()
    for draft in drafts:
        scene_index = draft.scene_index
        scene_key = scene_index if scene_index is not None else -1
        scene_zone = zones.get(scene_key, CaptionPlacementZone.LOWER.value)
        scene_reason, _, _, both = details.get(
            scene_key,
            ("DEFAULT_LOWER_ZONE", 0.0, 0.0, False),
        )
        zone = scene_zone
        reason = scene_reason
        if both:
            reasons.add(CAPTION_COLLISION_UNRESOLVED)
            if draft.is_hero:
                zone = (
                    CaptionPlacementZone.UPPER.value
                    if draft.upper_fraction < draft.lower_fraction
                    else CaptionPlacementZone.LOWER.value
                )
                reason = CAPTION_COLLISION_UNRESOLVED
        evidence: dict[str, object] = {
            "scene_index": scene_index,
            "lower_fraction": round(draft.lower_fraction, 6),
            "upper_fraction": round(draft.upper_fraction, 6),
            "threshold": COLLISION_AREA_THRESHOLD,
        }
        events.append(
            CaptionEvent(
                event_id=_event_id(draft.block_index, draft.word_start_index, draft.word_end_index),
                block_index=draft.block_index,
                word_start_index=draft.word_start_index,
                word_end_index=draft.word_end_index,
                start=round(draft.start, 4),
                end=round(draft.end, 4),
                text=draft.text,
                lines=draft.lines,
                placement_zone=zone,
                placement_reason=reason,
                collision_evidence=evidence,
            )
        )
    events.sort(key=lambda event: (event.start, event.word_start_index, event.block_index))
    return events, zones, reasons


def build_caption_plan(
    spans: Sequence[object],
    word_timestamps: Sequence[Mapping[str, object]],
    style: CaptionStyle,
    config: Stage51Config,
    *,
    scene_face_boxes: Sequence[object] = (),
    previous_scene_zones: Mapping[int, str] | None = None,
    hero_block_index: int | None = None,
) -> CaptionPlan:
    """Build the deterministic FINAL_CLIP caption plan for selected spans.

    ``spans`` accepts the documented six-field tuple
    ``(block_index, start, end, word_start_index, word_end_index, source_role)``
    or any object exposing the same attributes (for example ``BoundSpan``).
    ``scene_face_boxes`` accepts :class:`SceneFaceBox`, a seven-field tuple, a
    ``(scene_index, start, end, face_detection)`` tuple, or an equivalent mapping.
    """

    safe_zone = safe_zone_for(config.safe_zone_profile_key)
    max_width_px = _spans_for_max_width(style, safe_zone)
    words = index_words(word_timestamps)
    boxes = tuple(_coerce_scene_face_box(raw) for raw in scene_face_boxes)
    drafts, markers, reasons = _build_event_drafts(
        spans,
        words,
        style,
        max_width_px,
        safe_zone,
        boxes,
        hero_block_index,
    )
    events, zones, event_reasons = _emit_events(
        drafts,
        previous_scene_zones or {},
    )
    all_reasons = sorted({*reasons, *event_reasons})
    return CaptionPlan(
        policy_version=CAPTION_LAYOUT_POLICY_VERSION,
        events=tuple(events),
        missing_evidence=tuple(markers),
        scene_zones=zones,
        reason_codes=tuple(all_reasons),
    )


__all__ = [
    "BIDI_CONTROL_NEUTRALIZED",
    "CAPTION_COLLISION_UNRESOLVED",
    "CAPTION_EVIDENCE_MISSING",
    "COLLISION_AREA_THRESHOLD",
    "CaptionPlan",
    "MissingEvidenceMarker",
    "SceneFaceBox",
    "SpanInput",
    "build_caption_plan",
    "estimate_text_width",
    "index_words",
    "wrap_text",
]
