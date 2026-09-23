"""Deterministic ASS subtitle serialization for Stage 5.1.

One :class:`~app.composition.captions.CaptionPlan` becomes exactly one
deterministic AdvSubStation (ASS) document. The module is pure except for
:func:`write_ass_asset`, which performs one atomic storage write. No rendering
happens here; the real libass render is exercised by the render tests.

Escaping is proven against the installed libass rather than assumed:

- a source ``{`` is emitted as ``\\{`` and a source ``}`` as ``\\}``, which
  libass renders as literal braces and never executes as an override block;
- a source backslash is emitted literally, except that when it is immediately
  followed by ``N``/``n``/``h`` (libass' newline/space tags) a zero-width word
  joiner is inserted so the pair renders literally instead of changing layout;
- the characters libass treats specially therefore can never start an override
  block or a drawing command from source text.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from app.composition import bidi
from app.composition.captions import CaptionPlan
from app.composition.policy import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    CaptionPlacementZone,
    CaptionStyle,
    PlanReasonCode,
    SafeZoneProfile,
)
from app.composition.types import CaptionEvent
from app.services.storage import StorageService

_WORD_JOINER = "\u2060"
_BACKSLASH_FOLLOWERS = frozenset("Nnh")
_BIDI_CONTROL_CODEPOINTS = frozenset(range(0x202A, 0x202F)) | frozenset(range(0x2066, 0x206A))

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


def escape_ass_text(text: str) -> str:
    """Escape source text so libass renders it literally, never as overrides."""

    return escape_ass_text_with_evidence(text)[0]


def escape_ass_text_with_evidence(text: str) -> tuple[str, tuple[str, ...]]:
    """Escape source text and report any recorded neutralization reason codes."""

    characters: list[str] = []
    bidi_found = False
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        codepoint = ord(character)
        if codepoint in _BIDI_CONTROL_CODEPOINTS:
            bidi_found = True
            index += 1
            continue
        if character in "\t\n\r\v\f" or codepoint in (0x2028, 0x2029):
            characters.append(" ")
            index += 1
            continue
        if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
            index += 1
            continue
        if character == "\\":
            characters.append("\\")
            following = text[index + 1] if index + 1 < length else ""
            if following in _BACKSLASH_FOLLOWERS:
                characters.append(_WORD_JOINER)
            index += 1
            continue
        if character == "{":
            characters.append("\\{")
            index += 1
            continue
        if character == "}":
            characters.append("\\}")
            index += 1
            continue
        characters.append(character)
        index += 1

    reasons: tuple[str, ...] = ()
    if bidi_found:
        reasons = (PlanReasonCode.BIDI_CONTROL_NEUTRALIZED.value,)
    return "".join(characters), reasons


def contains_bidi_control(text: str) -> bool:
    """Return True when ``text`` carries any Unicode bidi control character."""

    return any(ord(character) in _BIDI_CONTROL_CODEPOINTS for character in text)


def neutralize_text(text: str) -> tuple[str, tuple[str, ...]]:
    """Alias used by callers that only need the derived-asset neutralization."""

    return escape_ass_text_with_evidence(text)


def _format_ass_time(seconds: float) -> str:
    total_centiseconds = round(max(seconds, 0.0) * 100)
    centiseconds = total_centiseconds % 100
    total_seconds = total_centiseconds // 100
    second = total_seconds % 60
    minute = (total_seconds // 60) % 60
    hour = total_seconds // 3600
    return f"{hour}:{minute:02d}:{second:02d}.{centiseconds:02d}"


def _style_line(
    name: str,
    alignment: int,
    vertical_margin: float,
    style: CaptionStyle,
    safe_zone: SafeZoneProfile,
) -> str:
    return (
        f"Style: {name},{style.font_family},{style.font_size},{style.primary_color},"
        f"{style.secondary_color},{style.outline_color},&H00000000,"
        f"0,0,0,0,100,100,{style.spacing},0,1,{style.outline_width},{style.shadow},"
        f"{alignment},{safe_zone.left_px()},{safe_zone.right_px()},{int(round(vertical_margin))},1"
    )


def _event_lines(event: CaptionEvent) -> tuple[str, ...]:
    if event.lines:
        return tuple(str(line) for line in event.lines)
    return (str(event.text),)


def _token_lines(event: CaptionEvent) -> tuple[list[str], list[list[int]]]:
    """Map canonical tokens to their logical lines as global token indexes.

    Canonical ``event.text`` is never rewritten; each line is a contiguous slice
    of the same token order, so token index ``i`` always matches
    ``event.word_timings[i]``.
    """

    tokens = event.text.split(" ")
    raw_lines = [str(line) for line in event.lines] if event.lines else [event.text]
    if " ".join(raw_lines) == event.text:
        counts = [len(line.split(" ")) for line in raw_lines]
        if sum(counts) == len(tokens):
            lines: list[list[int]] = []
            cursor = 0
            for count in counts:
                lines.append(list(range(cursor, cursor + count)))
                cursor += count
            return tokens, lines
    return tokens, [list(range(len(tokens)))]


def _visual_line_indexes(tokens: list[str], line: list[int], base: str) -> list[int]:
    """Reorder one logical line's token indexes into visual left-to-right order."""

    order = bidi.visual_order([tokens[index] for index in line], base)
    return [line[position] for position in order]


def _event_text(event: CaptionEvent, style: CaptionStyle, active_index: int | None = None) -> str:
    """Render one event with identical per-token scaffolding in every state.

    Every token is wrapped in the same ``{\\c<color>}token{\\c<primary>}``
    structure regardless of which word is active (or none). Only the color value
    changes between the active and inactive tokens, so the override-tag
    boundaries, BiDi run segmentation, glyph metrics, wrapping, and block
    position are structurally identical across all active-word states and the
    static state. Tokens are emitted in the derived visual order for the event's
    base direction; canonical text is never rewritten.
    """

    tokens, lines = _token_lines(event)
    base = bidi.paragraph_direction(event.text)
    restore = f"{{\\c{style.primary_color}&}}"
    rendered_lines: list[str] = []
    for line in lines:
        rendered_tokens: list[str] = []
        for index in _visual_line_indexes(tokens, line, base):
            color = style.active_color if index == active_index else style.primary_color
            rendered_tokens.append(f"{{\\c{color}&}}{escape_ass_text(tokens[index])}{restore}")
        rendered_lines.append(" ".join(rendered_tokens))
    return _placement_override(event) + "\\N".join(rendered_lines)


#: A spoken word must be active this long to be emphasized; shorter spans
#: degrade the whole event to one static phrase line rather than flicker.
_ACTIVE_MIN_STATE_SECONDS = 0.05
_TIMING_TOLERANCE = 1e-4


def _placement_override(event: CaptionEvent) -> str:
    if event.placement_zone == CaptionPlacementZone.UPPER.value:
        return "{\\an8}"
    return "{\\an2}"


def _dynamic_event_text(event: CaptionEvent, style: CaptionStyle, active_index: int) -> str:
    """Render one event with the active word colored, structure held constant.

    Delegates to :func:`_event_text` so the active-word state has exactly the
    same per-token wrapper structure as every other state; only the active
    token's color value differs.
    """

    return _event_text(event, style, active_index)


def _active_word_states(event: CaptionEvent, style: CaptionStyle) -> list[tuple[float, float, str]]:
    """Resolve one stationary Dialogue state per spoken word.

    Returns ``[(start, end, text), ...]`` tiling ``[event.start, event.end]``
    with non-overlapping intervals; the active word changes at the exact
    FINAL_CLIP word boundary. Emphasis degrades to a single static phrase
    state whenever per-word timing is missing, degenerate, out of range, or too
    short to render cleanly, so unreliable timing is never turned into
    fabricated highlighting.
    """

    words = event.word_timings
    static = [(event.start, event.end, _event_text(event, style))]
    if not style.active_emphasis or len(words) < 2:
        return static
    if len(words) != len(event.text.split(" ")):
        return static
    if any(not math.isfinite(word.start) or not math.isfinite(word.end) for word in words):
        return static
    if any(word.end <= word.start for word in words):
        return static
    if abs(words[0].start - event.start) > _TIMING_TOLERANCE:
        return static

    boundaries: list[float] = [event.start, *(word.start for word in words[1:]), event.end]
    for earlier, later in zip(boundaries, boundaries[1:]):
        if later - earlier < _ACTIVE_MIN_STATE_SECONDS:
            return static
    return [
        (boundaries[index], boundaries[index + 1], _dynamic_event_text(event, style, index))
        for index in range(len(words))
    ]


def _style_name(event: CaptionEvent) -> str:
    if event.placement_zone == CaptionPlacementZone.UPPER.value:
        return "CaptionUpper"
    return "CaptionLower"


def build_ass_text(plan: CaptionPlan, style: CaptionStyle, safe_zone: SafeZoneProfile) -> str:
    """Build the deterministic ASS document text for one caption plan."""

    bottom_margin = safe_zone.bottom_px() + safe_zone.caption_bottom_gap_px
    top_margin = safe_zone.top_px() + safe_zone.caption_top_gap_px
    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {OUTPUT_WIDTH}",
        f"PlayResY: {OUTPUT_HEIGHT}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        _STYLE_FORMAT,
        _style_line("CaptionLower", style.alignment_lower, bottom_margin, style, safe_zone),
        _style_line("CaptionUpper", style.alignment_upper, top_margin, style, safe_zone),
        "",
        "[Events]",
        _EVENT_FORMAT,
    ]

    ordered = sorted(
        plan.events,
        key=lambda event: (event.start, event.word_start_index, event.block_index),
    )
    for event in ordered:
        style_name = _style_name(event)
        for state_start, state_end, state_text in _active_word_states(event, style):
            lines.append(
                "Dialogue: 0,"
                f"{_format_ass_time(state_start)},{_format_ass_time(state_end)},"
                f"{style_name},,0,0,0,,{state_text}"
            )
    return "\n".join(lines) + "\n"


def serialize_ass(plan: CaptionPlan, style: CaptionStyle, safe_zone: SafeZoneProfile) -> bytes:
    """Serialize a caption plan to UTF-8 ASS bytes (LF, no BOM), deterministically."""

    return build_ass_text(plan, style, safe_zone).encode("utf-8")


def ass_asset_relative_path(source_id: str, plan_fingerprint: str) -> str:
    """Return the canonical storage-relative ASS asset path for a plan."""

    return f"sources/{source_id}/visual-composition/{plan_fingerprint}.ass"


def write_ass_asset(
    storage: StorageService,
    source_id: str,
    plan_fingerprint: str,
    data: bytes,
) -> tuple[Path, str]:
    """Atomically write the ASS asset under the source and return path + sha256.

    An existing asset with an identical digest is reused without rewriting.
    """

    digest = hashlib.sha256(data).hexdigest()
    directory = storage.source_directory(source_id) / "visual-composition"
    destination = directory / f"{plan_fingerprint}.ass"
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
        return destination, digest
    written = storage.atomic_write(destination, [data])
    return written, digest


def aggregate_reason_codes(plan: CaptionPlan) -> tuple[str, ...]:
    """Return the deterministic union of plan reason codes and neutralizations."""

    reasons = set(plan.reason_codes)
    for event in plan.events:
        for line in _event_lines(event):
            _, found = escape_ass_text_with_evidence(line)
            reasons.update(found)
    return tuple(sorted(reasons))


__all__ = [
    "aggregate_reason_codes",
    "ass_asset_relative_path",
    "build_ass_text",
    "contains_bidi_control",
    "escape_ass_text",
    "escape_ass_text_with_evidence",
    "neutralize_text",
    "serialize_ass",
    "write_ass_asset",
]
