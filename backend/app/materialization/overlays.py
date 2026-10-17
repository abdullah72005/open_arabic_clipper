"""Deterministic Stage 6 authored ASS overlay and readability evidence.

Stage 6 writes a separate ``authored.ass`` in *output* time, applied by the
renderer after source video concatenation. It never fabricates word timing or
karaoke: spoken authored copy is a static card/phrase display, and it reuses the
shared ASS escaping/bidi-safety helpers so provider text is always rendered
literally. The layout is versioned and deliberately restrained (stage-safe
margins, at most three wrapped lines).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.composition.ass import escape_ass_text_with_evidence
from app.materialization.types import AuthoredTimeline, TimelineSegment

AUTHORED_ASS_POLICY_VERSION = "stage6-authored-ass-v1"
CARD_KIND = "CARD"
ANNOTATION_KIND = "TEXTUAL_ANNOTATION"

_MAX_LINES = 3
_ALIGNMENT_CARD = 2  # bottom-center
_ALIGNMENT_ANNOTATION = 8  # top-center

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


@dataclass(frozen=True)
class AuthoredAssResult:
    """One deterministic authored ASS asset plus readability evidence."""

    data: bytes
    sha256: str
    event_count: int
    line_count: int
    policy_version: str
    findings: tuple[Mapping[str, object], ...] = ()


def _format_ass_time(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


def _wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _authored_events(
    segments: Sequence[TimelineSegment],
) -> list[tuple[TimelineSegment, str, int]]:
    events: list[tuple[TimelineSegment, str, int]] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        if segment.kind == ANNOTATION_KIND:
            events.append((segment, text, _ALIGNMENT_ANNOTATION))
        else:
            # Standalone cards (and any authored text delivery) are bottom-center.
            events.append((segment, text, _ALIGNMENT_CARD))
    return events


def build_authored_ass(
    timeline: AuthoredTimeline,
    *,
    play_res_x: int,
    play_res_y: int,
    font_family: str = "DejaVu Sans",
    font_size: int = 48,
    max_chars_per_line: int = 22,
) -> AuthoredAssResult:
    """Build the authored ASS bytes, event/line counts, and readability findings."""

    events = _authored_events(timeline.segments)
    findings: list[Mapping[str, object]] = []
    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        _STYLE_FORMAT,
        (
            f"Style: Card,{font_family},{font_size},&H00FFFFFF,&H000000FF,"
            f"&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,0,{_ALIGNMENT_CARD},"
            f"{int(play_res_x * 0.08)},{int(play_res_x * 0.08)},{int(play_res_y * 0.10)},1"
        ),
        "",
        "[Events]",
        _EVENT_FORMAT,
    ]
    total_lines = 0
    for segment, text, alignment in events:
        escaped, reasons = escape_ass_text_with_evidence(text)
        wrapped = _wrap(escaped, max_chars_per_line)
        if reasons:
            findings.append(
                {
                    "dimension": "authored_ass_bidi",
                    "level": "CONCERN",
                    "block_index": segment.plan_block_index,
                    "reason_codes": list(reasons),
                }
            )
        if len(wrapped) > _MAX_LINES:
            findings.append(
                {
                    "dimension": "authored_readability",
                    "level": "FAIL",
                    "block_index": segment.plan_block_index,
                    "reason_codes": ["AUTHORED_TEXT_REQUIRES_SHORTENING"],
                    "evidence": {"wrapped_lines": len(wrapped)},
                }
            )
            wrapped = wrapped[:_MAX_LINES]
        total_lines += len(wrapped)
        display = "\\N".join(wrapped)
        alignment_override = "" if alignment == _ALIGNMENT_CARD else f"{{\\an{alignment}}}"
        lines.append(
            f"Dialogue: 0,{_format_ass_time(segment.output_start)},"
            f"{_format_ass_time(segment.output_end)},Card,,0,0,0,,"
            f"{alignment_override}{display}"
        )

    data = ("\n".join(lines) + "\n").encode("utf-8")
    return AuthoredAssResult(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        event_count=len(events),
        line_count=total_lines,
        policy_version=AUTHORED_ASS_POLICY_VERSION,
        findings=tuple(findings),
    )


__all__ = [
    "AUTHORED_ASS_POLICY_VERSION",
    "ANNOTATION_KIND",
    "CARD_KIND",
    "AuthoredAssResult",
    "build_authored_ass",
]
