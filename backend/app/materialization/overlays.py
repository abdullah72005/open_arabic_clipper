"""Deterministic Stage 6 authored ASS overlay and measured readability evidence.

Stage 6 writes a separate ``authored.ass`` in *output* time, applied by the
renderer after source video concatenation. It never fabricates word timing or
karaoke: spoken authored copy is a static card/phrase display, and it reuses the
shared ASS escaping/bidi-safety helpers so provider text is always rendered
literally.

Layout ownership: authored copy is laid out inside the existing shared
Shorts/Reels safe-area rectangle, centred within its usable area, using the
installed Arabic-capable caption font and the existing caption sizing/width
primitives. At most three lines are allowed. Copy is never silently truncated or
shrunk to fit: an overlong/unreadable card records an explicit ``FAIL`` finding
so the materialization cannot become review-ready.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

from app.composition.ass import escape_ass_text_with_evidence
from app.composition.policy import SafeZoneProfile, safe_zone_for
from app.materialization.types import AuthoredTimeline, TimelineSegment

AUTHORED_ASS_POLICY_VERSION = "stage6-authored-ass-v2"
CARD_KIND = "CARD"
ANNOTATION_KIND = "TEXTUAL_ANNOTATION"

MAX_AUTHORED_LINES = 3
READING_CHARS_PER_SECOND = 14.0
DEFAULT_AUTHORED_FONT_FAMILY = "Noto Sans Arabic"
DEFAULT_AUTHORED_FONT_SIZE = 88
#: Conservative advance-width estimate (fraction of font size) for the installed
#: Arabic-capable font. Deliberately generous so the measured width is a safe
#: upper bound rather than an optimistic guess.
_CHAR_WIDTH_FACTOR = 0.55
_LINE_HEIGHT_FACTOR = 1.25
_READING_TOLERANCE_SECONDS = 0.05

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


@dataclass(frozen=True)
class AuthoredAssResult:
    """One deterministic authored ASS asset plus measured readability evidence."""

    data: bytes
    sha256: str
    event_count: int
    line_count: int
    policy_version: str
    findings: tuple[Mapping[str, object], ...] = ()
    layout: tuple[Mapping[str, object], ...] = ()

    @property
    def failed(self) -> bool:
        return any(str(finding.get("level")) == "FAIL" for finding in self.findings)


def _format_ass_time(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


def _wrap(text: str, max_chars: int) -> list[str]:
    """Word-wrap without splitting tokens or truncating content."""

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


def _max_chars_for_width(usable_width: int, font_size: int) -> int:
    char_width = max(1.0, font_size * _CHAR_WIDTH_FACTOR)
    return max(1, int(math.floor(usable_width / char_width)))


def _layout_finding(
    level: str,
    reason: str,
    segment: TimelineSegment,
    evidence: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        "dimension": "VISUAL_CAPTION_READABILITY",
        "level": level,
        "block_index": segment.plan_block_index,
        "reason_codes": [reason],
        "evidence": dict(evidence),
    }


def build_authored_ass(
    timeline: AuthoredTimeline,
    *,
    play_res_x: int,
    play_res_y: int,
    safe_zone: SafeZoneProfile | None = None,
    font_family: str = DEFAULT_AUTHORED_FONT_FAMILY,
    font_size: int = DEFAULT_AUTHORED_FONT_SIZE,
    max_chars_per_line: int | None = None,
) -> AuthoredAssResult:
    """Build the authored ASS bytes, event/line counts, and measured findings."""

    zone = safe_zone or safe_zone_for("SHORTS_VERTICAL_SAFE_ZONE_V1")
    left = zone.left_px()
    right = zone.right_px()
    top = zone.top_px()
    bottom = zone.bottom_px()
    usable_width = max(1, play_res_x - left - right)
    usable_height = max(1, play_res_y - top - bottom)
    center_x = left + usable_width / 2.0
    center_y = top + usable_height / 2.0
    max_chars = max_chars_per_line or _max_chars_for_width(usable_width, font_size)

    findings: list[Mapping[str, object]] = []
    layout: list[Mapping[str, object]] = []
    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        _STYLE_FORMAT,
        (
            f"Style: Card,{font_family},{font_size},&H00FFFFFF,&H000000FF,"
            f"&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,0,5,"
            f"{left},{right},{top},1"
        ),
        "",
        "[Events]",
        _EVENT_FORMAT,
    ]
    total_lines = 0
    event_count = 0
    authored_kinds = {CARD_KIND, ANNOTATION_KIND, "AUTHORED_CARD"}
    for segment in timeline.segments:
        if segment.kind not in authored_kinds:
            continue
        text = (segment.text or "").strip()
        if not text:
            findings.append(
                {
                    "dimension": "VISUAL_CAPTION_READABILITY",
                    "level": "FAIL",
                    "block_index": segment.plan_block_index,
                    "reason_codes": ["MISSING_REQUIRED_VALUE"],
                    "evidence": {"delivery": "STANDALONE_CARD"},
                }
            )
            continue
        escaped, reasons = escape_ass_text_with_evidence(text)
        wrapped = _wrap(escaped, max_chars)
        duration = max(0.0, segment.output_end - segment.output_start)
        reading_seconds = len(text) / READING_CHARS_PER_SECOND
        widest = max((len(line) for line in wrapped), default=0)
        estimated_width = widest * font_size * _CHAR_WIDTH_FACTOR
        estimated_height = len(wrapped) * font_size * _LINE_HEIGHT_FACTOR
        evidence: dict[str, object] = {
            "font_family": font_family,
            "font_size": font_size,
            "safe_area": {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "usable_width": usable_width,
                "usable_height": usable_height,
            },
            "characters": len(text),
            "wrapped_lines": len(wrapped),
            "estimated_width": round(estimated_width, 3),
            "estimated_height": round(estimated_height, 3),
            "duration_seconds": round(duration, 6),
            "reading_seconds": round(reading_seconds, 6),
            "delivery": "STANDALONE_CARD",
            "delivery_rationale": (
                "Stage 5.2 supports only card/source units; a source-hosted "
                "annotation payload is not executable, so authored text is "
                "delivered as a standalone readable card."
            ),
        }
        layout.append({"block_index": segment.plan_block_index, **evidence})

        if reasons:
            findings.append(
                {
                    "dimension": "authored_ass_bidi",
                    "level": "CONCERN",
                    "block_index": segment.plan_block_index,
                    "reason_codes": list(reasons),
                    "evidence": dict(evidence),
                }
            )
        if len(wrapped) > MAX_AUTHORED_LINES:
            findings.append(
                _layout_finding("FAIL", "AUTHORED_TEXT_REQUIRES_SHORTENING", segment, evidence)
            )
        if estimated_width > usable_width + 1e-6 or estimated_height > usable_height + 1e-6:
            findings.append(
                _layout_finding("FAIL", "AUTHORED_TEXT_EXCEEDS_SAFE_AREA", segment, evidence)
            )
        if (
            not segment.narration_chunk_id
            and reading_seconds > duration + _READING_TOLERANCE_SECONDS
        ):
            findings.append(
                _layout_finding("FAIL", "AUTHORED_READING_BUDGET_INSUFFICIENT", segment, evidence)
            )
        if not any(
            finding.get("block_index") == segment.plan_block_index
            and finding.get("level") == "FAIL"
            for finding in findings
        ):
            findings.append(
                _layout_finding("SATISFIED", "AUTHORED_LAYOUT_READABLE", segment, evidence)
            )

        total_lines += len(wrapped)
        event_count += 1
        display = "\\N".join(wrapped)
        lines.append(
            f"Dialogue: 0,{_format_ass_time(segment.output_start)},"
            f"{_format_ass_time(segment.output_end)},Card,,0,0,0,,"
            f"{{\\an5\\pos({center_x:.0f},{center_y:.0f})}}{display}"
        )

    data = ("\n".join(lines) + "\n").encode("utf-8")
    return AuthoredAssResult(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        event_count=event_count,
        line_count=total_lines,
        policy_version=AUTHORED_ASS_POLICY_VERSION,
        findings=tuple(findings),
        layout=tuple(layout),
    )


__all__ = [
    "AUTHORED_ASS_POLICY_VERSION",
    "ANNOTATION_KIND",
    "CARD_KIND",
    "DEFAULT_AUTHORED_FONT_FAMILY",
    "DEFAULT_AUTHORED_FONT_SIZE",
    "MAX_AUTHORED_LINES",
    "READING_CHARS_PER_SECOND",
    "AuthoredAssResult",
    "build_authored_ass",
]
