"""Real libass render proofs for Stage 5.1 Arabic/English/mixed captions.

These tests do not inspect strings or a frontend ``<bdi>`` element. They build an
ASS document with the production serializer, render it through ffmpeg's actual
libass ``ass`` filter against an lavfi source, decode the PNG, and assert ink
plus byte determinism. A naive character-reversed document is rendered as a
negative control.

The suite skips only when ffmpeg, the libass ``ass`` filter, or a usable
installed font is genuinely absent. In the project Docker test image they run.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from app.composition.ass import serialize_ass
from app.composition.bidi import active_emphasis_safe, paragraph_direction, visual_tokens
from app.composition.captions import (
    CaptionPlan,
    _spans_for_max_width,
    build_caption_plan,
    wrap_text,
)
from app.composition.policy import CaptionStyle, SafeZoneProfile, Stage51Config, safe_zone_for
from app.composition.types import CaptionEvent, CaptionWordTiming

WIDTH = 1080
HEIGHT = 1920
ARABIC_CHARSET = "0627"

REPRESENTATIVE = "أنا كنت content creator لمدة سنتين"

FIXTURES: dict[str, str] = {
    "arabic_only": "أنا كنت سعيدا جدا اليوم",
    "english_only": "hello world this is a test",
    "arabic_with_embedded_english": "أنا كنت content creator لمدة سنتين",
    "arabic_english_numbers": "أنا عمري 30 سنة and I code 5 days",
    "english_with_arabic_phrase": "I said مرحبا to everyone",
    "mixed_punctuation": "hello، world! أنا هنا؟ yes.",
    "representative": REPRESENTATIVE,
}

_BIDI_CONTROL_CODEPOINTS = frozenset(range(0x202A, 0x202F)) | frozenset(range(0x2066, 0x206A))
_WORD_JOINER = "\u2060"


def _capture(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def _installed_families() -> set[str]:
    families: set[str] = set()
    for line in _capture(["fc-list", "--format", "%{family}\n"]).splitlines():
        for family in line.split(","):
            stripped = family.strip()
            if stripped:
                families.add(stripped)
    return families


def _resolve_font(preferred: str) -> str | None:
    if shutil.which("fc-match") is None:
        return None
    installed = _installed_families()
    if not installed:
        return None
    matched = _capture(["fc-match", "-f", "%{family}", preferred]).strip()
    if matched and matched in installed:
        return matched
    arabic_capable = _capture(["fc-match", "-f", "%{family}", f":charset={ARABIC_CHARSET}"]).strip()
    if arabic_capable and arabic_capable in installed:
        return arabic_capable
    if "DejaVu Sans" in installed:
        return "DejaVu Sans"
    return sorted(installed)[0] if installed else None


def _has_ass_filter() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    return (
        re.search(r"(?m)^\s*[. ]*ass\s+\S+->\S+", _capture(["ffmpeg", "-hide_banner", "-filters"]))
        is not None
    )


def _detect_skip_reason() -> str | None:
    if shutil.which("ffmpeg") is None:
        return "ffmpeg is not installed"
    if not _has_ass_filter():
        return "ffmpeg was built without the libass 'ass' filter"
    if shutil.which("fc-match") is None or shutil.which("fc-list") is None:
        return "fontconfig tools are unavailable"
    if _resolve_font("Noto Sans Arabic") is None:
        return "no usable installed font was found"
    return None


SKIP_REASON = _detect_skip_reason()
FONT_FAMILY = _resolve_font("Noto Sans Arabic") or "DejaVu Sans"

pytestmark = pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "render unavailable")


def _safe_zone() -> SafeZoneProfile:
    return safe_zone_for("SHORTS_VERTICAL_SAFE_ZONE_V1")


def _build_render_inputs(text: str) -> tuple[bytes, CaptionPlan, CaptionStyle]:
    style = CaptionStyle(font_family=FONT_FAMILY)
    tokens = text.split(" ")
    words = [
        {"index": index, "text": token, "start": index * 0.5, "end": index * 0.5 + 0.4}
        for index, token in enumerate(tokens)
    ]
    plan = build_caption_plan(
        [(0, 0.0, len(tokens) * 0.5, 0, len(tokens) - 1, "HERO")],
        words,
        style,
        Stage51Config(),
    )
    return serialize_ass(plan, style, _safe_zone()), plan, style


def _render_png(ass_bytes: bytes, destination: Path) -> None:
    ass_path = destination.with_suffix(".ass")
    ass_path.write_bytes(ass_bytes)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={WIDTH}x{HEIGHT}:d=1",
        "-vf",
        f"ass={ass_path}",
        "-frames:v",
        "1",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)


def _decode_png(data: bytes) -> tuple[int, int, int, bytes]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position = 8
    width = height = bit_depth = color_type = 0
    compressed = bytearray()
    while position < len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        chunk_type = data[position + 4 : position + 8]
        chunk = data[position + 8 : position + 8 + length]
        position += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
            assert bit_depth == 8
            assert interlace == 0
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    raw = zlib.decompress(bytes(compressed))
    stride = width * channels
    unfiltered = bytearray(height * stride)
    previous = bytearray(stride)
    offset = 0
    for row in range(height):
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset : offset + stride])
        offset += stride
        _unfilter_line(line, previous, filter_type, channels)
        start = row * stride
        unfiltered[start : start + stride] = line
        previous = line
    return width, height, channels, bytes(unfiltered)


def _unfilter_line(line: bytearray, previous: bytearray, filter_type: int, bpp: int) -> None:
    if filter_type == 0:
        return
    if filter_type == 1:
        for index in range(bpp, len(line)):
            line[index] = (line[index] + line[index - bpp]) & 0xFF
        return
    if filter_type == 2:
        for index in range(len(line)):
            line[index] = (line[index] + previous[index]) & 0xFF
        return
    if filter_type == 3:
        for index in range(len(line)):
            left = line[index - bpp] if index >= bpp else 0
            line[index] = (line[index] + ((left + previous[index]) >> 1)) & 0xFF
        return
    if filter_type == 4:
        for index in range(len(line)):
            left = line[index - bpp] if index >= bpp else 0
            up = previous[index]
            up_left = previous[index - bpp] if index >= bpp else 0
            estimate = left + up - up_left
            distance_left = abs(estimate - left)
            distance_up = abs(estimate - up)
            distance_up_left = abs(estimate - up_left)
            if distance_left <= distance_up and distance_left <= distance_up_left:
                predictor = left
            elif distance_up <= distance_up_left:
                predictor = up
            else:
                predictor = up_left
            line[index] = (line[index] + predictor) & 0xFF


def _ink_stats(data: bytes) -> tuple[int, tuple[int, int, int, int] | None]:
    width, height, channels, pixels = _decode_png(data)
    array = cast(
        "npt.NDArray[np.uint8]",
        np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, channels),
    )
    rgb = array[:, :, :3].astype(np.int32)
    background = rgb[0, 0]
    mask = np.abs(rgb - background).sum(axis=2) > 40
    ink = int(mask.sum())
    if ink == 0:
        return 0, None
    ys, xs = np.nonzero(mask)
    return ink, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _dialogue_payloads(ass_text: str) -> list[str]:
    payloads: list[str] = []
    for line in ass_text.splitlines():
        if line.startswith("Dialogue: "):
            payloads.append(line.split(",", 9)[9])
    return payloads


def _visible_text(payload: str) -> str:
    text = re.sub(r"(?<!\\)\{[^}]*\}", "", payload)
    text = text.replace("\\N", " ").replace("\\n", " ")
    text = text.replace("\\{", "{").replace("\\}", "}")
    return text.replace(_WORD_JOINER, "")


def _assert_no_bidi_controls(text: str) -> None:
    assert all(ord(character) not in _BIDI_CONTROL_CODEPOINTS for character in text)


def _naive_ass_document(raw_text: str) -> bytes:
    style = CaptionStyle()
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {WIDTH}\n"
        f"PlayResY: {HEIGHT}\n"
        "WrapStyle: 2\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: CaptionLower,{FONT_FAMILY},{style.font_size},{style.primary_color},"
        f"{style.secondary_color},{style.outline_color},&H00000000,"
        f"0,0,0,0,100,100,{style.spacing},0,1,{style.outline_width},{style.shadow},"
        f"2,{_safe_zone().left_px()},{_safe_zone().right_px()},"
        f"{_safe_zone().bottom_px() + _safe_zone().caption_bottom_gap_px},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 0,0:00:00.00,0:00:02.00,CaptionLower,,0,0,0,,{{\\an2}}{raw_text}\n"
    )
    return header.encode("utf-8")


def _iter_dialogue_payloads(ass_text: str) -> list[str]:
    return _dialogue_payloads(ass_text)


@pytest.mark.parametrize("name", sorted(FIXTURES))  # type: ignore[untyped-decorator]
def test_real_libass_renders_fixture_with_ink_and_determinism(name: str, tmp_path: Path) -> None:
    text = FIXTURES[name]
    ass_bytes, plan, _ = _build_render_inputs(text)
    ass_text = ass_bytes.decode("utf-8")
    _assert_no_bidi_controls(ass_text)

    assert " ".join(event.text for event in plan.events) == text
    payloads = list(_iter_dialogue_payloads(ass_text))
    # Distinct dialogue texts: every per-word state of one event repeats the same
    # visual text, so dedupe before checking the token multiset.
    distinct: list[str] = []
    for payload in payloads:
        textline = _visible_text(payload)
        if textline and textline not in distinct:
            distinct.append(textline)
    assert distinct
    # The derived ASS must be a pure permutation of the canonical tokens: no
    # fabrication, no loss, no injected controls. (Actual visual order is
    # asserted separately by rendering the frames.)
    canonical_tokens = sorted(token for event in plan.events for token in event.text.split(" "))
    visible_tokens = sorted(token for textline in distinct for token in textline.split(" "))
    assert visible_tokens == canonical_tokens

    first = tmp_path / f"{name}-1.png"
    second = tmp_path / f"{name}-2.png"
    _render_png(ass_bytes, first)
    _render_png(ass_bytes, second)

    assert first.read_bytes() == second.read_bytes()
    ink, bounding_box = _ink_stats(first.read_bytes())
    assert ink > 0
    assert bounding_box is not None


def test_representative_mixed_string_renders_ink(tmp_path: Path) -> None:
    ass_bytes, plan, _ = _build_render_inputs(REPRESENTATIVE)
    assert " ".join(event.text for event in plan.events) == REPRESENTATIVE
    assert plan.events[0].word_start_index == 0
    assert plan.events[-1].word_end_index == len(REPRESENTATIVE.split(" ")) - 1

    destination = tmp_path / "representative.png"
    _render_png(ass_bytes, destination)
    ink, _ = _ink_stats(destination.read_bytes())
    assert ink > 0
    _assert_no_bidi_controls(ass_bytes.decode("utf-8"))


def test_naive_character_reversed_render_differs(tmp_path: Path) -> None:
    correct_ass, _, _ = _build_render_inputs(REPRESENTATIVE)
    reversed_ass, _, _ = _build_render_inputs(REPRESENTATIVE[::-1])

    correct_png = tmp_path / "correct.png"
    reversed_png = tmp_path / "reversed.png"
    _render_png(correct_ass, correct_png)
    _render_png(reversed_ass, reversed_png)

    assert correct_png.read_bytes() != reversed_png.read_bytes()


def test_escaped_drawing_command_does_not_execute(tmp_path: Path) -> None:
    hostile = "{\\p1}m 0 0 l 800 0 800 800 0 800{\\p0}"
    escaped_ass, _, _ = _build_render_inputs(hostile)
    naive_ass = _naive_ass_document(hostile)

    escaped_png = tmp_path / "escaped.png"
    naive_png = tmp_path / "naive.png"
    _render_png(escaped_ass, escaped_png)
    _render_png(naive_ass, naive_png)

    escaped_ink, _ = _ink_stats(escaped_png.read_bytes())
    naive_ink, _ = _ink_stats(naive_png.read_bytes())

    assert escaped_ink > 0
    assert naive_ink > escaped_ink * 4


def _render_png_at(ass_bytes: bytes, source_time: float, destination: Path) -> None:
    ass_path = destination.with_suffix(".ass")
    ass_path.write_bytes(ass_bytes)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-copyts",
        "-ss",
        f"{source_time:.4f}",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={WIDTH}x{HEIGHT}:d=8:r=10,format=rgb24",
        "-frames:v",
        "1",
        "-vf",
        f"ass={ass_path},format=rgb24",
        "-pix_fmt",
        "rgb24",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)


def _rgb(data: bytes) -> "npt.NDArray[np.int32]":
    width, height, channels, pixels = _decode_png(data)
    array = cast(
        "npt.NDArray[np.uint8]",
        np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, channels),
    )
    return array[:, :, :3].astype(np.int32)


def _ink_mask(data: bytes) -> "npt.NDArray[np.bool_]":
    rgb = _rgb(data)
    return np.abs(rgb - rgb[0, 0]).max(axis=2) > 30


def _accent_mask(data: bytes) -> "npt.NDArray[np.bool_]":
    rgb = _rgb(data)
    return (rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 180) & (rgb[:, :, 2] < 120)


def _mask_bbox(mask: "npt.NDArray[np.bool_]") -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "name",
    [
        "arabic_only",
        "english_only",
        "arabic_with_embedded_english",
        "arabic_english_numbers",
        "english_with_arabic_phrase",
        "representative",
    ],
)
def test_real_libass_dynamic_highlight_moves_without_reflow_or_reshaping(
    name: str, tmp_path: Path
) -> None:
    text = FIXTURES[name]
    ass_bytes, plan, _ = _build_render_inputs(text)
    event = max(plan.events, key=lambda candidate: len(candidate.word_timings))
    if len(event.word_timings) < 2:
        pytest.skip("fixture produced no multi-word caption event")
    if not active_emphasis_safe(event.text):
        # Per-token override blocks reorder RTL/mixed words in the installed
        # libass; the serializer must degrade to one static phrase state.
        static_plan = CaptionStyle(font_family=FONT_FAMILY)
        static_ass = serialize_ass(plan, static_plan, _safe_zone()).decode("utf-8")
        dialogues = _dialogue_payloads(static_ass)
        assert dialogues
        assert all("\\c&H0000FFFF" not in dialogue for dialogue in dialogues)
        return

    bounds = [event.start, *[word.start for word in event.word_timings[1:]], event.end]
    midpoints = [
        (bounds[index] + bounds[index + 1]) / 2.0 for index in range(len(event.word_timings))
    ]
    static_ass = serialize_ass(
        plan, CaptionStyle(font_family=FONT_FAMILY, active_emphasis=False), _safe_zone()
    )
    # Same per-word state lines, but with the "active" color set to the base
    # primary color, so the only difference from a plain static render is the
    # override-tag syntax itself.
    no_op_ass = serialize_ass(
        plan,
        CaptionStyle(font_family=FONT_FAMILY, active_color=CaptionStyle().primary_color),
        _safe_zone(),
    )

    accent_masks: list[npt.NDArray[np.bool_]] = []
    boxes: list[tuple[int, int, int, int] | None] = []
    for index, midpoint in enumerate(midpoints):
        png = tmp_path / f"{name}-{index}.png"
        _render_png_at(ass_bytes, midpoint, png)
        ink, _ = _ink_stats(png.read_bytes())
        assert ink > 0
        accent_masks.append(_accent_mask(png.read_bytes()))
        boxes.append(_mask_bbox(_ink_mask(png.read_bytes())))

        # The override tags must not reflow, reorder, or reshape the script:
        # with a no-op color they render pixel-identically to the plain static
        # render of the same event.
        no_op_png = tmp_path / f"{name}-{index}-noop.png"
        static_png = tmp_path / f"{name}-{index}-static.png"
        _render_png_at(no_op_ass, midpoint, no_op_png)
        _render_png_at(static_ass, midpoint, static_png)
        assert np.array_equal(_rgb(no_op_png.read_bytes()), _rgb(static_png.read_bytes()))

    # The caption block never moves (within a couple of pixels of antialias
    # noise at the caption edge; a bouncing block would move by tens of pixels).
    reference_box = boxes[0]
    assert reference_box is not None
    for box in boxes:
        assert box is not None
        assert all(abs(box[edge] - reference_box[edge]) <= 2 for edge in range(4))
    # Every state shows exactly one accent, and the accent advances.
    assert all(bool(mask.any()) for mask in accent_masks)
    assert len({mask.tobytes() for mask in accent_masks}) >= 2


VISUAL_FIXTURES: list[tuple[str, str]] = [
    (
        "أنا كنت content creator لمدة سنتين",
        "سنتين لمدة content creator كنت أنا",
    ),
    (
        "أنا بستخدم Python و Django كل يوم",
        "يوم كل Django و Python بستخدم أنا",
    ),
    ("السعر 150 EGP بس", "بس 150 EGP السعر"),
    ("جربت GPT-5 وبعدها رجعت للشغل", "للشغل رجعت وبعدها GPT-5 جربت"),
    ("This is اختبار بسيط with English", "This is بسيط اختبار with English"),
    ("أنا سعيد، لأن Python رائع!", "رائع! Python لأن سعيد، أنا"),
    ("خصم 20% على السعر", "السعر على 20% خصم"),
]


def _single_event_plan(text: str, style: CaptionStyle) -> CaptionPlan:
    """Test-only: force one caption event to exercise per-line visual ordering.

    Production chunking keeps every event to at most two lines; this helper
    deliberately holds one event so the order test can inspect each line of the
    whole sentence in a single render. It is not product output.
    """

    tokens = text.split(" ")
    zone = _safe_zone()
    lines, _ = wrap_text(text, style, _spans_for_max_width(style, zone))
    timings = tuple(
        CaptionWordTiming(index=index, text=token, start=index * 0.6, end=index * 0.6 + 0.45)
        for index, token in enumerate(tokens)
    )
    event = CaptionEvent(
        event_id="visual-fixture",
        block_index=0,
        word_start_index=0,
        word_end_index=len(tokens) - 1,
        start=0.0,
        end=len(tokens) * 0.6,
        text=text,
        lines=lines,
        placement_zone="LOWER",
        placement_reason="DEFAULT_LOWER_ZONE",
        word_timings=timings,
    )
    return CaptionPlan(policy_version="visual-fixture", events=(event,))


def _logical_line_tokens(event: CaptionEvent) -> tuple[list[str], list[list[str]]]:
    tokens = event.text.split(" ")
    counts = [len(line.split(" ")) for line in event.lines]
    if event.lines and sum(counts) == len(tokens):
        lines: list[list[str]] = []
        cursor = 0
        for count in counts:
            lines.append(tokens[cursor : cursor + count])
            cursor += count
        return tokens, lines
    return tokens, [tokens]


#: Explicitly known reading order (visual left-to-right) for fixtures whose
#: glyphs are fully distinguishable. These are hand-written review anchors, not
#: derived from the implementation.
KNOWN_VISUAL_ORDER: dict[str, str] = {
    "كتير قوي ألماص": "ألماص قوي كتير",
    "طلباتهم كتير": "كتير طلباتهم",
}


def _trimmed_column_profile(mask: "npt.NDArray[np.bool_]") -> "npt.NDArray[np.float64] | None":
    profile = mask.sum(axis=0).astype(np.float64)
    xs = np.nonzero(profile)[0]
    if len(xs) == 0:
        return None
    return profile[int(xs.min()) : int(xs.max()) + 1]


def _locate_word_center(
    line_mask: "npt.NDArray[np.bool_]", word_mask: "npt.NDArray[np.bool_]"
) -> float:
    line = line_mask.sum(axis=0).astype(np.float64)
    word = word_mask.sum(axis=0).astype(np.float64)
    xs = np.nonzero(word)[0]
    assert len(xs) > 0
    word = word[int(xs.min()) : int(xs.max()) + 1]
    best_offset = 0
    best_score = -1.0
    word_energy = float((word * word).sum())
    for offset in range(0, len(line) - len(word) + 1):
        segment = line[offset : offset + len(word)]
        denominator = float(np.sqrt((segment * segment).sum() * word_energy))
        score = float((segment * word).sum() / denominator) if denominator else 0.0
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_offset + len(word) / 2.0


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("text", "expected"), sorted(KNOWN_VISUAL_ORDER.items())
)
def test_static_logical_emission_has_known_reading_order(
    text: str, expected: str, tmp_path: Path
) -> None:
    """Static captions emit logical text; libass owns the reading order.

    The production render must be pixel-identical to an independent test-local
    ASS document that writes the same logical text with no token overrides. The
    old manual permutation (what the failed acceptance MP4 contained) must
    render differently. Word positions are recovered from isolated-word rasters
    matched against the full line, then compared to a hand-written reading order.
    """

    style = CaptionStyle(font_family=FONT_FAMILY)
    plan = _single_event_plan(text, style)
    ass_bytes = serialize_ass(plan, style, _safe_zone())
    production = tmp_path / "production.png"
    _render_png(ass_bytes, production)

    reference = tmp_path / "reference.png"
    _render_png(_naive_ass_document(text), reference)
    assert np.array_equal(_rgb(production.read_bytes()), _rgb(reference.read_bytes()))

    base = paragraph_direction(text)
    manual_text = " ".join(visual_tokens(text.split(" "), base))
    manual = tmp_path / "manual.png"
    _render_png(_naive_ass_document(manual_text), manual)
    assert not np.array_equal(_rgb(production.read_bytes()), _rgb(manual.read_bytes()))

    line_mask = _ink_mask(production.read_bytes())
    centers: list[tuple[float, str]] = []
    for token in text.split(" "):
        word_plan = _single_event_plan(token, style)
        word_ass = serialize_ass(word_plan, style, _safe_zone())
        word_png = tmp_path / f"word-{len(centers)}.png"
        _render_png(word_ass, word_png)
        word_mask = _ink_mask(word_png.read_bytes())
        centers.append((_locate_word_center(line_mask, word_mask), token))

    rendered = [token for _, token in sorted(centers)]
    assert rendered == expected.split(" ")


def test_rtl_active_emphasis_degrades_to_one_static_phrase_state(tmp_path: Path) -> None:
    """RTL/mixed scripts must not receive per-token color overrides.

    The installed libass moves words when RTL runs are split by override blocks
    (confirmed on the failed production asset), so active emphasis degrades to a
    single static phrase state. The rendered static frame is also verified,
    rather than trusting the serializer's own token list.
    """

    text = "كتير قوي ألماص"
    style = CaptionStyle(font_family=FONT_FAMILY, active_emphasis=True)
    plan = _single_event_plan(text, style)
    dialogues = _dialogue_payloads(serialize_ass(plan, style, _safe_zone()).decode("utf-8"))

    assert len(dialogues) == 1
    assert "\\c&H0000FFFF" not in dialogues[0]
    assert " ".join(_visible_text(dialogues[0]).split(" ")) == text

    png = tmp_path / "rtl-static.png"
    _render_png(serialize_ass(plan, style, _safe_zone()), png)
    assert _ink_stats(png.read_bytes())[0] > 0


def test_ltr_active_emphasis_keeps_stable_per_word_geometry(tmp_path: Path) -> None:
    """Known-safe LTR scripts retain working active emphasis without reflow."""

    text = "hello distinct words"
    style = CaptionStyle(font_family=FONT_FAMILY, active_emphasis=True)
    plan = _single_event_plan(text, style)
    event = plan.events[0]
    bounds = [event.start, *[word.start for word in event.word_timings[1:]], event.end]
    ass_bytes = serialize_ass(plan, style, _safe_zone())

    boxes: list[tuple[int, int, int, int] | None] = []
    for index, _word in enumerate(event.word_timings):
        midpoint = (bounds[index] + bounds[index + 1]) / 2.0
        png = tmp_path / f"state-{index}.png"
        _render_png_at(ass_bytes, midpoint, png)
        accent = _accent_mask(png.read_bytes())
        assert bool(accent.any())
        boxes.append(_mask_bbox(_ink_mask(png.read_bytes())))

    reference = boxes[0]
    assert reference is not None
    for box in boxes:
        assert box is not None
        assert all(abs(box[edge] - reference[edge]) <= 2 for edge in range(4))


def _render_mp4_frame(ass_text: str, destination: Path, seconds: float = 0.32) -> None:
    """Encode the ASS through libx264 and decode one representative frame."""

    ass_path = destination.with_suffix(".ass")
    ass_path.write_text(ass_text, encoding="utf-8")
    mp4 = destination.with_suffix(".mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={WIDTH}x{HEIGHT}:r=25:d=2",
            "-vf",
            f"ass={ass_path}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{seconds:.3f}",
            "-i",
            str(mp4),
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "text", ["كتير قوي ألماص", "طلباتهم كتير"]
)
def test_encoded_mp4_preserves_static_logical_order(text: str, tmp_path: Path) -> None:
    style = CaptionStyle(font_family=FONT_FAMILY)
    plan = _single_event_plan(text, style)
    ass_text = serialize_ass(plan, style, _safe_zone()).decode("utf-8")

    png = tmp_path / "png.png"
    _render_png(serialize_ass(plan, style, _safe_zone()), png)
    mp4_frame = tmp_path / "mp4.png"
    _render_mp4_frame(ass_text, mp4_frame)

    fixed_png = _rgb(png.read_bytes())
    encoded = _rgb(mp4_frame.read_bytes())

    base = paragraph_direction(text)
    old_text = " ".join(visual_tokens(text.split(" "), base))
    old_frame = tmp_path / "old.png"
    _render_mp4_frame(_naive_ass_document(old_text).decode("utf-8"), old_frame)
    old_png = _rgb(old_frame.read_bytes())

    # Lossy encoding changes pixels; the encoded layout must still be much closer
    # to the logical reference than to the old manual permutation.
    to_logical = float(np.abs(encoded - fixed_png).mean())
    to_manual = float(np.abs(encoded - old_png).mean())
    assert to_logical < to_manual
    assert to_logical < 20.0
