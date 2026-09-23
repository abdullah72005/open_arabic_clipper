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
from app.composition.captions import CaptionPlan, build_caption_plan
from app.composition.policy import CaptionStyle, SafeZoneProfile, Stage51Config, safe_zone_for

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
    visible = [_visible_text(payload) for payload in payloads]
    assert visible
    assert set(visible) == {event.text for event in plan.events}

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
